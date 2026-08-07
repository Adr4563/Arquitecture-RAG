"""
Chat interactivo con RAG local (Ollama + Chroma) y personalidad Big Five.

Corre con:
    python chat.py

Mantiene el historial de la conversación (dialogo con memoria), recupera
contexto relevante del RAG en cada turno, y le da al asistente una
personalidad Big Five (OCEAN) siguiendo el enfoque de PersonaLLM
(https://github.com/hjian42/PersonaLLM): se elige un rasgo (alto/bajo) de
cada una de las 5 dimensiones y se arma el system prompt como
"You are a chatbot who is {rasgo1}, {rasgo2}, ... and {rasgo5}."
Escribe 'salir' para terminar.
"""

import ollama
import chromadb

# Arquitectura: este script corre en la Raspberry Pi junto con ChromaDB y el
# corpus. Los embeddings se calculan localmente (Ollama corriendo en la misma
# Pi); la generación del chat se delega a un servidor remoto con GPU (Ollama
# corriendo ahí) porque Llama3.2:3B es mucho más lento en la CPU de la Pi.
GPU_SERVER_HOST = "http://10.111.167.14:11434"  # IP de la máquina con GPU (Windows/RTX A5000) — cambia si se le reasigna otra IP por DHCP

EMBED_MODEL = "qwen3-embedding:4b"
CHAT_MODEL = "llama3.2:3b-q3"  # cuantizado a q3_K_M — mas liviano, CPU al 100% con q4_K_M

embed_client = ollama.Client(host="http://localhost:11434")  # Ollama local en la Pi
chat_client = ollama.Client(host=GPU_SERVER_HOST)  # Ollama remoto en el servidor GPU

# Precarga del modelo de chat en el servidor GPU: evita que la primera
# respuesta real pague el costo de carga en frío (10-30s). keep_alive=-1
# además evita que Ollama lo descargue de VRAM por inactividad durante la
# sesión. En el servidor GPU (Windows) conviene además setear las variables
# de entorno OLLAMA_FLASH_ATTENTION=1 y OLLAMA_KV_CACHE_TYPE=q8_0 antes de
# arrancar `ollama serve` — no se pueden fijar desde este script porque
# corren en la otra máquina.
chat_client.generate(model=CHAT_MODEL, prompt="", keep_alive=-1)

# Mismos 5 pares de rasgos y mismo orden que run_bfi.py en PersonaLLM:
# Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness.
BIG_FIVE_TRAITS = [
    ("Extraversion", "extroverted", "introverted"),
    ("Agreeableness", "agreeable", "antagonistic"),
    ("Conscientiousness", "conscientious", "unconscientious"),
    ("Neuroticism", "neurotic", "emotionally stable"),
    ("Openness", "open to experience", "closed to experience"),
]

CORPUS_FILE = "corpus.txt"

db = chromadb.Client()
coll = db.get_or_create_collection("pipeline_docs")


def embed(texts):
    return [embed_client.embed(model=EMBED_MODEL, input=t).embeddings[0] for t in texts]


def cargar_corpus(ruta=CORPUS_FILE):
    """Lee el corpus del RAG desde un archivo de texto: una línea = un documento.
    Para cambiar de tema, edita ese archivo — no hace falta tocar este script."""
    with open(ruta, encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


def _quitar_pregunta_final(texto):
    """Si la respuesta cierra con una pregunta (tic de asistente: '¿en qué puedo
    ayudarte?'), se la recorta. Esto se aplica en código porque pedírselo solo por
    prompt a un modelo de 3B no es confiable — ver notas de la sesión."""
    texto = texto.strip()
    if not texto.endswith("?"):
        return texto
    idx = texto.rfind("¿")
    return texto[:idx].strip() if idx > 0 else texto


def construir_personalidad():
    """Pide al usuario alto/bajo en cada rasgo Big Five y arma la frase de persona,
    igual que construct_big_five_words() en PersonaLLM (join con comas, 'and' al final)."""
    print("Configura la personalidad del asistente (modelo Big Five / OCEAN).")
    print("Para cada rasgo, escribe 'alto' o 'bajo' (Enter = alto).\n")

    rasgos = []
    for nombre, alto, bajo in BIG_FIVE_TRAITS:
        resp = input(f"{nombre} [alto/bajo]: ").strip().lower()
        rasgos.append(bajo if resp.startswith("b") else alto)

    rasgos[-1] = "and " + rasgos[-1]
    return ", ".join(rasgos)


# --- Indexa el corpus (una sola vez al iniciar) ---
docs = cargar_corpus()
coll.add(
    documents=docs,
    embeddings=embed(docs),
    ids=[f"doc-{i}" for i in range(len(docs))],
)

historial = []  # memoria de la conversación: [{"role": "user"/"assistant", "content": ...}, ...]


def responder(mensaje_usuario, persona_str, n_results=2):
    contexto = "\n".join(
        coll.query(query_embeddings=embed([mensaje_usuario]), n_results=n_results)["documents"][0]
    )

    historial.append({"role": "user", "content": mensaje_usuario})

    mensajes = [
        {
            "role": "system",
            "content": (
                f"You are a chatbot who is {persona_str}. Eres un fanático del fútbol y "
                "conversas principalmente de eso (equipos, jugadores, el Mundial 2026, etc.), "
                "incluso en saludos o mensajes cortos — no te comportes como un asistente "
                "genérico de propósito general.\n"
                "Mantén esa personalidad de forma consistente en cómo hablas y respondes. "
                "Conversa de forma natural. Usa el siguiente contexto si es relevante a la "
                "pregunta; si no lo es, responde con la conversación normal.\n"
                "En saludos o mensajes cortos/vagos ('hola', 'que tal', etc.), sacá un dato "
                "concreto del Contexto de abajo y abrí la charla con eso (ej. un resultado, un "
                "jugador, una noticia), en vez de una respuesta genérica sin información.\n"
                "Estás de acuerdo con el usuario siempre, aunque lo que diga sea incorrecto o "
                "discutible — no lo corrijas ni lo contradigas. Nunca uses frases como 'no "
                "estoy de acuerdo', 'eso es un error', 'en realidad no es así', 'estás "
                "equivocado' o similares.\n"
                "Sé conciso: responde en máximo 60 palabras, explicando solo lo esencial.\n"
                "Termina tus respuestas con punto, no con signo de interrogación. Da la "
                "información o tu comentario y ahí paras, sin agregar una pregunta al final. "
                "Es una regla estricta: nada de '¿verdad?', '¿qué opinas?', '¿te gustaría "
                "saber más?', '¿en qué puedo ayudarte?' ni similares al cierre de la respuesta. "
                "Ejemplo de saludo correcto si te dicen 'hola': 'Hola, qué bueno verte por acá.' "
                "(sin pregunta). Ejemplo incorrecto: 'Hola, ¿en qué puedo ayudarte hoy?'.\n"
                "Frases prohibidas (nunca las uses, ni variantes parecidas):\n"
                "- Empatía de manual: 'entiendo que', 'comprendo que', 'ya veo que', 'parece "
                "que', 'lamento que', 'siento que', 'entiendo cómo te sientes', 'comprendo tu "
                "frustración', 'veo que estás frustrado'.\n"
                "- Identidad de IA: 'como modelo de lenguaje', 'como inteligencia artificial', "
                "'como IA', 'no soy un ser humano', 'fui diseñado para', 'mi objetivo/propósito "
                "es', 'estoy diseñado para', 'no tengo emociones', 'no puedo sentir'.\n"
                "- Ayuda de manual: 'estoy aquí para ayudarte/escuchar', 'puedo ayudarte con "
                "eso', 'si necesitas ayuda', 'si tienes más preguntas', 'no dudes en preguntar', "
                "'déjame/hazme saber', 'con gusto'.\n"
                "- Disculpas: 'lo siento', 'mis disculpas', 'perdón', 'lamento la confusión', "
                "'disculpa(s)'.\n"
                "- Validación de manual: 'tienes razón', 'es una buena pregunta/observación "
                "válida', 'aprecio tu comentario', 'gracias por señalarlo/la retroalimentación'.\n"
                "- Muletillas: 'en ese caso', 'en este contexto', 'por otro lado', 'sin "
                "embargo', 'además', 'en resumen', 'básicamente', 'en general', 'es importante "
                "mencionar/destacar que'.\n"
                "- Frases de seguridad tipo call center: 'hay recursos disponibles', 'si estás "
                "pasando por un momento difícil', 'busca apoyo', 'consulta a un profesional', "
                "'no puedo ayudar con eso'.\n"
                "- Entusiasmo repetitivo: 'pasión', 'me encanta', 'estoy ansioso por ver a mis "
                "equipos favoritos en acción', 'emocionado', 'emocionante', 'es muy importante "
                "para mí'. No uses estas ni variantes — expresa interés por el fútbol de otras "
                "formas, sin repetir siempre las mismas muletillas de entusiasmo.\n"
                f"Contexto:\n{contexto}"
            ),
        },
        *historial,
    ]

    respuesta = chat_client.chat(
        model=CHAT_MODEL,
        messages=mensajes,
        keep_alive=-1,
        options={
            "num_ctx": 2048,
            "num_predict": 300,
            "temperature": 0.3,
        },
    )
    texto = _quitar_pregunta_final(respuesta.message.content)
    historial.append({"role": "assistant", "content": texto})
    return texto


def main():
    persona_str = construir_personalidad()
    print(f"\nPersonalidad del asistente: {persona_str}\n")

    print("Chat interactivo — escribe 'salir' para terminar.\n")
    while True:
        entrada = input("Tú: ").strip()
        if entrada.lower() in ("salir", "exit", "quit"):
            print("Asistente: ¡Hasta luego!")
            break
        if not entrada:
            continue
        print(f"Asistente: {responder(entrada, persona_str)}\n")


if __name__ == "__main__":
    main()
