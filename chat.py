"""
Chat interactivo con RAG local y personalidad Big Five.

Corre con:
    python chat.py

Mantiene el historial de la conversación (dialogo con memoria), recupera
contexto relevante del RAG en cada turno, y le da al asistente una
personalidad Big Five (OCEAN) siguiendo el enfoque de PersonaLLM
(https://github.com/hjian42/PersonaLLM): se elige un rasgo (alto/bajo) de
cada una de las 5 dimensiones y se arma el system prompt como
"You are a chatbot who is {rasgo1}, {rasgo2}, ... and {rasgo5}."
Escribe 'salir' para terminar.

Este archivo solo orquesta — no habla directo con ningun servidor. Los
embeddings/ChromaDB viven en embeddings.py, y el modelo de chat (llama) en
llama_chat.py. Cada uno corre en su propia maquina, por separado.
"""

import embeddings
import llama_chat

# Mismos 5 pares de rasgos y mismo orden que run_bfi.py en PersonaLLM:
# Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness.
BIG_FIVE_TRAITS = [
    ("Extraversion", "extroverted", "introverted"),
    ("Agreeableness", "agreeable", "antagonistic"),
    ("Conscientiousness", "conscientious", "unconscientious"),
    ("Neuroticism", "neurotic", "emotionally stable"),
    ("Openness", "open to experience", "closed to experience"),
]


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
embeddings.indexar_corpus()

historial = []  # memoria de la conversación: [{"role": "user"/"assistant", "content": ...}, ...]


def responder(mensaje_usuario, persona_str, n_results=2):
    contexto = embeddings.recuperar_contexto(mensaje_usuario, n_results=n_results)

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

    texto = _quitar_pregunta_final(llama_chat.generar_respuesta(mensajes))
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
