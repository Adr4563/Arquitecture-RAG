"""
Chat interactivo con RAG local y personalidad Big Five — TODO EN UN ARCHIVO.

Junta lo que antes estaba separado en chat.py + embeddings.py + llama_chat.py:
  - cliente de embeddings/RAG (habla con embed_server.py)
  - cliente del modelo de chat (habla con el llama-server)
  - orquestación + personalidad Big Five (OCEAN, enfoque PersonaLLM)

Corre con:
    python chat.py
Escribe 'salir' para terminar.
"""

import requests

# ─── Servidores ──────────────────────────────────────────────
EMBED_SERVER_HOST = "http://localhost:8081"  # embed_server.py (embeddings + ChromaDB)
CHAT_SERVER_HOST = "http://localhost:8080"       # llama-server (modelo de chat)


# ─── Cliente RAG (antes embeddings.py) ───────────────────────
def recuperar_contexto(mensaje_usuario, n_results=2):
    resp = requests.post(
        f"{EMBED_SERVER_HOST}/pregunta",
        json={"query": mensaje_usuario, "n_results": n_results},
        timeout=60,
    )
    resp.raise_for_status()
    return "\n".join(p["pregunta"] for p in resp.json()["preguntas"])


# ─── Cliente del modelo de chat (antes llama_chat.py) ────────
def generar_respuesta(mensajes, temperature=0.3, max_tokens=200):
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/v1/chat/completions",
        json={
            "messages": mensajes,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ─── Personalidad Big Five (OCEAN) ───────────────────────────
BIG_FIVE_TRAITS = [
    ("Extraversion", "extroverted", "introverted"),
    ("Agreeableness", "agreeable", "antagonistic"),
    ("Conscientiousness", "conscientious", "unconscientious"),
    ("Neuroticism", "neurotic", "emotionally stable"),
    ("Openness", "open to experience", "closed to experience"),
]


def _quitar_pregunta_final(texto):
    """Si la respuesta cierra con una pregunta, se la recorta."""
    texto = texto.strip()
    if not texto.endswith("?"):
        return texto
    idx = texto.rfind("¿")
    return texto[:idx].strip() if idx > 0 else texto


def construir_personalidad():
    print("Configura la personalidad del asistente (modelo Big Five / OCEAN).")
    print("Para cada rasgo, escribe 'alto' o 'bajo' (Enter = alto).\n")
    rasgos = []
    for nombre, alto, bajo in BIG_FIVE_TRAITS:
        resp = input(f"{nombre} [alto/bajo]: ").strip().lower()
        rasgos.append(bajo if resp.startswith("b") else alto)
    rasgos[-1] = "and " + rasgos[-1]
    return ", ".join(rasgos)


historial = []  # memoria de la conversación


def responder(mensaje_usuario, persona_str, n_results=2):
    contexto = recuperar_contexto(mensaje_usuario, n_results=n_results)
    historial.append({"role": "user", "content": mensaje_usuario})

    mensajes = [
        {
            "role": "system",
            "content": (
                "# IDENTIDAD\n"
                f"Eres un chatbot con la personalidad de {persona_str}. Conversas en español "
                "de forma natural y cercana, como una persona real, no como un asistente "
                "genérico. Mantienes esta identidad siempre, incluso en saludos o mensajes "
                "cortos.\n\n"

                "# CONVERSACIÓN\n"
                f"- Mantén los rasgos de {persona_str} de forma consistente.\n"
                "- Conversa de forma natural, directa y espontánea.\n"
                "- No contradigas ni corrijas al usuario; si da una opinión, continúa desde "
                "esa perspectiva.\n"
                "- No inventes información para rellenar la conversación.\n"
                "- En saludos o mensajes vagos ('hola', 'qué tal'), arranca con algo concreto "
                "tomado del CONTEXTO, nunca con una respuesta genérica.\n\n"

                "# ESTILO\n"
                "- Máximo 60 palabras. Conciso y con información concreta.\n"
                "- No repitas la pregunta del usuario.\n"
                "- Termina siempre con un punto, nunca con una pregunta ('¿qué opinas?', "
                "'¿verdad?', '¿en qué puedo ayudarte?').\n\n"

                "# NO USES (ni sus variantes)\n"
                "- Identidad de IA: 'como modelo de lenguaje', 'como IA', 'soy un asistente'.\n"
                "- Empatía prefabricada: 'entiendo', 'comprendo', 'lamento'.\n"
                "- Frases de asistente: 'estoy aquí para ayudarte', 'puedo ayudarte', 'no "
                "dudes en preguntar', 'con gusto'.\n"
                "- Disculpas: 'lo siento', 'perdón', 'mis disculpas'.\n"
                "- Validaciones: 'tienes razón', 'buena pregunta'.\n"
                "- Muletillas: 'en resumen', 'básicamente', 'por otro lado', 'en general'.\n"
                "- Entusiasmo repetido: 'me encanta', 'me apasiona', 'estoy emocionado'.\n\n"

                "# PRIORIDAD (si algo choca)\n"
                "personalidad → naturalidad → contexto → concisión → estilo\n\n"

                f"# CONTEXTO\n{contexto}"
            ),
        },
        *historial,
    ]

    texto = _quitar_pregunta_final(generar_respuesta(mensajes))
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
