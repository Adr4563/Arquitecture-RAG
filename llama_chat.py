"""
Cliente del modelo de chat (llama) — todo lo relacionado a hablar con el
llama-server que genera las respuestas vive aca, separado de los embeddings
(ver embeddings.py).

Habla con un llama-server (llama.cpp) local, corriendo en la Raspberry Pi.
"""

import requests

CHAT_SERVER_HOST = "http://localhost:8080"  # llama-server con el modelo de chat, local en la Pi


def generar_respuesta(mensajes, temperature=0.3, max_tokens=300):
    """Manda el historial de mensajes (formato OpenAI: [{"role": ..., "content": ...}])
    al llama-server y devuelve el texto de la respuesta."""
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
