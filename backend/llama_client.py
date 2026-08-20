"""
Clientes HTTP hacia los servidores de modelos.

  - recuperar_contexto(): habla con embed_server.py (embeddings + ChromaDB)
  - generar_respuesta():  habla con Ollama (API compatible con OpenAI)

Los workers y el manager importan estas funciones, no hablan directo
con los servidores.
"""

import json

import requests

# ─── Servidores ──────────────────────────────────────────────
EMBED_SERVER_HOST = "http://localhost:8081"  # embed_server.py (embeddings + ChromaDB)
CHAT_SERVER_HOST = "http://localhost:11434"  # Ollama (API compatible con OpenAI)
CHAT_MODEL = "llama3.2:3b"  # nombre del modelo en `ollama list`


SIN_CONTEXTO = None  # sentinel: no hubo resultados relevantes en la base de datos


def recuperar_contexto(mensaje_usuario, n_results=2):
    resp = requests.post(
        f"{EMBED_SERVER_HOST}/pregunta",
        json={"query": mensaje_usuario, "n_results": n_results},
        timeout=60,
    )
    resp.raise_for_status()
    resultados = resp.json()["preguntas"]
    if not resultados:
        return SIN_CONTEXTO
    return "\n\n".join(
        f"Pregunta: {p['pregunta']}\n"
        f"Respuesta: {p['respuesta_esperada']}\n"
        f"Cara: {p['cara']}"
        for p in resultados
    )


def generar_respuesta(mensajes, temperature=0.3, max_tokens=100, on_token=None):
    """Llama al modelo con streaming. Si se pasa on_token(chunk), se invoca por cada
    pedazo de texto a medida que llega (para imprimirlo en vivo); igual devuelve el
    texto completo al final. No baja el tiempo total, pero se percibe mucho más
    rápido porque el usuario ve la respuesta aparecer en vez de esperar en blanco."""
    resp = requests.post(
        f"{CHAT_SERVER_HOST}/v1/chat/completions",
        json={
            "model": CHAT_MODEL,
            "messages": mensajes,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )
    resp.raise_for_status()
    # Ollama manda 'Content-Type: text/event-stream' sin charset. Ante un text/*
    # sin charset, requests asume ISO-8859-1 (regla legacy de HTTP) y con
    # decode_unicode=True decodifica los bytes UTF-8 como latin-1: "triángulo"
    # llega como "triÃ¡ngulo". El cuerpo es UTF-8, así que hay que decirlo.
    resp.encoding = "utf-8"
    texto = []
    for linea in resp.iter_lines(decode_unicode=True):
        if not linea or not linea.startswith("data: "):
            continue
        payload = linea[len("data: "):]
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        if delta:
            texto.append(delta)
            if on_token:
                on_token(delta)
    return "".join(texto)
