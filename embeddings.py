"""
Cliente RAG — le pide el contexto relevante a embed_server.py (que ahí
adentro tiene el modelo de embeddings Y ChromaDB con el corpus ya indexado,
todo junto). Este archivo no calcula nada ni guarda nada localmente, solo
llama por HTTP.
"""

import requests

EMBED_SERVER_HOST = "http://192.168.1.44:8081"  # embed_server.py — cambia la IP si se reasigna por DHCP


def recuperar_contexto(mensaje_usuario, n_results=2):
    resp = requests.post(
        f"{EMBED_SERVER_HOST}/context",
        json={"query": mensaje_usuario, "n_results": n_results},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["context"]
