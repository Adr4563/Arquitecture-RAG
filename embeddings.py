"""
Cliente de embeddings — todo lo relacionado a calcular vectores e indexar el
corpus en ChromaDB vive aca, separado del modelo de chat (ver llama_chat.py).

Habla con embed_server.py, que corre en otra maquina de la red (Flask +
llama-cpp-python, modelo nomic-embed-text).
"""

import chromadb
import requests

EMBED_SERVER_HOST = "http://192.168.1.44:8081"  # embed_server.py — cambia la IP si se reasigna por DHCP
CORPUS_FILE = "corpus.txt"

_db = chromadb.Client()
coll = _db.get_or_create_collection("pipeline_docs")


def embed(texts):
    resp = requests.post(
        f"{EMBED_SERVER_HOST}/v1/embeddings",
        json={"input": texts},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [item["embedding"] for item in data]


def cargar_corpus(ruta=CORPUS_FILE):
    """Lee el corpus del RAG desde un archivo de texto: una línea = un documento.
    Para cambiar de tema, edita ese archivo — no hace falta tocar este script."""
    with open(ruta, encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


def indexar_corpus():
    """Calcula embeddings del corpus y los carga en ChromaDB. Se llama una
    sola vez al iniciar chat.py."""
    docs = cargar_corpus()
    coll.add(
        documents=docs,
        embeddings=embed(docs),
        ids=[f"doc-{i}" for i in range(len(docs))],
    )


def recuperar_contexto(mensaje_usuario, n_results=2):
    """Devuelve el texto de los n_results documentos mas relevantes para el
    mensaje del usuario, ya concatenados en un solo string."""
    resultado = coll.query(query_embeddings=embed([mensaje_usuario]), n_results=n_results)
    return "\n".join(resultado["documents"][0])
