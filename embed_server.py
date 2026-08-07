"""
Servidor de embeddings + RAG, todo en uno (Flask + llama-cpp-python + Chroma).

Al arrancar: carga el modelo de embeddings, lee corpus.txt, calcula los
vectores e indexa todo en ChromaDB — automático, sin scripts aparte.

Corre en esta máquina (la que tiene los embeddings, no el chat). El modelo
llama de chat corre por separado en la Raspberry Pi (ver llama_chat.py).

Uso:
    python embed_server.py

Endpoints:
    POST /v1/embeddings  {"input": [...]}           -> vectores crudos
    POST /context         {"query": "...", "n_results": 2} -> contexto del RAG
"""

from flask import Flask, request, jsonify
from llama_cpp import Llama
import chromadb

MODEL_PATH = "models/nomic-embed-text.gguf"
CORPUS_FILE = "corpus.txt"
PORT = 8081

app = Flask(__name__)
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False)

db = chromadb.PersistentClient(path="./chroma_data")
coll = db.get_or_create_collection("pipeline_docs")


def _embed(textos):
    resultado = llm.create_embedding(textos)
    return [item["embedding"] for item in resultado["data"]]


def _indexar_corpus():
    """Lee corpus.txt y lo indexa en ChromaDB. Se llama automático al
    arrancar el servidor — si ya estaba indexado, no repite el trabajo."""
    if coll.count() > 0:
        print(f"Corpus ya indexado ({coll.count()} documentos). Saltando.")
        return

    with open(CORPUS_FILE, encoding="utf-8") as f:
        docs = [linea.strip() for linea in f if linea.strip()]

    if not docs:
        print("corpus.txt está vacío, nada que indexar.")
        return

    coll.add(
        documents=docs,
        embeddings=_embed(docs),
        ids=[f"doc-{i}" for i in range(len(docs))],
    )
    print(f"Corpus indexado: {len(docs)} documentos.")


@app.route("/v1/embeddings", methods=["POST"])
def embeddings_endpoint():
    body = request.get_json()
    textos = body["input"]
    if isinstance(textos, str):
        textos = [textos]

    vectores = _embed(textos)
    data = [{"embedding": v, "index": i} for i, v in enumerate(vectores)]
    return jsonify({"data": data, "model": "nomic-embed-text"})


@app.route("/context", methods=["POST"])
def context_endpoint():
    body = request.get_json()
    query = body["query"]
    n_results = body.get("n_results", 2)

    resultado = coll.query(query_embeddings=_embed([query]), n_results=n_results)
    contexto = "\n".join(resultado["documents"][0])
    return jsonify({"context": contexto})


if __name__ == "__main__":
    _indexar_corpus()
    print(f"Servidor de embeddings + RAG escuchando en 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
