"""
Servidor de embeddings + RAG (Flask + llama-cpp-python + Chroma).

Al arrancar: carga el modelo de embeddings, lee base_datos/preguntas.jsonl,
calcula los vectores e indexa todo en ChromaDB (colección "preguntas").

Uso:
    python embed_server.py

Endpoints:
    POST /v1/embeddings  {"input": [...]}                  -> vectores crudos
    POST /pregunta       {"query": "...", "n_results": 2}  -> preguntas + cara + respuesta
"""

import os
import json
from flask import Flask, request, jsonify
from llama_cpp import Llama
import chromadb

HERE = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(HERE)
MODEL_PATH = os.path.join(RAIZ, "models", "nomic-embed-text.gguf")
PREGUNTAS_FILE = os.path.join(RAIZ, "base_datos", "preguntas.jsonl")
CHROMA_PATH = os.path.join(RAIZ, "base_datos", "chroma_data")
PORT = 8081

app = Flask(__name__)
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False)

db = chromadb.PersistentClient(path=CHROMA_PATH)
coll = db.get_or_create_collection("preguntas")


def _embed(textos):
    return [item["embedding"] for item in llm.create_embedding(textos)["data"]]


def _indexar_preguntas():
    """Lee preguntas.jsonl y lo indexa en ChromaDB. Si ya está indexado, no repite."""
    if coll.count() > 0:
        print(f"Preguntas ya indexadas ({coll.count()}). Saltando.")
        return

    with open(PREGUNTAS_FILE, encoding="utf-8") as f:
        preguntas = [json.loads(l) for l in f if l.strip()]

    if not preguntas:
        print("preguntas.jsonl está vacío, nada que indexar.")
        return

    coll.add(
        documents=[p["pregunta"] for p in preguntas],
        embeddings=_embed([p["pregunta"] for p in preguntas]),
        ids=[p["id"] for p in preguntas],
        metadatas=[{"cara": p.get("cara", "Neutral"),
                    "respuesta_esperada": p.get("respuesta_esperada", "")} for p in preguntas],
    )
    print(f"Preguntas indexadas: {len(preguntas)}.")


@app.route("/v1/embeddings", methods=["POST"])
def embeddings_endpoint():
    body = request.get_json()
    textos = body["input"]
    if isinstance(textos, str):
        textos = [textos]
    vectores = _embed(textos)
    data = [{"embedding": v, "index": i} for i, v in enumerate(vectores)]
    return jsonify({"data": data, "model": "nomic-embed-text"})


@app.route("/pregunta", methods=["POST"])
def pregunta_endpoint():
    body = request.get_json()
    query = body["query"]
    n_results = body.get("n_results", 2)
    resultado = coll.query(query_embeddings=_embed([query]), n_results=n_results)
    docs = resultado["documents"][0]
    metas = resultado["metadatas"][0]
    preguntas = [
        {"pregunta": d,
         "cara": m.get("cara", "Neutral"),
         "respuesta_esperada": m.get("respuesta_esperada", "")}
        for d, m in zip(docs, metas)
    ]
    return jsonify({"preguntas": preguntas})


if __name__ == "__main__":
    _indexar_preguntas()
    print(f"Servidor escuchando en 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
