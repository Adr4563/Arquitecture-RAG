"""
Servidor de embeddings en Python puro (Flask + llama-cpp-python), sin Ollama
y sin depender del binario llama-server de llama.cpp.

Corre en esta máquina (la que tiene los embeddings, no el chat). El modelo
llama de chat corre por separado en la Raspberry Pi.

Uso:
    python embed_server.py

Expone POST /v1/embeddings compatible con el formato que espera chat.py:
    {"input": ["texto1", "texto2"]} -> {"data": [{"embedding": [...]}, ...]}
"""

from flask import Flask, request, jsonify
from llama_cpp import Llama

MODEL_PATH = "models/nomic-embed-text.gguf"
PORT = 8081

app = Flask(__name__)
llm = Llama(model_path=MODEL_PATH, embedding=True, verbose=False)


@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    body = request.get_json()
    textos = body["input"]
    if isinstance(textos, str):
        textos = [textos]

    resultado = llm.create_embedding(textos)
    data = [{"embedding": item["embedding"], "index": i} for i, item in enumerate(resultado["data"])]
    return jsonify({"data": data, "model": "nomic-embed-text"})


if __name__ == "__main__":
    print(f"Servidor de embeddings escuchando en 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT)
