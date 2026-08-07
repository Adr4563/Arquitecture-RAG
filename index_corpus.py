"""
Indexa corpus.txt en ChromaDB, a mano. Corre esto vos mismo cuando quieras
(re)indexar — chat.py ya NO lo hace solo.

Requiere embed_server.py corriendo (ver embeddings.py para la IP/puerto).

Uso:
    python index_corpus.py
"""

import embeddings

if __name__ == "__main__":
    print("Indexando corpus.txt...")
    embeddings.indexar_corpus()
    print("Listo. Corpus indexado en ChromaDB.")
