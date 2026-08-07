# Imagen con Ollama + el modelo llama3.2:3b-q4s (cuantizado) + qwen3-embedding
# ya registrados adentro. docker run levanta todo listo, sin descargar ni
# registrar nada en la máquina destino.

FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ENV OLLAMA_FLASH_ATTENTION=1
ENV OLLAMA_KV_CACHE_TYPE=q8_0
ENV OLLAMA_KEEP_ALIVE=-1

RUN ollama serve & \
    sleep 5 && \
    ollama pull qwen3-embedding:4b && \
    curl -L -o /tmp/llama3.2-3b-q4_K_S.gguf \
      https://github.com/Adr4563/Arquitecture-RAG/releases/download/llama3.2-3b-q4_K_S/llama3.2-3b-q4_K_S.gguf && \
    printf 'FROM /tmp/llama3.2-3b-q4_K_S.gguf\n' > /tmp/Modelfile && \
    ollama create llama3.2:3b-q4s -f /tmp/Modelfile && \
    rm /tmp/llama3.2-3b-q4_K_S.gguf /tmp/Modelfile

EXPOSE 11434
ENTRYPOINT ["ollama", "serve"]
