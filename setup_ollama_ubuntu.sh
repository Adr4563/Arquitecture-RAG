#!/usr/bin/env bash
# Optimiza Ollama en Ubuntu: flash attention, cache KV cuantizada y keep_alive
# infinito, vía override de systemd. Correr en la máquina Ubuntu (ej. la
# Raspberry Pi o el servidor que reemplace al de Windows) después de clonar
# este repo.
#
# Uso:
#   chmod +x setup_ollama_ubuntu.sh
#   sudo ./setup_ollama_ubuntu.sh

set -euo pipefail

OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"

mkdir -p "$OVERRIDE_DIR"
cat > "$OVERRIDE_FILE" <<'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=-1"
EOF

systemctl daemon-reload
systemctl restart ollama

echo "Variables aplicadas:"
systemctl show ollama --property=Environment

echo "Descargando embeddings..."
ollama pull qwen3-embedding:4b

echo "Descargando llama3.2 ya cuantizado a q4_K_S (1.9GB, generado y subido a un Release del repo)..."
curl -L -o /tmp/llama3.2-3b-q4_K_S.gguf \
  https://github.com/Adr4563/Arquitecture-RAG/releases/download/llama3.2-3b-q4_K_S/llama3.2-3b-q4_K_S.gguf

cat > /tmp/Modelfile.q4s <<EOF
FROM /tmp/llama3.2-3b-q4_K_S.gguf
EOF
ollama create llama3.2:3b-q4s -f /tmp/Modelfile.q4s
rm /tmp/llama3.2-3b-q4_K_S.gguf /tmp/Modelfile.q4s
