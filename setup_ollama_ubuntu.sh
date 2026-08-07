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

echo "Descargando modelos..."
ollama pull llama3.2:3b
ollama pull qwen3-embedding:4b
