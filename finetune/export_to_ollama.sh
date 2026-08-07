#!/usr/bin/env bash
# Fusiona el adapter LoRA con el modelo base, lo convierte a GGUF y lo
# registra en Ollama con el mismo formato optimizado (q4_K_M) que el resto
# del repo. Correr en Ubuntu, después de train_lora.py.
#
# Requiere: llama.cpp clonado en ../llama.cpp (para convert_hf_to_gguf.py
# y llama-quantize). Si no lo tienes:
#   git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
#   cmake -B ../llama.cpp/build ../llama.cpp && cmake --build ../llama.cpp/build --config Release
#
# Uso:
#   ./export_to_ollama.sh

set -euo pipefail

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR="llama32-3b-finetuned"
MERGED_DIR="llama32-3b-merged"
GGUF_F16="llama32-3b-f16.gguf"
GGUF_QUANT="llama32-3b-q4_k_m.gguf"
OLLAMA_MODEL_NAME="llama3.2-finetuned"
LLAMA_CPP_DIR="../llama.cpp"

echo "1/4 Fusionando adapter LoRA con el modelo base..."
python - <<PYEOF
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("$ADAPTER_DIR", device_map="auto")
model = model.merge_and_unload()
model.save_pretrained("$MERGED_DIR")
AutoTokenizer.from_pretrained("$BASE_MODEL").save_pretrained("$MERGED_DIR")
PYEOF

echo "2/4 Convirtiendo a GGUF (FP16)..."
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" --outfile "$GGUF_F16" --outtype f16

echo "3/4 Cuantizando a q4_K_M (mismo formato recomendado en el artículo de optimización)..."
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_QUANT" q4_K_M

echo "4/4 Registrando el modelo en Ollama..."
cat > Modelfile <<EOF
FROM ./$GGUF_QUANT
PARAMETER num_ctx 2048
EOF
ollama create "$OLLAMA_MODEL_NAME" -f Modelfile

echo "Listo. Modelo disponible como '$OLLAMA_MODEL_NAME' en Ollama."
echo "Para usarlo en chat.py, cambia CHAT_MODEL = \"$OLLAMA_MODEL_NAME\""
