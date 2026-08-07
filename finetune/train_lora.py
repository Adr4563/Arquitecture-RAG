"""
Fine-tuning LoRA de Llama 3.2 3B (el mismo modelo que usa chat.py vía Ollama),
adaptado del tutorial de Robert Schaeffer (Medium, "How to Fine-Tune Llama 3.1
8B") para correr en Ubuntu con GPU local en vez de Google Colab.

Edita finetune/training_data.json con tus propios ejemplos antes de correr
esto — los que vienen son solo placeholders de ejemplo.

Requiere: pip install -r finetune/requirements.txt
Requiere acceso al modelo en HuggingFace (huggingface_hub login) y una
cuenta con acceso a meta-llama/Llama-3.2-3B-Instruct.

Uso:
    python finetune/train_lora.py

Salida: adapter LoRA guardado en ./llama32-3b-finetuned/ (checkpoint final
de TrainingArguments.output_dir). Ese adapter se fusiona y exporta a GGUF
para Ollama con export_to_ollama.sh.
"""

import json
import os

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import notebook_login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # equivalente HF de llama3.2:3b en Ollama
OUTPUT_MODEL = "llama32-3b-finetuned"
TRAINING_DATA_FILE = os.path.join(os.path.dirname(__file__), "training_data.json")


def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def prepare_train_data(data):
    data_df = pd.DataFrame(data)
    data_df["text"] = data_df[["prompt", "response"]].apply(
        lambda x: "<|im_start|>user\n" + x["prompt"] + " <|im_end|>\n<|im_start|>assistant\n"
        + x["response"] + "<|im_end|>\n",
        axis=1,
    )
    return Dataset.from_pandas(data_df)


def main():
    # Login en HuggingFace (necesario para descargar el modelo si es gated)
    if not os.environ.get("HF_TOKEN"):
        notebook_login()

    with open(TRAINING_DATA_FILE, encoding="utf-8") as f:
        training_data = json.load(f)

    model, tokenizer = get_model_and_tokenizer(MODEL_ID)
    data = prepare_train_data(training_data)

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    training_arguments = TrainingArguments(
        output_dir=OUTPUT_MODEL,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=3,
        max_steps=250,
        fp16=True,
        push_to_hub=False,  # local, no se sube a HF
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=1024,
    )

    trainer.train()
    trainer.save_model(OUTPUT_MODEL)
    print(f"Adapter LoRA guardado en ./{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
