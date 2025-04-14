import unsloth
import os
import torch
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import SFTTrainer
from peft import LoraConfig
import json

# ---------------- CONFIG ----------------
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
NUM_EPOCHS = 3
LR = 2e-5
OUTPUT_DIR = "/home/ayman/workspace/LLM_Project/qlora_finetuned_model"

TRAIN_FILE = "/home/ayman/workspace/LLM_Project/Training_Data/training_data/train.jsonl"
EVAL_FILE = "/home/ayman/workspace/LLM_Project/Training_Data/training_data/eval.jsonl"

# ---------------- GPU / CPU SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {device} | GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'None'}")

# ---------------- Load Model with Unsloth ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Automatically selects optimal precision (bf16/float16)
    load_in_4bit=True,
)

# Enable optimizations
FastLanguageModel.for_training(model)

# ---------------- LoRA CONFIG ----------------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ---------------- Dataset Loader ----------------
def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

train_dataset = load_jsonl(TRAIN_FILE)
eval_dataset = load_jsonl(EVAL_FILE)

# ---------------- Supervised Fine-tuning ----------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="instruction",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=os.cpu_count(),
    packing=True,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        warmup_steps=50,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        learning_rate=LR,
        save_total_limit=2,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        dataloader_num_workers=os.cpu_count()
    ),
    peft_config=lora_config,
)

# ---------------- Training Kick-off ----------------
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model fine-tuned and saved to: {OUTPUT_DIR}")
