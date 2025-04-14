import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import (
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

# 🔧 Environment optimization
os.environ["FLASH_ATTENTION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 🚀 Model and data paths
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
dataset_paths = {
    "train": "/home/ayman/workspace/LLM_Project/Training_Data/training_data/train.jsonl",
    "validation": "/home/ayman/workspace/LLM_Project/Training_Data/training_data/eval.jsonl"
}

# 🧐 Load dataset
dataset = load_dataset("json", data_files=dataset_paths)

# 💥 Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 🧺 Load model + tokenizer (Unsloth + QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    quantization_config=bnb_config,
    device_map="auto",
)

# 🔧 Apply QLoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    use_gradient_checkpointing=True,
)

# 🧾 Format training examples
def format_example(example):
    return {
        "text": f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['output']}"
    }

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# ✂️ Tokenization
def tokenize_and_format(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=1024
    )

tokenized_dataset = dataset.map(
    tokenize_and_format,
    batched=True,
    num_proc=4,
    remove_columns=["text"]
)

# 🔁 Causal LM data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ⚙️ Training config
save_path = "/home/ayman/workspace/LLM_Project/qlora_finetuned_model"
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2, 
    gradient_accumulation_steps=6,  
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    learning_rate=5e-4,
    bf16=False,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=False,
    warmup_steps=100,
    max_grad_norm=0.3,
    save_strategy="steps",
    save_steps=1000,
    dataloader_num_workers=4,
    report_to="none",
)

# 🚀 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 🔥 Activate training mode
FastLanguageModel.for_training(model)
torch.set_float32_matmul_precision('high')

# 🏁 Train
trainer.train()

# 💾 Save
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)