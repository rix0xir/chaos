import os
import json
import time
import torch
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import gc

# ------------------ Config ------------------ #
MODEL_DIR = "/home/ayman/workspace/LLM_Project/qlora_finetuned_model"
EVAL_FILE = "/home/ayman/workspace/LLM_Project/Training_Data/training_data/eval.jsonl"
RESULT_FILE = os.path.join(MODEL_DIR, "evaluation_results.jsonl")
# Dynamically estimate safe batch size for available VRAM
SAFE_BATCH_SIZE = 2  # Try 2 or 3 — adjust based on your 3060's real usage
MAX_INPUT_LEN = 1024
MAX_NEW_TOKENS = 1024
MAX_TOTAL_TOKENS = 2048  # Reduced to save memory
GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing to save memory

# Set PyTorch memory allocation to expandable segments to handle fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------ Set Random Seed ------------------ #
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ------------------ Load Model & Tokenizer ------------------ #
print("📥 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.padding_side = "left"

# Safety: Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with memory optimizations
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",
    low_cpu_mem_usage=True,     # Optimize CPU memory usage
)

# Enable gradient checkpointing to reduce memory footprint
if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

model.eval()

# Get the device that the model is on
device = next(model.parameters()).device

# ------------------ Prompt Formatter ------------------ #
def format_prompt(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )

# ------------------ Fuzzy Match Score ------------------ #
def fuzzy_match(a, b):
    return round(SequenceMatcher(None, a, b).ratio(), 3)

# ------------------ Memory Management ------------------ #
def clear_memory():
    """Aggressively clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# ------------------ Load Evaluation Data ------------------ #
print("📊 Loading evaluation data...")
eval_data = load_dataset("json", data_files=EVAL_FILE, split="train")

if len(eval_data) == 0:
    print("⚠️ No evaluation data loaded. Check your EVAL_FILE path and format.")
    exit(1)

# ------------------ Evaluate with Batch Processing ------------------ #
print("🚀 Running evaluation with batched processing...")
results = []

# Initialize batch containers
batch_prompts = []
batch_examples = []
batch_start_time = None

for idx in tqdm(range(len(eval_data)), desc="Evaluating", ncols=100):
    try:
        example = eval_data[idx]
        prompt = format_prompt(example)
        batch_prompts.append(prompt)
        batch_examples.append(example)
        
        # If batch is ready or this is the last example, run generation
        if len(batch_prompts) == SAFE_BATCH_SIZE or idx == len(eval_data) - 1:
            batch_start_time = time.time()
            
            # Tokenize the batch
            encoding = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            ).to(device)
            
            max_tokens = min(MAX_TOTAL_TOKENS - encoding["input_ids"].shape[1], MAX_NEW_TOKENS)
            
            # Generate outputs
            with torch.no_grad(), torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model.generate(
                    **encoding,
                    max_new_tokens=max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=False
                )
            
            batch_elapsed = time.time() - batch_start_time
            batch_time_per_item = batch_elapsed / len(batch_prompts)
            
            # Process results
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for prompt_text, decoded, example in zip(batch_prompts, decoded_outputs, batch_examples):
                response = decoded[len(prompt_text):].strip()
                score = fuzzy_match(response, example["output"])
                results.append({
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "expected_output": example["output"],
                    "generated_output": response,
                    "match_score": score,
                    "inference_time_sec": round(batch_time_per_item, 3)
                })
            
            # Save results incrementally in case of crashes
            if (idx + 1) % 10 == 0 or idx == len(eval_data) - 1:
                temp_file = RESULT_FILE + ".temp"
                with open(temp_file, "w", encoding="utf-8") as f:
                    for row in results:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
            # Clear after each batch
            clear_memory()
            batch_prompts.clear()
            batch_examples.clear()
            
    except torch.cuda.OutOfMemoryError:
        # Handle OOM by processing the current examples one by one
        clear_memory()
        print(f"\n⚠️ CUDA OOM at batch starting with example {idx-len(batch_prompts)+1}. Switching to single example mode.")
        
        # Process current batch items one by one
        for i, (single_prompt, single_example) in enumerate(zip(batch_prompts, batch_examples)):
            try:
                # Process individually
                single_encoding = tokenizer(
                    single_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_INPUT_LEN,
                ).to(device)
                
                start_time = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    single_output = model.generate(
                        **single_encoding,
                        max_new_tokens=max_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        do_sample=False
                    )
                elapsed = time.time() - start_time
                
                decoded_output = tokenizer.decode(single_output[0], skip_special_tokens=True)
                response = decoded_output[len(single_prompt):].strip()
                score = fuzzy_match(response, single_example["output"])
                
                results.append({
                    "instruction": single_example["instruction"],
                    "input": single_example["input"],
                    "expected_output": single_example["output"],
                    "generated_output": response,
                    "match_score": score,
                    "inference_time_sec": round(elapsed, 3)
                })
                
            except Exception as e:
                # Handle individual example errors
                print(f"\n⚠️ Error on fallback single example {idx-len(batch_prompts)+1+i}: {str(e)}")
                results.append({
                    "instruction": single_example["instruction"],
                    "input": single_example["input"],
                    "expected_output": single_example["output"],
                    "generated_output": f"ERROR: {str(e)}",
                    "match_score": 0.0,
                    "inference_time_sec": 0.0,
                    "error": str(e)
                })
            
            clear_memory()
        
        # After processing individually, reduce batch size for future batches
        SAFE_BATCH_SIZE = max(1, SAFE_BATCH_SIZE - 1)
        print(f"⚙️ Reduced batch size to {SAFE_BATCH_SIZE} for remaining examples")
        
        # Clear batch containers
        batch_prompts.clear()
        batch_examples.clear()
        
    except Exception as e:
        # Handle other errors
        print(f"\n⚠️ Error at batch starting with example {idx-len(batch_prompts)+1}: {str(e)}")
        for single_example in batch_examples:
            results.append({
                "instruction": single_example["instruction"],
                "input": single_example["input"],
                "expected_output": single_example["output"],
                "generated_output": f"ERROR: {str(e)}",
                "match_score": 0.0,
                "inference_time_sec": 0.0,
                "error": str(e)
            })
        
        clear_memory()
        batch_prompts.clear()
        batch_examples.clear()

# ------------------ Save Results ------------------ #
print(f"💾 Saving results to: {RESULT_FILE}")
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ------------------ Summary ------------------ #
successful_results = [r for r in results if "error" not in r]
if successful_results:
    avg_score = sum(r["match_score"] for r in successful_results) / len(successful_results)
    avg_time = sum(r["inference_time_sec"] for r in successful_results) / len(successful_results)
    print(f"\n🔍 Avg Fuzzy Match Score: {avg_score:.3f}")
    print(f"⏱️ Avg Inference Time: {avg_time:.3f} seconds per example")
print(f"✅ Evaluation complete. Processed {len(successful_results)}/{len(eval_data)} examples successfully.")