# 1. Import necessary libraries
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    TrainingArguments, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import warnings
import textwrap
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers.trainer_utils import EvalPrediction
from torch.nn import CrossEntropyLoss

warnings.filterwarnings("ignore")

# --- Check for GPU availability ---
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✅ {gpu_count} GPU(s) detected.")
    print(f"   - Primary GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda:0"
else:
    print("⚠️ No GPU detected, falling back to CPU. This will be very slow.")
    device = "cpu"
print("-" * 50)

# 2. Define model, dataset, and new directory paths
model_id = "microsoft/phi-2"
dataset_id = "HuggingFaceH4/no_robots"
finetuned_adapter_dir = "./phi-2/fine-tuned-adapter-enhanced"
finetuned_merged_dir = "./phi-2/fine-tuned-merged-enhanced"
offload_dir = "./phi-2/offload"
logs_dir = "./phi-2/logs"
results_dir = "./phi-2/results"

os.makedirs(finetuned_adapter_dir, exist_ok=True)
os.makedirs(finetuned_merged_dir, exist_ok=True)
os.makedirs(offload_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# 3. Configure Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 4. Load Tokenizer and Base Model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print(f"Base model loaded on device: {base_model.device}")
print("-" * 50)

# --- PHASE 1: Interact with the Base Model (Before Fine-Tuning) ---
print("\n--- PHASE 1: TESTING BASE MODEL ---")
user_question_base = input("Enter your question for the BASE model: ")
prompt = f"### Instruction:\n{user_question_base}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device) 
base_outputs = base_model.generate(
    **inputs, 
    max_new_tokens=150, 
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
base_response = tokenizer.batch_decode(base_outputs)[0]
print("\n--- Base Model Response ---")
print(base_response.split("### Response:\n")[1].strip().replace("<|endoftext|>", ""))
print("-" * 50)


# 5. Configure LoRA and Fine-Tune
peft_model = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(peft_model, lora_config)
peft_model.print_trainable_parameters()

train_dataset = load_dataset(dataset_id, split="train[:20%]")
eval_dataset = load_dataset(dataset_id, split="train[20%:25%]")

def formatting_func(batch):
    output_texts = []
    for i in range(len(batch['messages'])):
        messages = batch['messages'][i]
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
        assistant_message = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), None)
        if user_message and assistant_message:
            text = f"### Instruction:\n{user_message}\n\n### Response:\n{assistant_message}"
            output_texts.append(text)
    return output_texts

# MODIFIED: Added compute_metrics function for evaluation
def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    labels = labels[..., 1:].contiguous()
    predictions = predictions[..., :-1, :].contiguous()
    
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(predictions.view(-1, predictions.size(-1)), labels.view(-1))
    perplexity = torch.exp(loss).item()
    
    return {"perplexity": perplexity, "eval_loss": loss.item()}

training_args = TrainingArguments(
    output_dir=finetuned_adapter_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_dir=logs_dir,
    logging_steps=5,
    optim="paged_adamw_8bit",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics, # Added metrics computation
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\n--- Starting Fine-Tuning Process for Phi-2 ---")
train_result = trainer.train()
print("✅ Fine-tuning completed!")
trainer.save_model() 
print(f"✅ Best fine-tuned adapter saved to {finetuned_adapter_dir}!")
print("-" * 50)

# MODIFIED: Function to plot training progress
def plot_training_progress(log_history, results_dir):
    train_loss = [log['loss'] for log in log_history if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    train_steps = [log['step'] for log in log_history if 'loss' in log]
    eval_steps = [log['step'] for log in log_history if 'eval_loss' in log]

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label='Training Loss')
    plt.plot(eval_steps, eval_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(results_dir, "training_progress.png")
    plt.savefig(save_path)
    print(f"📈 Training progress chart saved to {save_path}")
    plt.show()

# MODIFIED: Plot and save the training progress
plot_training_progress(trainer.state.log_history, results_dir)

# --- Merge adapter and save the full fine-tuned model to disk ---
print("--- Merging adapter and saving full fine-tuned model ---")
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
fine_tuned_model = PeftModel.from_pretrained(
    base_model_for_merge, 
    finetuned_adapter_dir,
)
fine_tuned_model = fine_tuned_model.merge_and_unload()
fine_tuned_model.save_pretrained(finetuned_merged_dir)
tokenizer.save_pretrained(finetuned_merged_dir)
print(f"✅ Fully merged fine-tuned model saved to {finetuned_merged_dir}!")
print("-" * 50)

# --- Clear all models from memory before comparison phases ---
print("--- Clearing all models from memory to free up VRAM ---")
del base_model
del peft_model
del base_model_for_merge
del fine_tuned_model
torch.cuda.empty_cache()
print("✅ GPU Memory Cleared.")
print("-" * 50)


# --- PHASE 2: Interact with the Fine-Tuned Model ---
print("\n--- PHASE 2: TESTING FINE-TUNED MODEL ---")
fine_tuned_model_for_test = AutoModelForCausalLM.from_pretrained(
    finetuned_merged_dir,
    device_map="auto",
    trust_remote_code=True,
    offload_folder=offload_dir
)
user_question_finetuned = input("Enter your question for the FINE-TUNED model: ")
prompt = f"### Instruction:\n{user_question_finetuned}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
finetuned_outputs = fine_tuned_model_for_test.generate(
    **inputs, 
    max_new_tokens=150, 
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
finetuned_response = tokenizer.batch_decode(finetuned_outputs)[0]
print("\n--- Fine-Tuned Model Response ---")
print(finetuned_response.split("### Response:\n")[1].strip().replace("<|endoftext|>", ""))
del fine_tuned_model_for_test
torch.cuda.empty_cache()
print("-" * 50)


# --- PHASE 3: Interactive Side-by-Side Comparison ---
while True:
    print("\n\n" + "="*85)
    print(f"{'--- PHASE 3: INTERACTIVE COMPARISON ---':^85}")
    print("="*85)
    user_question_comp = input("Enter your question (or type 'quit' to exit): ")
    
    if user_question_comp.lower() in ['quit', 'exit']:
        print("Exiting comparison.")
        break
        
    prompt = f"### Instruction:\n{user_question_comp}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # --- Generate from RAW Model ---
    print("Loading RAW model for generation...")
    raw_model_for_comparison = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map=device, 
        trust_remote_code=True
    )
    raw_outputs = raw_model_for_comparison.generate(
        **inputs, 
        max_new_tokens=150, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    raw_response_text = tokenizer.batch_decode(raw_outputs)[0].split("### Response:\n")[1].strip().replace("<|endoftext|>", "")
    del raw_model_for_comparison
    torch.cuda.empty_cache()
    print("RAW model unloaded.")

    # --- Generate from FINE-TUNED Model ---
    print("Loading FINE-TUNED model for generation...")
    fine_tuned_model_for_comp = AutoModelForCausalLM.from_pretrained(
        finetuned_merged_dir, 
        device_map="auto", 
        trust_remote_code=True, 
        offload_folder=offload_dir
    )
    finetuned_outputs_comp = fine_tuned_model_for_comp.generate(
        **inputs, 
        max_new_tokens=150, 
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    finetuned_response_text = tokenizer.batch_decode(finetuned_outputs_comp)[0].split("### Response:\n")[1].strip().replace("<|endoftext|>", "")
    del fine_tuned_model_for_comp
    torch.cuda.empty_cache()
    print("FINE-TUNED model unloaded.")

    # --- Print Comparison Table ---
    print("\n" + "-" * 85)
    print(f"| {'RAW MODEL':^38} | {'FINE-TUNED MODEL':^40} |")
    print("-" * 85)

    wrapper = textwrap.TextWrapper(width=38)
    raw_lines = wrapper.wrap(raw_response_text)
    finetuned_lines = wrapper.wrap(finetuned_response_text)

    max_lines = max(len(raw_lines), len(finetuned_lines))
    raw_lines += [''] * (max_lines - len(raw_lines))
    finetuned_lines += [''] * (max_lines - len(finetuned_lines))

    for i in range(max_lines):
        print(f"| {raw_lines[i]:<38} | {finetuned_lines[i]:<40} |")

    print("-" * 85)

# --- Final Evaluation Results ---
print("\n\n" + "="*50)
print(f"{'--- FINAL EVALUATION METRICS ---':^50}")
print("="*50)
final_metrics = trainer.state.log_history[-1]
print(f"Final Training Loss: {final_metrics.get('train_loss', 'N/A'):.4f}")
print(f"Final Validation Loss: {final_metrics.get('eval_loss', 'N/A'):.4f}")
print(f"Final Perplexity: {final_metrics.get('eval_perplexity', 'N/A'):.4f}")
print(f"Total Training Steps: {trainer.state.global_step}")
print(f"Epoch at which training stopped: {trainer.state.epoch:.2f}")
print("="*50)
