# 1. Import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import warnings
import textwrap
import os

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
print("-" * 30)

# 2. Define model, dataset, and new directory paths
model_id = "microsoft/phi-2"
dataset_id = "HuggingFaceH4/no_robots"
raw_model_dir = "./phi-2/raw"
finetuned_model_dir = "./phi-2/fine-tuned"
offload_dir = "./phi-2/offload"

os.makedirs(raw_model_dir, exist_ok=True)
os.makedirs(finetuned_model_dir, exist_ok=True)
os.makedirs(offload_dir, exist_ok=True)

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
print("-" * 30)

# --- PHASE 1: Interact with the Base Model (Before Fine-Tuning) ---
print("\n--- PHASE 1: TESTING BASE MODEL ---")
user_question_base = input("Enter your question for the BASE model: ")
prompt = f"Instruct: {user_question_base}\nOutput:"
# MODIFIED: Removed `return_attention_mask=False` to include the attention mask
inputs = tokenizer(prompt, return_tensors="pt").to(device) 
# MODIFIED: Added `pad_token_id` to the generate call
base_outputs = base_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
base_response = tokenizer.batch_decode(base_outputs)[0]
print("\n--- Base Model Response ---")
print(base_response.split("Output:")[1].strip())
print("-" * 30)

# --- Save the raw model for later comparison ---
print(f"--- Saving base model to {raw_model_dir} ---")
base_model.save_pretrained(raw_model_dir)
tokenizer.save_pretrained(raw_model_dir)
print("✅ Base model saved!")
print("-" * 30)


# 5. Configure LoRA and Fine-Tune
peft_model = prepare_model_for_kbit_training(base_model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(peft_model, lora_config)
dataset = load_dataset(dataset_id, split="train[:10%]")

def formatting_func(batch):
    output_texts = []
    for i in range(len(batch['messages'])):
        messages = batch['messages'][i]
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
        assistant_message = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), None)
        if user_message and assistant_message:
            text = f"Instruct: {user_message}\nOutput: {assistant_message}"
            output_texts.append(text)
    return output_texts

training_args = TrainingArguments(
    output_dir=finetuned_model_dir, # Save checkpoints and final adapter here
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    max_steps=100,
    optim="paged_adamw_8bit",
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

print("\n--- Starting Fine-Tuning Process for Phi-2 ---")
trainer.train()
print("✅ Fine-tuning completed!")
trainer.save_model() # Saves the adapter to the output_dir
print(f"✅ Fine-tuned adapter saved to {finetuned_model_dir}!")
print("-" * 30)

# --- Load the fully merged fine-tuned model ---
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    offload_folder=offload_dir
)
fine_tuned_model = PeftModel.from_pretrained(
    base_model_for_merge, 
    finetuned_model_dir,
    offload_folder=offload_dir
)
fine_tuned_model = fine_tuned_model.merge_and_unload()


# --- PHASE 2: Interact with the Fine-Tuned Model ---
print("\n--- PHASE 2: TESTING FINE-TUNED MODEL ---")
user_question_finetuned = input("Enter your question for the FINE-TUNED model: ")
prompt = f"Instruct: {user_question_finetuned}\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
finetuned_outputs = fine_tuned_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
finetuned_response = tokenizer.batch_decode(finetuned_outputs)[0]
print("\n--- Fine-Tuned Model Response ---")
print(finetuned_response.split("Output:")[1].strip())
print("-" * 30)

# --- Load the saved raw model for comparison ---
print(f"--- Loading saved raw model from {raw_model_dir} for comparison ---")
raw_model_for_comparison = AutoModelForCausalLM.from_pretrained(
    raw_model_dir,
    device_map="auto",
    trust_remote_code=True,
    offload_folder=offload_dir
)
print("✅ Models ready for comparison.")


# --- PHASE 3: Interactive Side-by-Side Comparison ---
while True:
    print("\n\n" + "="*85)
    print(f"{'--- PHASE 3: INTERACTIVE COMPARISON ---':^85}")
    print("="*85)
    user_question_comp = input("Enter your question (or type 'quit' to exit): ")
    
    if user_question_comp.lower() in ['quit', 'exit']:
        print("Exiting comparison.")
        break
        
    prompt = f"Instruct: {user_question_comp}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # --- Generate from RAW Model ---
    raw_outputs = raw_model_for_comparison.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    raw_response_text = tokenizer.batch_decode(raw_outputs)[0].split("Output:")[1].strip()

    # --- Generate from FINE-TUNED Model ---
    finetuned_outputs_comp = fine_tuned_model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    finetuned_response_text = tokenizer.batch_decode(finetuned_outputs_comp)[0].split("Output:")[1].strip()

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
