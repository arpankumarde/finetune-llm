# 1. Import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import warnings
import textwrap
warnings.filterwarnings("ignore")

# --- Check for GPU availability ---
if torch.cuda.is_available():
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"✅ {gpu_count} GPU(s) detected.")
    # Get the name of the primary GPU
    print(f"   - Primary GPU: {torch.cuda.get_device_name(0)}")
    # Set the device to the first available GPU
    device = "cuda:0"
else:
    print("⚠️ No GPU detected, falling back to CPU. This will be very slow.")
    device = "cpu"
print("-" * 30)


# 2. Define the model and dataset names
model_id = "microsoft/phi-2" # MODIFIED: Switched to Phi-2
dataset_id = "HuggingFaceH4/no_robots"

# 3. Configure Quantization (for loading model in 4-bit)
# This reduces VRAM requirements significantly and is loaded onto the GPU.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 4. Load the tokenizer and the base model for training
# Phi-2 requires `trust_remote_code=True`
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Load the model with the 4-bit quantization config.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatically maps to detected device
    trust_remote_code=True
)

# --- Confirm model device placement ---
print(f"Model successfully loaded on device: {model.device}")
print("-" * 30)

# 5. Configure LoRA (Parameter-Efficient Fine-Tuning)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    # MODIFIED: Target modules updated for Phi-2
    target_modules=["q_proj", "k_proj", "v_proj", "dense"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, lora_config)

# 6. Load the dataset
dataset = load_dataset(dataset_id, split="train[:10%]")

# --- MODIFIED: Create a formatting function for the Phi-2 instruction format ---
def formatting_func(batch):
    output_texts = []
    for i in range(len(batch['messages'])):
        messages = batch['messages'][i]
        # We're assuming a simple user-assistant turn for the instruction format
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), None)
        assistant_message = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), None)
        
        if user_message and assistant_message:
            text = f"Instruct: {user_message}\nOutput: {assistant_message}"
            output_texts.append(text)
    return output_texts

# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./phi2-no_robots-finetuned", # MODIFIED: Updated output directory
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    max_steps=100,
    optim="paged_adamw_8bit",
)

# 8. Initialize the SFTTrainer
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func, # Using the new formatting function
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# 9. Start the fine-tuning process
print("\n--- Starting Fine-Tuning Process for Phi-2 ---")
trainer.train()
print("✅ Fine-tuning completed!")

# Explicitly save the final adapter to the output directory
print("--- Saving final adapter ---")
trainer.save_model()
print("✅ Adapter saved!")


# 10. Load models for comparison
print("\n--- Loading models for interactive comparison ---")

# Load the original base model (quantized)
base_model_for_comparison = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned model
base_model_for_merge = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
final_adapter_path = "./phi2-no_robots-finetuned"
fine_tuned_model = PeftModel.from_pretrained(base_model_for_merge, final_adapter_path)
fine_tuned_model = fine_tuned_model.merge_and_unload()

print("✅ Models ready for comparison.")


# --- Step 11 - Interactive Side-by-Side Comparison ---
while True:
    print("\n\n" + "="*85)
    print(f"{'--- INTERACTIVE COMPARISON (PHI-2) ---':^85}")
    print("="*85)
    user_question = input("Enter your question (or type 'quit' to exit): ")
    
    if user_question.lower() in ['quit', 'exit']:
        print("Exiting comparison.")
        break
        
    # Create the prompt in the correct format for Phi-2
    prompt = f"Instruct: {user_question}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    
    # --- Generate from Base Model ---
    base_outputs = base_model_for_comparison.generate(**inputs, max_new_tokens=100)
    base_response = tokenizer.batch_decode(base_outputs)[0]
    # Parse the response for Phi-2 format
    base_model_response_text = base_response.split("Output:")[1].strip()

    # --- Generate from Fine-Tuned Model ---
    finetuned_outputs = fine_tuned_model.generate(**inputs, max_new_tokens=100)
    finetuned_response = tokenizer.batch_decode(finetuned_outputs)[0]
    # Parse the response for Phi-2 format
    fine_tuned_model_response_text = finetuned_response.split("Output:")[1].strip()

    # --- Print Comparison Table ---
    print("\n" + "-" * 85)
    print(f"| {'BASE MODEL':^38} | {'FINE-TUNED MODEL':^40} |")
    print("-" * 85)

    wrapper = textwrap.TextWrapper(width=38)
    base_lines = wrapper.wrap(base_model_response_text)
    finetuned_lines = wrapper.wrap(fine_tuned_model_response_text)

    max_lines = max(len(base_lines), len(finetuned_lines))
    base_lines += [''] * (max_lines - len(base_lines))
    finetuned_lines += [''] * (max_lines - len(finetuned_lines))

    for i in range(max_lines):
        print(f"| {base_lines[i]:<38} | {finetuned_lines[i]:<40} |")

    print("-" * 85)
