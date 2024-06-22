from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, wandb, platform, gradio, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

def print_system_specs():
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print("CUDA Available:", is_cuda_available)
# Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_cuda_devices)
    if is_cuda_available:
        for i in range(num_cuda_devices):
            # Get CUDA device properties
            device = torch.device('cuda', i)
            print(f"--- CUDA Device {i} ---")
            print("Name:", torch.cuda.get_device_name(i))
            print("Compute Capability:", torch.cuda.get_device_capability(i))
            print("Total Memory:", torch.cuda.get_device_properties(i).total_memory, "bytes")
    # Get CPU information
    print("--- CPU Information ---")
    print("Processor:", platform.processor())
    print("System:", platform.system(), platform.release())
    print("Python Version:", platform.python_version())
def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n'
    B_INST, E_INST = "### Instruction:\n", "### Response:\n"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Despite returning the usual output, the streamer will also print the generated text to stdout.
    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=500)
print_system_specs()
# Pre trained model
model_name = "meta-llama/Llama-2-7b-hf"

# Dataset name
dataset_name = "vicgalle/alpaca-gpt4"

# Hugging face repository link to save fine-tuned model(Create new repository in huggingface,copy and paste here)
new_model = "LLama3_fine_tuned"
# Create an IPython shell instance
login(token='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')

# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train[0:10000]")
print(dataset["text"][0])

print(dataset['instruction'][0])
print('\n\n' + '*' * 80)
print(dataset['input'][0])
print('\n\n' + '*' * 80)
print(dataset['output'][0])
##
# EMPTY CACHE
torch.cuda.empty_cache()
##
# Specify the directory where you want to save the model
save_directory = "./local_model_directory"
if os.path.exists(save_directory):
        # Load the model from the local directory
    model = AutoModelForCausalLM.from_pretrained(save_directory)



if not os.path.exists(save_directory) or (os.path.exists(save_directory) and not os.listdir(save_directory)):
    # Load base model(llama-2-7b-hf) and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.float16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model)
    # Save the model locally
    model.save_pretrained(save_directory)
else:
    # Load the model from the local directory
    model = AutoModelForCausalLM.from_pretrained(save_directory)

model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
print(tokenizer.add_bos_token, tokenizer.add_eos_token)

# monitering login
wandb.login(key="49875b16a455d44adac0b4f13991e39580b6ae07")
run = wandb.init(project='Fine tuning llama-2-7B', job_type="training", anonymous="allow")

peft_config = LoraConfig(
    lora_alpha= 8,
    lora_dropout= 0.1,
    r= 16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 30,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "linear",
    report_to="wandb",
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

# Train model
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()

stream("what is newtons 2rd law and its formula")


