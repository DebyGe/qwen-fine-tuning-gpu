"""
Fine-tuning Qwen2.5-0.5B (smallest Qwen) with Unsloth on NVIDIA RTX A500 Laptop GPU.
Training data is loaded from a local JSON file.

JSON format expected (ShareGPT / chat style):
[
  {
    "conversations": [
      {"from": "human", "value": "What is 2+2?"},
      {"from": "gpt",   "value": "4"}
    ]
  },
  ...
]

Alternatively, instruction-style (also supported via DATA_FORMAT below):
[
  {"instruction": "...", "input": "...", "output": "..."},
  ...
]
"""

# ---------------------------------------------------------------------------
# 0.  Imports
# ---------------------------------------------------------------------------
import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# 1.  Configuration – edit these to match your setup
# ---------------------------------------------------------------------------

MODEL_NAME   = "unsloth/Qwen3.5-0.8B" # "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"  # 4-bit quantised, fits easily in 4 GB VRAM
MAX_SEQ_LEN  = 2048          # max token length per sample
LORA_RANK    = 16            # LoRA rank (8–32 is a good range for small models)
LORA_ALPHA   = 16
LORA_DROPOUT = 0.0           # 0 is recommended by Unsloth

# Training hyper-parameters
BATCH_SIZE          = 2      # per-device batch size (low for 4 GB VRAM)
GRAD_ACCUM_STEPS    = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
WARMUP_STEPS        = 10
MAX_STEPS           = 100    # set to -1 to use NUM_EPOCHS instead
NUM_EPOCHS          = 3
LEARNING_RATE       = 2e-4
WEIGHT_DECAY        = 0.01
LR_SCHEDULER        = "cosine"
OPTIMIZER           = "adamw_8bit"   # 8-bit Adam from bitsandbytes
FP16                = not torch.cuda.is_bf16_supported()
BF16                = torch.cuda.is_bf16_supported()
SEED                = 42

# Paths
JSON_DATA_PATH  = "data.json"        # your training data file
OUTPUT_DIR      = "outputs"          # checkpoints are saved here
FINAL_MODEL_DIR = "qwen-finetuned"   # merged / final model saved here

# "sharegpt" | "alpaca"
DATA_FORMAT = "sharegpt"

# ---------------------------------------------------------------------------
# 2.  GPU sanity check
# ---------------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA not found – make sure the NVIDIA RTX A500 drivers and CUDA toolkit are installed."
gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[GPU] {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

# ---------------------------------------------------------------------------
# 3.  Load model + tokeniser via Unsloth (4-bit quantised)
# ---------------------------------------------------------------------------

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,   # auto-detect (bf16 on Ampere+)
    load_in_4bit    = True,
)

# Apply chat template so the tokeniser knows how to format conversations
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ---------------------------------------------------------------------------
# 4.  Attach LoRA adapters
# ---------------------------------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_RANK,
    target_modules      = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # saves extra VRAM
    random_state        = SEED,
    use_rslora          = False,
    loftq_config        = None,
)

print(model.print_trainable_parameters())

# ---------------------------------------------------------------------------
# 5.  Load & pre-process the JSON dataset
# ---------------------------------------------------------------------------

assert os.path.isfile(JSON_DATA_PATH), f"Training file not found: {JSON_DATA_PATH}"

with open(JSON_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

assert isinstance(raw_data, list) and len(raw_data) > 0, "JSON file must contain a non-empty list."


def format_sharegpt(sample: dict) -> dict:
    """Convert a ShareGPT-style conversation to a single tokenised string."""
    conversations = sample.get("conversations", [])
    messages = []
    for turn in conversations:
        role  = "user" if turn["from"] == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_alpaca(sample: dict) -> dict:
    """Convert an Alpaca-style instruction sample to a single tokenised string."""
    instruction = sample.get("instruction", "")
    inp         = sample.get("input", "")
    output      = sample.get("output", "")
    user_msg    = f"{instruction}\n{inp}".strip() if inp else instruction
    messages = [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


format_fn = format_sharegpt if DATA_FORMAT == "sharegpt" else format_alpaca
formatted  = [format_fn(s) for s in raw_data]
dataset    = Dataset.from_list(formatted)

print(f"[Data] {len(dataset)} training samples loaded from '{JSON_DATA_PATH}'.")

# ---------------------------------------------------------------------------
# 6.  Trainer
# ---------------------------------------------------------------------------

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS,
    warmup_steps                = WARMUP_STEPS,
    max_steps                   = MAX_STEPS,    # set to -1 to rely on num_train_epochs
    num_train_epochs            = NUM_EPOCHS,
    learning_rate               = LEARNING_RATE,
    weight_decay                = WEIGHT_DECAY,
    lr_scheduler_type           = LR_SCHEDULER,
    optim                       = OPTIMIZER,
    fp16                        = FP16,
    bf16                        = BF16,
    logging_steps               = 10,
    save_steps                  = 50,
    save_total_limit            = 2,
    seed                        = SEED,
    report_to                   = "none",       # change to "wandb" if you use W&B
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LEN,
    args            = training_args,
)

# ---------------------------------------------------------------------------
# 7.  Train
# ---------------------------------------------------------------------------

print("[Train] Starting fine-tuning …")
trainer_stats = trainer.train()
print("[Train] Done.")
print(f"  Peak VRAM used: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
print(f"  Training loss:  {trainer_stats.training_loss:.4f}")

# ---------------------------------------------------------------------------
# 8.  Save – LoRA adapters only (fast, small)
# ---------------------------------------------------------------------------

adapter_dir = os.path.join(FINAL_MODEL_DIR, "lora_adapters")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"[Save] LoRA adapters saved to '{adapter_dir}'")

# ---------------------------------------------------------------------------
# 9.  Optional: merge LoRA weights into full model and save as fp16
#     (uncomment if you need a standalone model without Unsloth at inference)
# ---------------------------------------------------------------------------

# merged_dir = os.path.join(FINAL_MODEL_DIR, "merged_fp16")
# model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
# print(f"[Save] Merged fp16 model saved to '{merged_dir}'")

# ---------------------------------------------------------------------------
# 10. Quick inference test (optional)
# ---------------------------------------------------------------------------

FastLanguageModel.for_inference(model)   # enable native 2x faster inference

# Build prompt manually – Qwen chat format: <|im_start|>role\ncontent<|im_end|>
# Pass text= explicitly so the multimodal processor skips its image pipeline
prompt_text = "<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
inputs = tok(prompt_text, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    outputs = model.generate(
        input_ids  = inputs,
        max_new_tokens = 128,
        temperature    = 0.7,
        top_p          = 0.9,
        do_sample      = True,
    )

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"\n[Inference test]\nPrompt : Hello, who are you?\nResponse: {response}\n")
