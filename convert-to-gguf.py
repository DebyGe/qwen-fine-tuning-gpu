"""
Converte il modello fine-tunato (LoRA o merged) in formato GGUF
e crea un Modelfile pronto per Ollama.

Prerequisiti:
    pip install unsloth transformers
    # llama.cpp deve essere clonato e compilato (vedi sezione 0)

Workflow:
    1. Carica il modello fine-tunato (LoRA adapters o merged)
    2. Merge dei pesi LoRA nel modello base (se non già fatto)
    3. Salva come fp16 safetensors
    4. Converte in GGUF tramite llama.cpp convert_hf_to_gguf.py
    5. (Opzionale) Quantizza il GGUF in Q4_K_M
    6. Genera il Modelfile per Ollama
"""

# ---------------------------------------------------------------------------
# 0.  Configurazione
# ---------------------------------------------------------------------------

# Percorso agli adapter LoRA salvati dallo script di fine-tuning
LORA_ADAPTER_DIR  = "qwen-finetuned/lora_adapters"

# Dove salvare il modello merged fp16 (input per llama.cpp)
MERGED_DIR        = "qwen-finetuned/merged_fp16"

# Output GGUF
GGUF_OUTPUT_DIR   = "qwen-finetuned/gguf"
GGUF_FILENAME     = "qwen-finetuned.gguf"
GGUF_Q_FILENAME   = "qwen-finetuned-q4_k_m.gguf"   # quantizzato

# Percorso alla repo di llama.cpp (clonala con: git clone https://github.com/ggerganov/llama.cpp)
LLAMA_CPP_DIR     = "C:/llama.cpp"   # <-- modifica con il tuo percorso

# System prompt da inserire nel Modelfile
SYSTEM_PROMPT     = "You are a helpful assistant."

# Parametri Modelfile
TEMPERATURE       = 0.7
TOP_P             = 0.9
OLLAMA_MODEL_NAME = "qwen-finetuned"   # nome con cui apparirà in 'ollama list'

# Quantizzazione: True = esegue quantizzazione Q4_K_M dopo la conversione
DO_QUANTIZE       = True

# ---------------------------------------------------------------------------
# 1.  Imports
# ---------------------------------------------------------------------------

import os
import sys
import subprocess

# ---------------------------------------------------------------------------
# 2.  Merge LoRA → fp16
# ---------------------------------------------------------------------------

def merge_lora():
    print("[1/4] Merge LoRA adapters → fp16 safetensors …")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("ERROR: unsloth non trovato. Installa con: pip install unsloth")

    if not os.path.isdir(LORA_ADAPTER_DIR):
        sys.exit(f"ERROR: cartella adapter non trovata: {LORA_ADAPTER_DIR}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = LORA_ADAPTER_DIR,
        max_seq_length = 2048,
        dtype        = None,
        load_in_4bit = True,
    )

    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"   Modello merged salvato in: {MERGED_DIR}")


# ---------------------------------------------------------------------------
# 3.  Conversione in GGUF tramite llama.cpp
# ---------------------------------------------------------------------------

def convert_to_gguf():
    print("[2/4] Conversione in GGUF …")

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.isfile(convert_script):
        # prova il vecchio nome
        convert_script = os.path.join(LLAMA_CPP_DIR, "convert-hf-to-gguf.py")
    if not os.path.isfile(convert_script):
        sys.exit(
            f"ERROR: script di conversione non trovato in {LLAMA_CPP_DIR}\n"
            "Clona llama.cpp con: git clone https://github.com/ggerganov/llama.cpp\n"
            "e imposta LLAMA_CPP_DIR correttamente."
        )

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    gguf_out = os.path.join(GGUF_OUTPUT_DIR, GGUF_FILENAME)

    cmd = [
        sys.executable, convert_script,
        MERGED_DIR,
        "--outfile", gguf_out,
        "--outtype", "f16",
    ]
    print(f"   Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        sys.exit("ERROR: conversione GGUF fallita.")

    print(f"   GGUF salvato in: {gguf_out}")
    return gguf_out


# ---------------------------------------------------------------------------
# 4.  Quantizzazione Q4_K_M (opzionale)
# ---------------------------------------------------------------------------

def quantize_gguf(gguf_path: str) -> str:
    print("[3/4] Quantizzazione Q4_K_M …")

    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize.exe")
    if not os.path.isfile(quantize_bin):
        # fallback build senza sottocartella bin
        quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "llama-quantize.exe")
    if not os.path.isfile(quantize_bin):
        print("   WARN: llama-quantize.exe non trovato, salto la quantizzazione.")
        print("   Per compilarlo: cd llama.cpp && cmake -B build && cmake --build build --config Release")
        return gguf_path

    gguf_q_out = os.path.join(GGUF_OUTPUT_DIR, GGUF_Q_FILENAME)
    cmd = [quantize_bin, gguf_path, gguf_q_out, "Q4_K_M"]
    print(f"   Comando: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("   WARN: quantizzazione fallita, uso il GGUF f16.")
        return gguf_path

    print(f"   GGUF quantizzato salvato in: {gguf_q_out}")
    return gguf_q_out


# ---------------------------------------------------------------------------
# 5.  Genera Modelfile per Ollama
# ---------------------------------------------------------------------------

def create_modelfile(gguf_path: str):
    print("[4/4] Generazione Modelfile …")

    # Ollama su Windows vuole il path con backslash
    gguf_abs = os.path.abspath(gguf_path).replace("/", "\\")

    modelfile_content = f"""FROM {gguf_abs}

PARAMETER temperature {TEMPERATURE}
PARAMETER top_p {TOP_P}
PARAMETER stop "<|im_end|>"

SYSTEM \"{SYSTEM_PROMPT}\"
"""

    modelfile_path = os.path.join(GGUF_OUTPUT_DIR, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"   Modelfile salvato in: {modelfile_path}")
    print()
    print("=" * 60)
    print("  Carica il modello in Ollama con questi comandi:")
    print("=" * 60)
    print(f"  ollama create {OLLAMA_MODEL_NAME} -f {modelfile_path}")
    print(f"  ollama run {OLLAMA_MODEL_NAME}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Conversione GGUF per Ollama ===\n")

    merge_lora()
    gguf_path = convert_to_gguf()

    if DO_QUANTIZE:
        gguf_path = quantize_gguf(gguf_path)

    create_modelfile(gguf_path)

    print("\nConversione completata.")
