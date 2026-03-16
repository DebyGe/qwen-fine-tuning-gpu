Ecco come convertire un modello fine-tuned (LoRA o merged) in formato GGUF:

1. Prerequisiti
Clona llama.cpp e installa le dipendenze:
bashgit clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt

2. Caso A — Modello già merged (LoRA applicata)
Se hai già un modello HuggingFace completo (LoRA già fusa nei pesi), vai direttamente alla conversione:
bashpython convert_hf_to_gguf.py /percorso/modello \
  --outfile modello.gguf \
  --outtype f16
--outtype può essere: f32, f16, bf16, q8_0, o auto.

3. Caso B — LoRA separata (adapter non merged)
Prima devi fondere la LoRA nel modello base con peft:
pythonfrom peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("modello-base")
model = PeftModel.from_pretrained(base_model, "/percorso/lora-adapter")

# Merge e salva
merged = model.merge_and_unload()
merged.save_pretrained("/percorso/modello-merged")

tokenizer = AutoTokenizer.from_pretrained("modello-base")
tokenizer.save_pretrained("/percorso/modello-merged")
Poi procedi come nel Caso A.

4. Quantizzazione (opzionale ma consigliata)
Dopo la conversione in GGUF, puoi quantizzare per ridurre dimensioni e memoria:
bash# Prima converti in f16
python convert_hf_to_gguf.py /percorso/modello --outfile modello-f16.gguf --outtype f16

# Poi quantizza con llama-quantize
./llama-quantize modello-f16.gguf modello-q4_k_m.gguf Q4_K_M
I formati di quantizzazione più comuni:
FormatoQualitàDimensioneQ8_0Ottima~1xQ6_KMolto buona~0.75xQ4_K_MBuona (bilanciata)~0.5xQ4_K_SDiscreta~0.45xQ3_K_MAccettabile~0.37x

Note importanti

convert_hf_to_gguf.py sostituisce il vecchio convert.py nelle versioni recenti di llama.cpp — verifica quale script è presente nel tuo clone.
Il modello deve essere in formato HuggingFace (cartella con config.json, tokenizer.json, ecc.).
Architetture supportate: LLaMA, Mistral, Phi, Gemma, Qwen, Falcon e molte altre — controlla la lista aggiornata nel repo di llama.cpp.
