# qwen-fine-tuning-gpu
Fine-tuning Qwen2.5-0.5B (smallest Qwen) with Unsloth on NVIDIA RTX A500 Laptop GPU.


Note:


   3 nvcc --version
   5 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   6 pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl tran...
   7 python -c "import torch; print(torch.version.cuda)"
   9 pip uninstall triton triton-windows -y
  13 python.exe -m pip install --upgrade pip
  14 pip install triton-windows==3.1.0
  15 pip uninstall torchao -y
  16 python -c "import torch; print(torch.__version__)"
  17 pip install "triton-windows==3.2.0.post21"
  
python -c "import torch; print(torch.__version__)"

# PyTorch 2.5.x → triton 3.1.x
pip install "triton-windows==3.1.0.post17"

# PyTorch 2.6.x → triton 3.2.x
pip install "triton-windows==3.2.0.post21"

pip install "transformers>=4.46,<4.50" --upgrade
