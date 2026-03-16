# qwen-fine-tuning-gpu
Fine-tuning Qwen2.5-0.5B (smallest Qwen) with Unsloth on NVIDIA RTX A500 Laptop GPU.


Note:

python -c "import torch; print(torch.__version__)"

# PyTorch 2.5.x → triton 3.1.x
pip install "triton-windows==3.1.0.post17"

# PyTorch 2.6.x → triton 3.2.x
pip install "triton-windows==3.2.0.post21"

pip install "transformers>=4.46,<4.50" --upgrade
