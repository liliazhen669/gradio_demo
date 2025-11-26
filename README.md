# 💼 Installation

## Conda environment setup

```bash
conda create -n relightmoe python=3.10
conda activate relightmoe
pip install -r requirements.txt
pip install -U huggingface_hub
```

## Inference in bash with gradio

Now you can have a try!

```bash
python app.py --lora_dir "your lora dir" 
```