# ICEdit Training Repository

This repository contains the training code for ICEdit, a model for image editing based on text instructions. It utilizes conditional generation to perform instructional image edits.

This codebase is based heavily on the [OminiControl](https://github.com/Yuanshi9815/OminiControl) repository. We thank the authors for their work and contributions to the field!

## Setup and Installation

```bash
# Create a new conda environment
conda create -n train python=3.10
conda activate train

# Install requirements
pip install -r requirements.txt
```

## Project Structure

- `src/`: Source code directory
  - `train/`: Training modules
    - `train.py`: Main training script
    - `data.py`: Dataset classes for handling different data formats
    - `model.py`: Model definition using Flux pipeline
    - `callbacks.py`: Training callbacks for logging and checkpointing
  - `flux/`: Flux model implementation
- `assets/`: Asset files
- `parquet/`: Parquet data files
- `requirements.txt`: Dependency list

## Datasets

Download training datasets (part of OmniEdit) to the `parquet/` directory. You can use the provided scripts `parquet/prepare.sh`.

```bash
cd parquet
bash prepare.sh
```

## Training

```bash
bash train/script/train.sh
```

You can modify the training configuration in `train/config/normal_lora.yaml`. 



## MoE-LoRA Training