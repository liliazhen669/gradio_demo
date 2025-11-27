import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Use the modified diffusers & peft library
import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../icedit"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import random
import time
import numpy as np
from datasets import load_dataset

from .data import (
    EditDataset,
    OminiDataset,
    EditDataset_with_Omini,
    IDLightDataset,
)
from .model import OminiModel
from .callbacks import TrainingCallback


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

def display_batch(dataset, batch_size=4, num_show=4, save_path="batch_vis.png"):
    """
    在 headless 环境下可用：
    从 EditDataset_with_Omini 数据集中可视化一个 batch，
    显示拼接图与对应的 condition(mask)，保存为单张图片。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch = next(iter(loader))

    # 打印 shape 信息
    print("=== Batch Shape Info ===")
    shape_info = {}
    for k, v in batch.items():
        if hasattr(v, 'shape'):
            shape_info[k] = tuple(v.shape)
            print(f"{k}: {tuple(v.shape)}")
        else:
            shape_info[k] = type(v)
            print(f"{k}: {type(v)}, {v}")

    # Tensor 转 PIL
    def tensor_to_pil(img_tensor):
        img = (img_tensor * 255).clamp(0, 255).byte()
        img = img.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(img)

    def mask_to_pil(mask_tensor):
        mask = (mask_tensor.squeeze(0) * 255).clamp(0, 255).byte().cpu().numpy()
        return Image.fromarray(mask, mode="L")

    num_show = min(num_show, len(batch["image"]))
    images = [tensor_to_pil(batch["image"][i]) for i in range(num_show)]
    masks = [mask_to_pil(batch["condition"][i]) for i in range(num_show)]
    instructions = batch["description"]

    # 拼接图像：每个样本两行（图 + mask）
    w, h = images[0].size
    padding = 20
    font_size = 20
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 每个样本占 2h + 文本高度 + padding
    sample_height = h * 2 + font_size + padding
    total_height = num_show * sample_height
    combined = Image.new("RGB", (w, total_height), (255, 255, 255))

    draw = ImageDraw.Draw(combined)
    y = 0
    for i in range(num_show):
        combined.paste(images[i], (0, y))
        y += h
        # mask 可视化为三通道灰度
        mask_rgb = Image.merge("RGB", (masks[i], masks[i], masks[i]))
        combined.paste(mask_rgb, (0, y))
        y += h
        text = instructions[i] if isinstance(instructions[i], str) else str(instructions[i])
        draw.text((10, y), text[:100] + ("..." if len(text) > 100 else ""), fill=(0, 0, 0), font=font)
        y += font_size + padding

    combined.save(save_path)
    print(f"Batch visualization saved to: {save_path}")
    return shape_info

def main():
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)
        
    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    
    if training_config["dataset"]["type"] == "edit":
        dataset = load_dataset('osunlp/MagicBrush')
        dataset = EditDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "omini":
        dataset = load_dataset(training_config["dataset"]["path"])
        dataset = OminiDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    # Edited by wangzh
    elif training_config["dataset"]["type"] == "edit_with_omini":
        omni = load_dataset("parquet", data_files=os.path.abspath(training_config["dataset"]["path"]), split="train")
        # magic = load_dataset('osunlp/MagicBrush')
        magic = load_dataset(
            "parquet",
            data_files={'train':"/home/wangzhen/Techniques/ICEdit/train/parquet/MagicBrush/train-00000-of-00051-9fd9f23e2b1cb397.parquet",
                        'dev': "/home/wangzhen/Techniques/ICEdit/train/parquet/MagicBrush/dev-00000-of-00004-f147d414270a90e1.parquet"}
        )
        dataset = EditDataset_with_Omini(
            magic,
            omni,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    elif training_config["dataset"]["type"] == "edit_with_idlight":
        dataset = IDLightDataset(
            csv_path=training_config["dataset"]["csv_path"],
            image_dir=training_config["dataset"]["image_dir"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    info = display_batch(dataset, batch_size=1, num_show=1, save_path="batch_vis.png")
    print("info:", info)

    print("Dataset length:", len(dataset))

    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_fill_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )
    
    from lightning.pytorch.strategies import DDPStrategy
    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        strategy=DDPStrategy(find_unused_parameters=True), # Modified by wangzh
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
