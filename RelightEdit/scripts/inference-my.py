# Use the modified diffusers & peft library
import sys
import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline

# Below is the original library
import torch
from PIL import Image
import numpy as np
import argparse
import random
    
parser = argparse.ArgumentParser() 
parser.add_argument("--image", type=str, help="Name of the image to be edited", required=True)
parser.add_argument("--instruction", type=str, help="Instruction for editing the image", required=True)
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='sanaka87/ICEdit-MoE-LoRA', help="Path to the LoRA weights")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
parser.add_argument("--tasks", type=str,required=False, help="labels for MoE, e.g., 'color,intensity'")

args = parser.parse_args()
# pipe = FluxFillPipeline.from_pretrained(args.flux_path).to(dtype=torch.bfloat16)
# pipe.load_lora_weights(args.lora_path)

pipe = FluxFillPipeline.from_pretrained(
    args.flux_path,
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(args.lora_path)

# if args.enable_model_cpu_offload:
#     pipe.enable_model_cpu_offload() 
# else:
#     pipe = pipe.to("cuda")

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload() 
else:
    try:
        pipe = pipe.to("cuda")
    except NotImplementedError:
        print("\033[93m[WARNING] Detected meta tensors, using to_empty() instead of to().\033[0m")
        pipe.to_empty(device="cuda")

image = Image.open(args.image)
image = image.convert("RGB")

if image.size[0] != 512:
    print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
    new_width = 512
    scale = new_width / image.size[0]
    new_height = int(image.size[1] * scale)
    new_height = (new_height // 8) * 8  
    image = image.resize((new_width, new_height))
    print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")


# instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}'
instruction = "changing the color of the left door light to the green, intensity to the strong" #"changing the color of the car light to the green"
instruction = args.instruction
print(f"Instruction: {instruction}")

width, height = image.size
combined_image = Image.new("RGB", (width * 2, height))
combined_image.paste(image, (0, 0))
combined_image.paste(image, (width, 0))
mask_array = np.zeros((height, width * 2), dtype=np.uint8)
mask_array[:, width:] = 255 
mask = Image.fromarray(mask_array)

mp = {
    'intensity': 0,
    'color': 1,
    'direction': 2,
    'transfer': 3,
}
if args.tasks is not None:
    task = args.tasks
else:
    task = "color,intensity"
labels = [mp[t.strip()] for t in task.split(",") if t.strip() in mp]
labels = torch.tensor(labels, dtype=torch.int64)  # (1, num_labels)

result_image = pipe(
    prompt=instruction,
    image=combined_image,
    mask_image=mask,
    height=height,
    width=width * 2,
    guidance_scale=50,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(args.seed) if args.seed is not None else None,
    labels=labels, # Edited by wangzh
).images[0]

result_image = result_image.crop((width,0,width*2,height))

os.makedirs(args.output_dir, exist_ok=True)

image_name = args.image.split("/")[-1]
result_image.save(os.path.join(args.output_dir, f"{image_name}"))
print(f"\033[92mResult saved as {os.path.abspath(os.path.join(args.output_dir, image_name))}\033[0m")

'''
python scripts/inference-my.py \
    --image "/mnt/data/wangzh/kontext/data/multiple/17_3.png" \
    --instruction "changing the color of the car light to green, intensity to the strong." \
    --flux-path "/mnt/data/wangzh/models/FLUX.1-Fill-dev/FLUX.1-Fill-dev" \
    --lora-path "/mnt/data/wangzh/kontext/checkpoints/multiple/runs/20251107-091236/ckpt/50000/pytorch_lora_weights.safetensors" \
    --tasks "color,intensity"
'''
