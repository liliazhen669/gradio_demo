import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "./cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./cache"
os.environ["TRANSFORMERS_CACHE"] = "./cache"
os.environ["TORCH_HOME"] = "./cache"

import sys
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'qwenimage'))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

import gradio as gr
import numpy as np
import spaces
import torch
import random
from PIL import Image
from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

from huggingface_hub import hf_hub_download

from diffusers import FluxKontextPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    # Ensure dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Relighting Demo for RelightMoE")
    parser.add_argument("--lora_dir", type=str, default="./cache")

    return parser.parse_args()

MAX_SEED = np.iinfo(np.int32).max
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

args = parse_args()
lora_dir = args.loar_dir

# kontextpipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
# kontextpipe.load_lora_weights(os.path.join(lora_dir, "kontext-relight.safetensors"), adapter_name="kontext-relight")
# kontextpipe.load_lora_weights(os.path.join(lora_dir, "kontext-combine.safetensors"), adapter_name="combine")

# qwenpipe = QwenImageEditPlusPipeline.from_pretrained(
#     "Qwen/Qwen-Image-Edit-2509",
#     transformer=QwenImageTransformer2DModel.from_pretrained(
#         "linoyts/Qwen-Image-Edit-Rapid-AIO", # [transformer weights extracted from: Phr00t/Qwen-Image-Edit-Rapid-AIO]
#         subfolder='transformer',
#         torch_dtype=dtype,
#         device_map='cuda'
#     ),
#     torch_dtype=dtype
# ).to(device)

# qwenpipe.load_lora_weights(os.path.join(lora_dir, "qwen-delight.safetensors)",
#                        weight_name="qwen-delight",)
# qwenpipe.load_lora_weights(os.path.join(lora_dir, "qwen-relight.safetensors"),
#                        adapter_name="qwen-relight",
#                        # weight_name="qwen-relight.safetensors")

@spaces.GPU
def kontextinfer(input_image, 
                 prompt, 
                 lora_adapter='relight', 
                 seed=42, 
                 randomize_seed=False, 
                 guidance_scale=2.5, 
                 steps=30, 
                 progress=gr.Progress(track_tqdm=True)):
    """
    Performs relighting on an input image using the FLUX.1-Kontext model.
    """

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    if lora_adapter == "relight":
        kontextpipe.set_adapters(["kontext-relight"], adapter_weights=[1.0])
    elif lora_adapter == "combine":
        kontextpipe.set_adapters(["combine"], adapter_weights=[1.0])

    image = kontextpipe(
        image=input_image, 
        prompt=prompt,
        guidance_scale=guidance_scale,
        width=input_image.size[0],
        height=input_image.size[1],
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    return [input_image, image], seed, prompt

@spaces.GPU(duration=30)
def qweninfer(
    input_image,
    prompt,
    lora_adapter,
    seed,
    randomize_seed,
    guidance_scale,
    steps,
    progress=gr.Progress(track_tqdm=True)
):
    if input_image is None:
        raise gr.Error("Please upload an image to edit.")

    if lora_adapter == "Delight":
        qwenpipe.set_adapters(["delight"], adapter_weights=[1.0])
    elif lora_adapter == "Relight":
        qwenpipe.set_adapters(["relight"], adapter_weights=[1.0])
        
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"

    original_image = input_image.convert("RGB")
    
    # Use the new function to update dimensions
    width, height = update_dimensions_on_upload(original_image)

    result = qwenpipe(
        image=original_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator,
        true_cfg_scale=guidance_scale,
    ).images[0]

    return result, seed

@spaces.GPU(duration=30)
def infer_example(input_image, prompt, lora_adapter):
    input_pil = input_image.convert("RGB")
    guidance_scale = 1.0
    steps = 4
    result, seed = qweninfer(input_pil, prompt, lora_adapter, 0, True, guidance_scale, steps)
    return result, seed

css = """
#col-container {
    margin: 0 auto;
    max-width: 1020px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# Relighting Demo For RelightMoE")
        gr.Markdown("Perform diverse image relighting using Qwen-Image-Edit and FLUX.1-Kontext.")

        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil", height=290)
                prompt = gr.Text(label="Edit Prompt", placeholder="e.g., transform lighting to blue.")
                run_button = gr.Button("Edit Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Output Image", interactive=False, format="png", height=350)

                with gr.Row():
                    lora_adapter = gr.Dropdown(
                        label="Choose Editing Style",
                        choices=["Relight"],
                        value="Relight"
                    )
                    base_model = gr.Dropdown(
                        label="Choose Base Model",
                        choices=["Kontext", "Qwen"],
                        value="Kontext"
                    )

                with gr.Accordion("Advanced Settings", open=False, visible=True):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, step=1, value=4)

        status_box = gr.Textbox(label="Status", interactive=False)

    def on_base_change(model_name, current_lora):
        if model_name == "Kontext":
            choices = ["Relight"]
            value = "Relight"
            status = "Kontext activated — only Relight available."
        else:
            choices = ["Relight", "Delight"]
            value = current_lora if current_lora in choices else "Relight"
            status = "Qwen activated — Relight & Delight available."

        return gr.update(choices=choices, value=value), status

    def dispatch_infer(
        base_model,
        input_image,
        prompt,
        lora_adapter,
        seed,
        randomize_seed,
        guidance_scale,
        steps
    ):
        if base_model == "Kontext":
            return kontextinfer(
                input_image,
                prompt,
                seed,
                randomize_seed,
                guidance_scale
            )
        else:  # Qwen
            return qweninfer(
                input_image,
                prompt,
                lora_adapter,
                seed,
                randomize_seed,
                guidance_scale,
                steps
            )

    base_model.change(
        fn=on_base_change,
        inputs=[base_model, lora_adapter],
        outputs=[lora_adapter, status_box]
    )

    run_button.click(
        fn=dispatch_infer,
        inputs=[base_model, input_image, prompt, lora_adapter, seed, randomize_seed, guidance_scale, steps],
        outputs=[output_image, seed]
    )

demo.launch(mcp_server=True)

