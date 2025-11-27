import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from PIL import Image
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
def parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False):
    offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
    model_configs = []
    if model_paths is not None:
        model_paths = json.loads(model_paths)
        model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
    if model_id_with_origin_paths is not None:
        model_id_with_origin_paths = model_id_with_origin_paths.split(",")
        model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
    return model_configs

root = '/mnt/data/wangzh/wangzhen/models'
model_paths_str = '''
[
    "${root}/flux1-kontext-dev.safetensors",
    "${root}/text_encoder/model.safetensors",
    "${root}/text_encoder_2/",
    "${root}/ae.safetensors"
]
'''
model_paths = model_paths_str.replace("${root}", root)

model_id_with_origin_paths=None
model_configs = parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
pipe = FluxImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=model_configs)
pipe.load_lora(pipe.dit, "/mnt/data/wangzh/kontext/checkpoints/intensity/epoch-49.safetensors", alpha=1)

input_name = '37_0.png'
image_path = f"/mnt/data/wangzh/kontext/data/intensity/{input_name}"
image_path = '/home/wangzh/infer/i2.png'
w, h = Image.open(image_path).size
image = pipe(
    prompt="changing the intensity of the light of sunset to the weak illumination",
    kontext_images=Image.open(image_path).resize((w, h)),
    height=h, width=w, # 768
    seed=0
)
image.save("image_FLUX.1-Kontext-dev_lora.jpg")
Image.open(image_path).resize((w, h)).save("image_kontext_images.jpg")

# import random
# from tqdm import tqdm
# save_dir = '/home/wangzhen/DiffSynth-Studio/examples/flux/model_training/validate_lora/test'
# data_dir = '/oss-mt-sysu-release/wangzh/idlight/data/test/transfer'
# nums = list(i for i in range(1, 53))
# cnt = ['0', '1', '10', '12', '13', '14', '15']

# total = sum((len(cnt) - 1) * 2 for _ in cnt)  # 每个 chosen 对应 len(cnt)-1 refs, 每个2个num
# with tqdm(total=total, desc="Generating images") as pbar:
#     for chosen in cnt:
#         remaining = [x for x in cnt if x != chosen]
#         num = random.sample(nums, 2)
#         for ref in remaining:
#             for n in num:
#                 image = pipe(
#                     prompt="Transfering the light effect or the lens flare",
#                     kontext_images=Image.open(f"/oss-mt-sysu-release/wangzh/idlight/data/test/transfer/{chosen}.png").resize((768, 768)),
#                     ref_images=Image.open(f"/oss-mt-sysu-release/wangzh/idlight/data/test/transfer/{ref}_{n}.png").resize((768, 768)),
#                     height=768, width=768,
#                     seed=0
#                 )
#                 image.save(os.path.join(save_dir,f"output-{chosen}-{ref}_{n}.png" ))
