"""
Run our method on a composition of customconcept101.
"""

import argparse
import json
import torch
from diffusers import DPMSolverMultistepScheduler
from pipeline_stable_diffusion import StableDiffusionPipeline
from utils import ptp_utils
from utils.ptp_utils import AttentionStore
from pathlib import Path


def get_token_indices(prompt, instance_name, tokenizer):
    prompt_ids = tokenizer(prompt)['input_ids']
    # print({idx: pipe.tokenizer.decode(t)
    #     for idx, t in enumerate(prompt_ids) if 0 < idx < len(prompt_ids) - 1})
    # print(instance_name)
    instance_ids = tokenizer(instance_name)['input_ids'][1:-1]  # <|startoftext|>, <|endoftext|>
    token_indices = []
    for i in range(1, len(prompt_ids) - len(instance_ids)):
        if prompt_ids[i:i + len(instance_ids)] == instance_ids:
            for j in range(i, i + len(instance_ids)):
                token_indices.append(j)
    # print(token_indices)
    return token_indices

def parse_args():
    parser = argparse.ArgumentParser("sample", add_help=False)
    parser.add_argument("--composition_num", type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    composition_num = args.composition_num  # 0 - 87
    cfg_scale = [7, 7, 7]  # cfg weights
    weights = [0.2, 0.8, 0.8]
    lora_scale = [0.7, 0.7]
    dataset_dir = Path('path/to/customconcept101')
    sd_path = Path('path/to/stable-diffusion-v1-5')
    lora_dir = Path('path/to/loras-dir')
    output_dir = Path('path/to/result-dir')
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(16))

    with open(dataset_dir / 'dataset_multiconcept.json', 'r') as f:
        dataset = json.load(f)

    pipe = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None)
    pipe.safety_checker = None
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, solver_order=2
    )

    pipe1 = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None)
    pipe1.safety_checker = None
    pipe1.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe1.scheduler.config, use_karras_sigmas=True, solver_order=2
    )

    pipe2 = StableDiffusionPipeline.from_pretrained(sd_path, safety_checker=None)
    pipe2.safety_checker = None
    pipe2.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe2.scheduler.config, use_karras_sigmas=True, solver_order=2
    )

    controller1 = AttentionStore()
    ptp_utils.register_attention_control(pipe1.unet, controller1)
    controller2 = AttentionStore()
    ptp_utils.register_attention_control(pipe2.unet, controller2)
    controller_list = [controller1, controller2]

    pipe.register_additional_models(pipe1)
    pipe.register_additional_models(pipe2)
    
    composition = dataset[composition_num]

    instance1_data_dir = dataset_dir / composition[0]["instance_data_dir"][2:]
    instance2_data_dir = dataset_dir / composition[1]["instance_data_dir"][2:]
    instance1_name = str(instance1_data_dir).split('/')[-1]
    instance2_name = str(instance2_data_dir).split('/')[-1]

    outdir = output_dir / f'{instance1_name}+{instance2_name}' / f'lora_{lora_scale[0]}_{lora_scale[1]}_cfg_{cfg_scale[0]}_{cfg_scale[1]}_{cfg_scale[2]}_weights_{weights[0]}_{weights[1]}_{weights[2]}'
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load loras
    lora1_path = lora_dir / f'{instance1_name}.safetensors'
    lora2_path = lora_dir / f'{instance2_name}.safetensors'
    pipe1.load_lora_weights(".", weight_name=str(lora1_path))
    pipe1.fuse_lora(lora_scale=0.7)
    pipe2.load_lora_weights(".", weight_name=str(lora2_path))
    pipe2.fuse_lora(lora_scale=0.7)

    # Get benchmark prompts
    with open(dataset_dir / composition[2]["prompt_filename_compose"], 'r') as f:
        prompts = f.read().splitlines()
    
    # Generate images for each prompt
    for prompt in prompts:
        print(prompt)
        outdir_ = outdir / prompt
        outdir_.mkdir(parents=True, exist_ok=True)

        # Construct the sub-prompts
        prompt1 = prompt
        prompt2 = prompt
        prompt = prompt.replace('{0}', composition[0]["class_prompt"])
        prompt = prompt.replace('{1}', composition[1]["class_prompt"])
        prompt1 = prompt1.replace('{0}', instance1_name)
        prompt1 = prompt1.replace('{1}', composition[1]["class_prompt"])
        prompt1 = prompt1.replace('.', f', a {instance1_name}.')
        prompt2 = prompt2.replace('{1}', instance2_name)
        prompt2 = prompt2.replace('{0}', composition[0]["class_prompt"])
        prompt2 = prompt2.replace('.', f', a {instance2_name}.')
        print(prompt)
        print(prompt1)
        print(prompt2)
        
        # Calc the token indices
        instance1_token_indices = get_token_indices(prompt1, instance1_name, pipe.tokenizer)
        instance2_token_indices = get_token_indices(prompt2, instance2_name, pipe.tokenizer)
        print(instance1_token_indices)
        print(instance2_token_indices)
        
        token_indices = [
            [-1],
            instance1_token_indices,
            instance2_token_indices
        ]

        prompt_ = [prompt, prompt1, prompt2]

        # For each seed, generate an image
        for seed in seeds:
            generator = torch.Generator(device='cuda').manual_seed(seed)
            image = pipe(prompt_, token_indices, cfg_scale, weights, controller_list, 
                    run_standard_sd=False, 
                    num_inference_steps=30,
                    max_iter_to_alter=25,
                    aggregate_attn_map=True,
                    loss_no=3,
                    alpha=0.83,
                    generator=generator, 
                    negative_prompt="",
                    scale_factor=20,
                    ).images[0]
            image.save(outdir_ / f'{seed}.png')
        
    print('Done!')
