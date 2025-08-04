import torch
import argparse
import os
import numpy as np
import datetime
import random
import sys
from diffusers import DiffusionPipeline
import gradio as gr
from optimum.quanto import freeze, qint8, quantize
from huggingface_hub import snapshot_download, login

# --- 1. Model Configuration & Automatic Download ---

# Define the model repository on Hugging Face and the local download directory
MODEL_REPO = "Qwen/Qwen-Image"
MODELS_BASE_DIR = "models_downloads" # This folder will be created in your current directory

# Construct the full local path for the model, e.g., "models_downloads/Qwen-Image"
model_folder_name = MODEL_REPO.split('/')[-1]
local_model_path = os.path.join(MODELS_BASE_DIR, model_folder_name)

# Check if the model exists locally, if not, download it
print(f"Checking for model at: {local_model_path}")
if not os.path.exists(local_model_path):
    print(f"Model not found locally. Starting download of '{MODEL_REPO}'...")
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    try:
        snapshot_download(repo_id=MODEL_REPO, local_dir=local_model_path, resume_download=True)
        print("Download complete!")
    except Exception as e:
        if "401" in str(e) or "GatedRepo" in str(e):
            print("\nThis model requires a Hugging Face login.")
            print("Please create an access token with 'read' permissions here: https://huggingface.co/settings/tokens")
            token = input("Enter your Hugging Face access token: ").strip()
            try:
                login(token=token)
                print("Login successful. Retrying download...")
                snapshot_download(repo_id=MODEL_REPO, local_dir=local_model_path, resume_download=True)
                print("Download complete!")
            except Exception as login_e:
                print(f"\nDownload failed after login: {login_e}")
                sys.exit(1)
        else:
            print(f"\nAn unexpected error occurred during download: {e}")
            sys.exit(1)
else:
    print("Model found locally. Loading from cache.")

# --- 2. Application Setup ---

parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP address, set to 0.0.0.0 for LAN access")
parser.add_argument("--server_port", type=int, default=7891, help="Port to use")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument('--vram', type=str, default='high', choices=['low', 'high'], help='VRAM usage mode')
parser.add_argument('--lora', type=str, default="None", help='Path to LoRA model')
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

MAX_SEED = np.iinfo(np.int32).max
os.makedirs("outputs", exist_ok=True)

# --- 3. Load the Model from the Local Path ---

print(f"Loading diffusion pipeline from '{local_model_path}'...")
pipe = DiffusionPipeline.from_pretrained(local_model_path, torch_dtype=dtype)
if args.lora != "None":
    pipe.load_lora_weights(args.lora)
    print(f"Loaded LoRA: {args.lora}")

if args.vram == "high":
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
else:
    quantize(pipe.transformer, qint8)
    freeze(pipe.transformer)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

print("Pipeline loaded successfully. Starting Gradio interface...")

# --- 4. Gradio UI and Logic ---

ASPECT_RATIOS = {
    "1:1 Square": (1328, 1328),
    "16:9 Landscape": (1664, 928),
    "9:16 Portrait": (928, 1664),
    "4:3 Landscape": (1472, 1140),
    "3:4 Portrait": (1140, 1472)
}

def generate(
    user_prompt,
    style_suffix,
    negative_prompt,
    resolution_mode,
    aspect_ratio_key,
    custom_width,
    custom_height,
    num_inference_steps,
    guidance_scale,
    seed_param,
):
    # Combine the user prompt and the style suffix
    prompt_parts = []
    if user_prompt:
        prompt_parts.append(user_prompt.strip())
    if style_suffix:
        prompt_parts.append(style_suffix.strip())
    prompt = ", ".join(filter(None, prompt_parts))

    # Determine resolution based on user's choice
    if resolution_mode == "Preset":
        width, height = ASPECT_RATIOS[aspect_ratio_key]
    else: # Custom
        width, height = custom_width, custom_height

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed_param)

    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating image with full prompt: '{prompt}'")
    print(f"Dimensions: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    output_path = f"outputs/{timestamp}_{seed}.png"
    image.save(output_path)
    return output_path, seed

def update_resolution_controls(mode):
    is_preset = (mode == "Preset")
    return {
        aspect_ratio: gr.update(visible=is_preset),
        custom_width: gr.update(visible=not is_preset),
        custom_height: gr.update(visible=not is_preset),
    }

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Qwen-Image</h2>
            </div>
            """)
    with gr.TabItem("Text-2-Image"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Your Prompt (What you want to see)", value="A majestic lion king on a cliff overlooking the savannah")
                style_suffix = gr.Textbox(label="Style Modifiers (Appended to your prompt)", value="Ultra HD, 4K, cinematic composition")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="lowres, bad anatomy, bad hands, cropped, worst quality")

                with gr.Group():
                    resolution_mode = gr.Radio(
                        label="Resolution Mode",
                        choices=["Preset", "Custom"],
                        value="Preset"
                    )
                    aspect_ratio = gr.Radio(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="16:9 Landscape",
                        visible=True
                    )
                    custom_width = gr.Slider(
                        label="Custom Width",
                        minimum=256, maximum=2656, step=32, value=1328,
                        visible=False
                    )
                    custom_height = gr.Slider(
                        label="Custom Height",
                        minimum=256, maximum=2656, step=32, value=1328,
                        visible=False
                    )

                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, step=1, value=50)
                guidance_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=10, step=0.1, value=4.0)
                seed_param = gr.Number(label="Seed (Enter a positive integer, -1 for random)", value=-1, precision=0)
                generate_button = gr.Button("🎬 Generate", variant='primary')
            with gr.Column():
                image_output = gr.Image(label="Generated Image")
                seed_output = gr.Textbox(label="Seed")


    with gr.TabItem("Image-2-image Edit"):
        gr.Markdown(
            """
            <div style='display: flex; justify-content: center; align-items: center; height: 50vh;'>
                <h2 style='font-size: 24px; color: #888;'>Waiting to be release...https://github.com/QwenLM/Qwen-Image/issues/3#issuecomment-3151573614</h2>
            </div>
            """
        )


    resolution_mode.change(
        fn=update_resolution_controls,
        inputs=resolution_mode,
        outputs=[aspect_ratio, custom_width, custom_height]
    )

    generate_button.click(
        fn=generate,
        inputs=[
            prompt,
            style_suffix,
            negative_prompt,
            resolution_mode,
            aspect_ratio,
            custom_width,
            custom_height,
            num_inference_steps,
            guidance_scale,
            seed_param,
        ],
        outputs=[image_output, seed_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )
