import torch
import argparse
import os
import numpy as np
import datetime
import random
import sys
from diffusers import DiffusionPipeline, QwenImageEditPipeline
import gradio as gr
from optimum.quanto import freeze, qint8, quantize
from huggingface_hub import snapshot_download, login
from PIL import Image

# --- 1. Model Configuration & Automatic Download ---

# --- Text-to-Image Model ---
T2I_MODEL_REPO = "Qwen/Qwen-Image"
MODELS_BASE_DIR = "models_downloads"
t2i_model_folder_name = T2I_MODEL_REPO.split('/')[-1]
local_t2i_model_path = os.path.join(MODELS_BASE_DIR, t2i_model_folder_name)

# --- Image-to-Image Edit Model ---
EDIT_MODEL_REPO = "Qwen/Qwen-Image-Edit"
edit_model_folder_name = EDIT_MODEL_REPO.split('/')[-1]
local_edit_model_path = os.path.join(MODELS_BASE_DIR, edit_model_folder_name)

def download_model(repo_id, local_dir, repo_name):
    """Checks for a model locally and downloads it if it doesn't exist."""
    print(f"[{repo_name}] Checking for model at: {local_dir}")
    if not os.path.exists(local_dir):
        print(f"[{repo_name}] Model not found locally. Starting download of '{repo_id}'...")
        os.makedirs(MODELS_BASE_DIR, exist_ok=True)
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
            print(f"[{repo_name}] Download complete!")
        except Exception as e:
            if "401" in str(e) or "GatedRepo" in str(e):
                print(f"\n[{repo_name}] This model requires a Hugging Face login.")
                print("Please create an access token with 'read' permissions here: https://huggingface.co/settings/tokens")
                token = input("Enter your Hugging Face access token: ").strip()
                try:
                    login(token=token)
                    print("Login successful. Retrying download...")
                    snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True)
                    print(f"[{repo_name}] Download complete!")
                except Exception as login_e:
                    print(f"\n[{repo_name}] Download failed after login: {login_e}")
                    sys.exit(1)
            else:
                print(f"\n[{repo_name}] An unexpected error occurred during download: {e}")
                sys.exit(1)
    else:
        print(f"[{repo_name}] Model found locally. Loading from cache.")

# Download both models
download_model(T2I_MODEL_REPO, local_t2i_model_path, "Text-to-Image")
download_model(EDIT_MODEL_REPO, local_edit_model_path, "Image-Edit")


# --- 2. Application Setup ---

parser = argparse.ArgumentParser()
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP address, set to 0.0.0.0 for LAN access")
parser.add_argument("--server_port", type=int, default=7891, help="Port to use")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
parser.add_argument('--vram', type=str, default='high', choices=['low', 'high'], help='VRAM usage mode (affects T2I model)')
parser.add_argument('--lora', type=str, default="None", help='Path to LoRA model (for T2I model)')
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
    # Use bfloat16 for Ampere architecture and newer, float16 for older.
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

MAX_SEED = np.iinfo(np.int32).max
os.makedirs("outputs", exist_ok=True)

# --- 3. Load the Models ---

# Load Text-to-Image pipeline
print(f"Loading Text-to-Image pipeline from '{local_t2i_model_path}'...")
pipe = DiffusionPipeline.from_pretrained(local_t2i_model_path, torch_dtype=dtype)
if args.lora != "None":
    pipe.load_lora_weights(args.lora)
    print(f"Loaded LoRA: {args.lora}")

if args.vram == "high":
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
else: # 'low' vram mode
    quantize(pipe.transformer, qint8)
    freeze(pipe.transformer)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

print("Text-to-Image pipeline loaded successfully.")

# We will load the Edit pipeline "lazily" (on first use) to save VRAM on startup
edit_pipe = None
print("Image-Edit pipeline will be loaded on first use.")


# --- 4. Gradio UI and Logic ---

ASPECT_RATIOS = {
    "1:1 Square": (1328, 1328),
    "16:9 Landscape": (1664, 928),
    "9:16 Portrait": (928, 1664),
    "4:3 Landscape": (1472, 1140),
    "3:4 Portrait": (1140, 1472)
}

# --- Text-to-Image Generation Function ---
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
    prompt_parts = []
    if user_prompt:
        prompt_parts.append(user_prompt.strip())
    if style_suffix:
        prompt_parts.append(style_suffix.strip())
    prompt = ", ".join(filter(None, prompt_parts))

    if resolution_mode == "Preset":
        width, height = ASPECT_RATIOS[aspect_ratio_key]
    else:
        width, height = custom_width, custom_height

    if seed_param < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed_param)

    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"T2I Generating: '{prompt}'")
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{timestamp}_{seed}_t2i.png"
    image.save(output_path)
    return output_path, seed

# --- Image-to-Image Edit Function ---
def generate_edit(
    input_image,
    edit_prompt,
    num_inference_steps,
    guidance_scale,
    seed_param
):
    global edit_pipe # Use the global variable for the pipeline

    # Lazy loading: If the pipeline isn't loaded yet, load it now.
    if edit_pipe is None:
        print(f"Loading Image-Edit pipeline for the first time from '{local_edit_model_path}'...")
        edit_pipe = QwenImageEditPipeline.from_pretrained(
            local_edit_model_path,
            torch_dtype=dtype # Use the same dtype as the T2I pipe
        )
        edit_pipe.to(device)
        # Always enable cpu offload for the edit pipe to conserve VRAM
        edit_pipe.enable_model_cpu_offload()
        print("Image-Edit pipeline loaded successfully.")

    if input_image is None:
        raise gr.Error("You must upload an input image for editing.")

    if seed_param < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed_param)

    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Image Edit Generating: '{edit_prompt}'")
    print(f"Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")

    output = edit_pipe(
        image=input_image.convert("RGB"), # Ensure image is in RGB format
        prompt=edit_prompt,
        generator=generator,
        true_cfg_scale=guidance_scale,
        negative_prompt=" ", # As per example
        num_inference_steps=num_inference_steps,
    )
    output_image = output.images[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{timestamp}_{seed}_edit.png"
    output_image.save(output_path)

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
                <h2 style="font-size: 30px;text-align: center;">Qwen-Image Suite</h2>
            </div>
            """)
    with gr.TabItem("Text-to-Image"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Your Prompt (What you want to see)", value="A majestic lion king on a cliff overlooking the savannah")
                style_suffix = gr.Textbox(label="Style Modifiers (Appended to your prompt)", value="Ultra HD, 4K, cinematic composition")
                negative_prompt = gr.Textbox(label="Negative Prompt", value="lowres, bad anatomy, bad hands, cropped, worst quality")

                with gr.Group():
                    resolution_mode = gr.Radio(
                        label="Resolution Mode", choices=["Preset", "Custom"], value="Preset"
                    )
                    aspect_ratio = gr.Radio(
                        label="Aspect Ratio", choices=list(ASPECT_RATIOS.keys()), value="16:9 Landscape", visible=True
                    )
                    custom_width = gr.Slider(
                        label="Custom Width", minimum=256, maximum=2656, step=32, value=1328, visible=False
                    )
                    custom_height = gr.Slider(
                        label="Custom Height", minimum=256, maximum=2656, step=32, value=1328, visible=False
                    )

                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, step=1, value=50)
                guidance_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=10, step=0.1, value=4.0)
                seed_param = gr.Number(label="Seed (Enter a positive integer, -1 for random)", value=-1, precision=0)
                generate_button = gr.Button("🎬 Generate", variant='primary')
            with gr.Column():
                image_output = gr.Image(label="Generated Image")
                seed_output = gr.Textbox(label="Seed")

    with gr.TabItem("Image-to-Image Edit"):
        with gr.Row():
            with gr.Column():
                edit_input_image = gr.Image(label="Input Image", type="pil", height=400)
                edit_prompt = gr.Textbox(label="Edit Prompt (e.g., 'Change the sky to a starry night')", value="Change the rabbit's color to purple, with a flash light background.")
                edit_num_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, step=1, value=50)
                edit_cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=10, step=0.1, value=4.0)
                edit_seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                edit_generate_button = gr.Button("🎨 Generate Edit", variant='primary')
            with gr.Column():
                edit_image_output = gr.Image(label="Edited Image")
                edit_seed_output = gr.Textbox(label="Seed")


    # --- Event Listeners ---
    resolution_mode.change(
        fn=update_resolution_controls,
        inputs=resolution_mode,
        outputs=[aspect_ratio, custom_width, custom_height]
    )

    generate_button.click(
        fn=generate,
        inputs=[
            prompt, style_suffix, negative_prompt, resolution_mode, aspect_ratio,
            custom_width, custom_height, num_inference_steps, guidance_scale, seed_param,
        ],
        outputs=[image_output, seed_output]
    )

    edit_generate_button.click(
        fn=generate_edit,
        inputs=[
            edit_input_image, edit_prompt, edit_num_steps, edit_cfg_scale, edit_seed
        ],
        outputs=[edit_image_output, edit_seed_output]
    )


if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=True,
    )
