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
    
    # Ensure the parent directory exists
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    
    try:
        # Attempt to download the model
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=local_model_path,
            resume_download=True
        )
        print("Download complete!")
    except Exception as e:
        # Handle the common case of needing to log in for gated models
        if "401" in str(e) or "GatedRepo" in str(e):
            print("\nThis model requires a Hugging Face login.")
            print("Please create an access token with 'read' permissions here: https://huggingface.co/settings/tokens")
            
            # Prompt user for their token
            token = input("Enter your Hugging Face access token: ").strip()
            
            try:
                # Log in and retry the download
                login(token=token)
                print("Login successful. Retrying download...")
                snapshot_download(
                    repo_id=MODEL_REPO,
                    local_dir=local_model_path,
                    resume_download=True
                )
                print("Download complete!")
            except Exception as login_e:
                print(f"\nDownload failed after login: {login_e}")
                print("Please check your token and network connection. Exiting.")
                sys.exit(1) # Exit if we can't get the model
        else:
            # Handle other potential download errors
            print(f"\nAn unexpected error occurred during download: {e}")
            print("Please check your network connection. Exiting.")
            sys.exit(1) # Exit if we can't get the model
else:
    print("Model found locally. Loading from cache.")

# --- 2. Application Setup ---

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument('--vram', type=str, default='high', choices=['low', 'high'], help='显存模式')
parser.add_argument('--lora', type=str, default="None", help='lora模型路径')
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
pipe = DiffusionPipeline.from_pretrained(
    local_model_path, # Use the path to the downloaded model
    torch_dtype=dtype,
)
if args.lora!="None":
    pipe.load_lora_weights(args.lora)
    print(f"加载{args.lora}")

if args.vram=="high":
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
else:
    quantize(pipe.transformer, qint8)
    freeze(pipe.transformer)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

print("Pipeline loaded successfully. Starting Gradio interface...")

# --- 4. Gradio UI and Logic (Unchanged) ---

def generate(
    prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    true_cfg_scale, 
    seed_param,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param < 0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = int(seed_param)
        
    generator = torch.Generator(device=device).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=true_cfg_scale, # Note: This parameter is often called guidance_scale
        generator=generator
    ).images[0]
    
    output_path = f"outputs/{timestamp}_{seed}.png"
    image.save(output_path)
    return output_path, seed
    

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Qwen-Image</h2>
            </div>
            """)
    with gr.TabItem("Qwen-Image文生图"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt = gr.Textbox(label="负面提示词", value="")
                width = gr.Slider(label="宽度（推荐1328x1328、1664x928、1472x1140）", minimum=256, maximum=2656, step=32, value=1328)
                height = gr.Slider(label="高度", minimum=256, maximum=2656, step=32, value=1328)
                num_inference_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, step=1, value=50)
                # Corrected the name to match the diffusers pipeline parameter
                guidance_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=10, step=0.1, value=4.0)
                seed_param = gr.Number(label="种子，请输入正整数，-1为随机", value=-1, precision=0)
                generate_button = gr.Button("🎬 开始生成", variant='primary')
            with gr.Column():
                image_output = gr.Image(label="生成图片")
                seed_output = gr.Textbox(label="种子")

    generate_button.click(
        fn=generate,
        inputs=[
            prompt,
            negative_prompt,
            width,
            height,
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
