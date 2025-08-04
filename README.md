<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

## Introduction
We are thrilled to release **Qwen-Image**, a 20B MMDiT image foundation model that achieves significant advances in **complex text rendering** and **precise image editing**. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.


![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/bench.png#center)

## News
- 2025.08.04: We released the [Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf) of Qwen-Image!
- 2025.08.04: We released Qwen-Image weights! Check at [Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [Modelscope](https://modelscope.cn/models/Qwen/Qwen-Image)!
- 2025.08.04: We released Qwen-Image! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!
- The editing version of Qwen-Image will be released soon. Stay tuned!
- Due to heavy traffic, if you'd like to experience our demo online, we also recommend visiting DashScope, WaveSpeed, and LibLib. Please find the links below in the community support.


## Start with the App


✅ Install using Miniconda or Anaconda:

```
git clone https://github.com/tpc2233/Qwen-Image.git
cd Qwen-Image

conda create -n QwenImage python=3.12.9 -y 
conda activate QwenImage
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 
pip install -r requirements.txt
```

✅ Run the App:

```
python App_t.py
```






## Quick Start

1. Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)

2. Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

The following contains a code snippet illustrating how to use the model to generate images based on text prompts:

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": "Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": "超清，4K，电影级构图" # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

## Show Cases

One of its standout capabilities is high-fidelity text rendering across diverse images. Whether it's alphabetic languages like English or logographic scripts like Chinese, Qwen-Image preserves typographic details, layout coherence, and contextual harmony with stunning accuracy. Text isn't just overlaid—it's seamlessly integrated into the visual fabric.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

Beyond text, Qwen-Image excels at general image generation with support for a wide range of artistic styles. From photorealistic scenes to impressionist paintings, from anime aesthetics to minimalist design, the model adapts fluidly to creative prompts, making it a versatile tool for artists, designers, and storytellers.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

When it comes to image editing, Qwen-Image goes far beyond simple adjustments. It enables advanced operations such as style transfer, object insertion or removal, detail enhancement, text editing within images, and even human pose manipulation—all with intuitive input and coherent output. This level of control brings professional-grade editing within reach of everyday users.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

But Qwen-Image doesn't just create or edit, it understands. It supports a suite of image understanding tasks, including object detection, semantic segmentation, depth and edge (Canny) estimation, novel view synthesis, and super-resolution. These capabilities, while technically distinct, can all be seen as specialized forms of intelligent image editing, powered by deep visual comprehension.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

Together, these features make Qwen-Image not just a tool for generating pretty pictures, but a comprehensive foundation model for intelligent visual creation and manipulation—where language, layout, and imagery converge.

### Advanced Usage

#### Prompt Enhance
For enhanced prompt optimization and multi-language support, we recommend using our official Prompt Enhancement Tool powered by Qwen-Plus .

You can integrate it directly into your code:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)

```

Alternatively, run the example script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```


## Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server will start a Gradio-based web interface with:
- Multi-GPU parallel processing
- Queue management for high concurrency
- Automatic prompt optimization
- Support for multiple aspect ratios

Configuration via environment variables:
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```


## AI Arena

To comprehensively evaluate the general image generation capabilities of Qwen-Image and objectively compare it with state-of-the-art closed-source APIs, we introduce [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform built on the Elo rating system. AI Arena provides a fair, transparent, and dynamic environment for model evaluation.

In each round, two images—generated by randomly selected models from the same prompt—are anonymously presented to users for pairwise comparison. Users vote for the better image, and the results are used to update both personal and global leaderboards via the Elo algorithm, enabling developers, researchers, and the public to assess model performance in a robust and data-driven way. AI Arena is now publicly available, welcoming everyone to participate in model evaluations. 

![AI Arena](assets/figure_aiarena_website.png)

The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image)

If you wish to deploy your model on AI Arena and participate in the evaluation, please contact weiyue.wy@alibaba-inc.com.

## Community Support

### Huggingface

Diffusers has supported Qwen-Image since day 0. Support for LoRA and finetuning workflows is currently in development and will be available soon.

### Modelscope
* **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** provides comprehensive support for Qwen-Image, including low-GPU-memory layer-by-layer offload (inference within 4GB VRAM), FP8 quantization, LoRA / full training.
* **[DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)** delivers advanced optimizations for Qwen-Image inference and deployment, including FBCache-based acceleration, classifier-free guidance (CFG) parallel, and more.
* **[ModelScope AIGC Central](https://www.modelscope.cn/aigc)** provides hands-on experiences on Qwen Image, including: 
    - [Image Generation](https://www.modelscope.cn/aigc/imageGeneration): Generate high fidelity images using the Qwen Image model.
    - [LoRA Training](https://www.modelscope.cn/aigc/modelTraining): Easily train Qwen Image LoRAs for personalized concepts.

### WaveSpeedAI

WaveSpeed has deployed Qwen-Image on their platform from day 0, visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image) for more details.

### LiblibAI

LiblibAI offers native support for Qwen-Image from day 0. Visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c) page for more details and discussions.


## License Agreement

Qwen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@article{qwen-image,
    title={Qwen-Image Technical Report}, 
    author={Qwen Team},
    journal={arXiv preprint},
    year={2025}
}
```


## Contact and Join Us


If you'd like to get in touch with our research team, we'd love to hear from you! Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png) — we're always open to discussion and collaboration.

If you have questions about this repository, feedback to share, or want to contribute directly, we welcome your issues and pull requests on GitHub. Your contributions help make Qwen-Image better for everyone. 

If you're passionate about fundamental research, we're hiring full-time employees (FTEs) and research interns. Don't wait — reach out to us at fulai.hr@alibaba-inc.com




