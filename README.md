---
license: mit
library_name: pytorch
pipeline_tag: text-to-video
language:
  - en
tags:
  - xens
  - xen
  - pytorch
  - custom-architecture
---

# XEN-GEN1-V

XEN-GEN1-V is the **video generation model** in the XEN family. It is designed for local text-to-video generation with a small video diffusion architecture.

## What it can do

- Text-to-video generation
- Prompt-conditioned video generation
- Image-to-video generation using a reference image
- Adjustable motion/variation strength
- Local CUDA/CPU inference
- Train on your own video + caption dataset
- MP4 output
- 1–15 second generation interface
- Hugging Face-compatible packaging

## Supported platforms

- Windows
- Linux
- macOS

## Requirements

Recommended development hardware:

- NVIDIA GPU with CUDA support
- RTX 4060 8GB is the current development baseline
- 32GB RAM recommended
- Python 3.10+
- PyTorch 2.4+

### Windows

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system uses `python` for Python 3, you can use `python` instead of `python3`.

## Generate a video

After you have a trained checkpoint in `outputs/xen-gen1-v`, run:

```bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a red ball rolling across a wooden table" --duration 5 --fps 8
```

The generated video is saved as an MP4 in the model output directory according to the inference script.

### Generation examples

5-second video:

```bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a red ball rolling across a wooden table" --duration 5 --fps 8
```

10-second video:

```bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a cinematic drone shot of a futuristic city at night" --duration 10 --fps 8
```

15-second video:

```bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --prompt "a golden eagle flying above a mountain valley" --duration 15 --fps 8
```

The `--duration` value accepts 1–15 seconds. Longer videos require more memory and computation because more frames are generated.

### Important

XEN-GEN1-V is a from-scratch research/development model. The repository does **not** contain a pretrained commercial checkpoint. You need to train the model first or provide a compatible checkpoint.

## Train the model

Create:

`datasets/video_data.jsonl`

Example:

```json
{"video":"datasets/videos/cat.mp4","caption":"a small orange cat walking through a garden"}
{"video":"datasets/videos/car.mp4","caption":"a red sports car driving through a city at night"}
```

Put the MP4 files at the paths used by the JSONL file.

Start training:

```bash
python training/video_train.py --data datasets/video_data.jsonl --output outputs/xen-gen1-v --frames 16 --size 128 --batch-size 1 --steps 10000 --grad-accumulation 8
```

The initial target is 16 frames at 128x128 because this is much more practical for an RTX 4060 8GB than a large video model.
The training script accumulates gradients over `--grad-accumulation` micro-batches and writes resumable native PyTorch checkpoints every 500 optimizer steps. Resume with `--resume path/to/checkpoint-N.pt` when a real dataset and suitable runtime are available. This repository is currently **not yet trained**; training is pending a real dataset and CUDA-capable environment.

## Dataset format

Each line must contain:

- `video`: path to an MP4/video file
- `caption`: text description of the video

Example:

```json
{"video":"datasets/videos/dog.mp4","caption":"a brown dog running across a green field"}
```

## Local usage flow

1. Install Python dependencies.
2. Prepare video/caption JSONL data.
3. Train XEN-GEN1-V.
4. Keep the resulting checkpoint in `outputs/xen-gen1-v`.
5. Run `inference/video_generate.py` with a prompt.
6. Open the generated MP4.

## Hugging Face

XEN-GEN1-V can be packaged for a Hugging Face Space. A GPU Space is recommended because video diffusion is significantly more computationally expensive than text generation.

## Image-to-video

Use `--image` to turn a reference image into a short generated video. The current implementation uses the reference image as the visual starting point and preserves the reference composition in the first frame. This is an initial GEN1-V feature; stronger motion consistency will require dedicated image-to-video training.

Example:

```bash
python inference/video_generate.py --model-dir outputs/xen-gen1-v --image input.png --prompt "the subject slowly moves through a futuristic city" --duration 5 --fps 8 --strength 0.65
```

Lower `--strength` keeps more of the reference appearance. Higher values allow more variation.

## Current scope

GEN1-V supports text-to-video and an initial image-to-video workflow. Image-to-video quality will improve as dedicated reference-conditioned training is added.

## Performance and VRAM

The 1–15 second interface does not mean every duration will fit comfortably on every GPU. Memory usage increases with frame count and resolution. If generation runs out of VRAM, use a shorter duration, lower resolution, or fewer frames where supported.

## Folder structure

```text
xen-gen1-v/
├── configs/
│   └── system_prompt_v.txt
├── datasets/
│   └── video_data.jsonl
├── inference/
│   └── video_generate.py
├── model/
│   ├── tokenizer.py
│   ├── video_conditioner.py
│   └── video_model.py
├── training/
│   └── video_train.py
├── requirements.txt
└── README.md
```

## License

MIT
