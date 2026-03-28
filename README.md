# GPT-5.4 → Qwen 3.5-9B Computer Use Distillation Pipeline

Distill GPT-5.4's web browsing capabilities into Qwen 3.5-9B via trajectory-based
supervised fine-tuning (SFT). Based on the "Distilling GPT-5.4 Computer Use into
Qwen 3.5-9B" blueprint.

## Overview

The pipeline has 3 stages:

1. **Load tasks** — Pull web browsing tasks from the Mind2Web dataset
2. **Record trajectories** — GPT-5.4 CUA performs each task in a Playwright browser
   while we record screenshots + actions at every step
3. **Train** — Convert trajectories to ms-swift format and fine-tune Qwen 3.5-9B-Base
   with LoRA

## Prerequisites

- Python 3.10+
- OpenAI API key with GPT-5.4 CUA access
- For training: GPU with 16-24GB+ VRAM, ms-swift installed

## Setup

    pip install -r requirements.txt
    playwright install chromium
    export OPENAI_API_KEY=your_key_here

For training (on GPU machine):

    pip install ms-swift[llm]

## Usage

### Step 1: Load tasks from Mind2Web

    python load_tasks.py --output tasks.jsonl
    python load_tasks.py --output tasks.jsonl --limit 100  # limit for testing

### Step 2: Record trajectories

    python record_trajectories_from_gpt_api.py --tasks tasks.jsonl --output trajectories/

Options:
- `--max-steps 15` — max actions per task (default: 15)
- `--viewport 1280x720` — browser viewport size (default: 1280x720)
- `--resume` — skip tasks that already have a trajectory directory

### Step 3: Convert trajectories to training format

    python train_qwen.py convert --input trajectories/ --output training_data/

Options:
- `--val-split 0.1` — fraction of data for validation (default: 0.1)

### Step 4: Train Qwen 3.5-9B

    python train_qwen.py train --config config.yaml --dataset training_data/sft_dataset.jsonl

Options:
- `--epochs 3` — override number of epochs
- `--lr 5e-5` — override learning rate
- `--batch-size 1` — override batch size

## Output Structure

    trajectories/
      task_mind2web_0001/
        metadata.json
        step_001_screenshot.png
        step_002_screenshot.png
        ...
        trajectory.json

## Training Configuration

See `config.yaml` for default hyperparameters. Key settings from the blueprint:
- Model: Qwen/Qwen3.5-9B-Base
- LoRA rank 64, alpha 128
- Learning rate 5e-5, 3 epochs
- Max sequence length 4096
- Vision encoder frozen (freeze_vit: true)

## Cost Estimates

- ~$0.55 per trajectory (GPT-5.4 API)
- 1,000 trajectories ≈ $550
- Training: 20-40 GPU hours on A100 80GB (~$16-32 on RunPod)
