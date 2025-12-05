import os
import wandb
from ultralytics import YOLO
from huggingface_hub import HfApi, create_repo
import torch
import yaml
from pathlib import Path
from datetime import datetime

# ================================
# Configuration
# ================================
class Config:
    # Paths
    DATASET_PATH = '/workspace/data/yolo/data.yaml'
    OUTPUT_DIR = '/workspace/USIS10K/work_dirs/yolo'
    
    # Model Selection - Try different models
    MODEL_VARIANTS = {
        'nano': 'yolo11n-seg.pt',
        'small': 'yolo11s-seg.pt',
        'medium': 'yolo11m-seg.pt',
        'large': 'yolo11l-seg.pt',
        'xlarge': 'yolo11x-seg.pt'
    }
    
    # Training Parameters
    MODEL_SIZE = 'medium'  # Change to: nano, small, medium, large, xlarge
    EPOCHS = 100
    BATCH_SIZE = 16  # Increased as requested
    IMG_SIZE = 640
    DEVICE = "cuda"
    
    # Weights & Biases
    WANDB_PROJECT = "USIS10K-Underwater-Segmentation"
    WANDB_ENTITY = None  # Your W&B username (or None for default)
    WANDB_NAME = f"yolo11{MODEL_SIZE[0]}-usis10k-{datetime.now().strftime('%Y%m%d-%H%M')}"
    
    # Hugging Face - UPDATE THIS WITH YOUR USERNAME!
    HF_USERNAME = "your-username"  # <--- CHANGE THIS to your HF username
    HF_REPO_NAME = "yolo11-usis10k-underwater-segmentation"
    HF_MODEL_NAME = f"yolo11{MODEL_SIZE[0]}-usis10k"

# ================================
# Setup Weights & Biases
# ================================
def setup_wandb():
    """Initialize Weights & Biases tracking"""
    print("🔧 Setting up Weights & Biases...")
    
    # Login to W&B (run 'wandb login' in terminal first)
    try:
        wandb.login()
        print("✓ W&B authentication successful")
    except Exception as e:
        print(f"⚠️  W&B login failed: {e}")
        print("Run 'wandb login' in terminal to authenticate")
        return False
    
    return True

# ================================
# Train Multiple Models
# ================================
def train_model(model_size='medium', use_wandb=True):
    """
    Train YOLO model with professional settings
    
    Args:
        model_size: nano, small, medium, large, or xlarge
        use_wandb: Enable Weights & Biases tracking
    """
    config = Config()
    config.MODEL_SIZE = model_size
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Training: YOLO11-{model_size.upper()}")
    print(f"{'='*60}\n")
    
    # Initialize model
    pretrained_weights = config.MODEL_VARIANTS[model_size]
    model = YOLO(pretrained_weights)
    
    # Training arguments
    train_args = {
        'data': config.DATASET_PATH,
        'epochs': config.EPOCHS,
        'batch': config.BATCH_SIZE,
        'imgsz': config.IMG_SIZE,
        'device': config.DEVICE,
        
        # Optimization
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Regularization
        'dropout': 0.0,
        'label_smoothing': 0.0,
        
        # Data Augmentation (Underwater-optimized)
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,  # No vertical flip for underwater
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Training settings
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'workers': 8,
        'project': config.OUTPUT_DIR,
        'name': f'finetune_{model_size}',
        'exist_ok': True,
        
        # Validation & Visualization
        'val': True,
        'plots': True,
        'verbose': True,
        
        # Close mosaic augmentation
        'close_mosaic': 10,  # Disable mosaic last 10 epochs
    }
    
    # Add W&B integration if enabled
    if use_wandb:
        train_args['project'] = config.WANDB_PROJECT
        train_args['name'] = config.WANDB_NAME
    
    # Start training
    print(f"📊 Training with batch size: {config.BATCH_SIZE}")
    print(f"📊 Total epochs: {config.EPOCHS}")
    print(f"🎯 Model: {pretrained_weights}")
    
    results = model.train(**train_args)
    
    # Get best model path
    best_model_path = Path(config.OUTPUT_DIR) / f'finetune_{model_size}' / 'weights' / 'best.pt'
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"📁 Best model: {best_model_path}")
    print(f"{'='*60}\n")
    
    return model, best_model_path, results

# ================================
# Upload to Hugging Face
# ================================
def upload_to_huggingface(model_path, model_size='medium'):
    """Upload trained model to Hugging Face Hub"""
    config = Config()
    
    print(f"\n{'='*60}")
    print(f"📤 Uploading to Hugging Face Hub...")
    print(f"{'='*60}\n")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository
        repo_id = f"{config.HF_USERNAME}/{config.HF_REPO_NAME}"
        
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True
            )
            print(f"✓ Repository created/found: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repo creation: {e}")
        
        # Upload model weights
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=f"{config.HF_MODEL_NAME}.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        
        # Create model card
        model_card = f"""---
tags:
- ultralytics
- yolo
- yolo11
- object-detection
- instance-segmentation
- underwater
- marine-robotics
datasets:
- USIS10K
metrics:
- mAP
library_name: ultralytics
---

# YOLO11-{model_size.upper()} for Underwater Instance Segmentation

Fine-tuned YOLO11-{model_size} model on the USIS10K dataset for underwater salient instance segmentation.

## Model Details

- **Model Type:** Instance Segmentation
- **Base Model:** YOLO11-{model_size}
- **Dataset:** USIS10K (Underwater Salient Instance Segmentation)
- **Classes:** 7 underwater objects
  - wrecks/ruins
  - fish
  - reefs
  - aquatic plants
  - human divers
  - robots
  - sea-floor

## Training Details

- **Epochs:** {config.EPOCHS}
- **Batch Size:** {config.BATCH_SIZE}
- **Image Size:** {config.IMG_SIZE}
- **Optimizer:** AdamW
- **Learning Rate:** 0.001

## Usage

from ultralytics import YOLO
Load model

model = YOLO('{config.HF_MODEL_NAME}.pt')
Run inference

results = model.predict('underwater_image.jpg')

text

## Citation

If you use this model, please cite:

@software{{yolo11_usis10k,
author = {{Your Name}},
title = {{YOLO11 for Underwater Instance Segmentation}},
year = {{2025}},
url = {{https://huggingface.co/{repo_id}}}
}}

text
"""
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🔗 View at: https://huggingface.co/{repo_id}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("Make sure you've run: huggingface-cli login")

# ================================
# Main Training Pipeline
# ================================
def main():
    """Complete professional training pipeline"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  YOLO11 Professional Training Pipeline                   ║
    ║  USIS10K Underwater Instance Segmentation               ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Setup W&B
    use_wandb = setup_wandb()
    
    # Train model (choose your model size)
    model_sizes = ['medium']  # Change to ['nano', 'small', 'medium'] to train multiple
    
    for model_size in model_sizes:
        # Train
        model, best_model_path, results = train_model(
            model_size=model_size,
            use_wandb=use_wandb
        )
        
        # Upload to Hugging Face
        upload_choice = input(f"\n📤 Upload {model_size} model to Hugging Face? (y/n): ")
        if upload_choice.lower() == 'y':
            upload_to_huggingface(best_model_path, model_size)
    
    print("\n🎉 All training complete!")

if __name__ == "__main__":
    main()