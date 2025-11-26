# import os
# from ultralytics import YOLO
# import torch
# import yaml

# # ================================
# # Paths Configuration
# current_dir = os.getcwd()
# # ================================
# # Update these paths with your dataset and configuration
# DATASET_PATH = os.path.join(current_dir, 'dataset', 'data.yaml')
# PRETRAINED_WEIGHTS = os.path.join(current_dir, 'yolo11x-seg.pt') #you can change this to the model available , but i hope you can run this on the xl , just check how much you need
# OUTPUT_DIR = os.path.join(current_dir, 'runs', 'segment', 'train')

# # ================================
# # Training Parameters
# # ================================
# EPOCHS = 50  # Number of epochs to train change this if needed but the minimum should be 30
# BATCH_SIZE = 16  # Adjust based on GPU memory
# IMG_SIZE = 640  # Input image size (adjust based on your dataset)
# DEVICE = "cuda" 

# # ================================
# # Train YOLOv11 Segmentation Model
# # ================================
# def train_yolo():
#     print("Starting YOLOv11 segmentation training...")
    
#     # Initialize model
#     model = YOLO(PRETRAINED_WEIGHTS)  # Load YOLOv11 segmentation pretrained weights

#     # Train the model
#     model.train(
#         data=DATASET_PATH,         # Path to dataset.yaml
#         epochs=EPOCHS,             # Number of epochs
#         batch=BATCH_SIZE,          # Batch size
#         imgsz=IMG_SIZE,            # Image size
#         device=DEVICE,             # Device: 'cuda' or 'cpu'
#         workers=8,                 # Number of workers for data loading
#         project=OUTPUT_DIR,        # Where to save training results
#         name="exp1",              # Experiment name
#         exist_ok=True              # Overwrite if directory exists
#     )

#     print("Training complete! Results saved in:", OUTPUT_DIR)

# # ================================
# # Run the Training Script
# # ================================
# if __name__ == "__main__":
#     train_yolo()



import os
from ultralytics import YOLO
import torch
import yaml

# ================================
# Paths Configuration
# ================================
current_dir = os.getcwd()
# DATASET_PATH = '/home/osim-mir/student/mo/sam/data/yolo/data.yaml'

DATASET_PATH = '/workspace/data/yolo/data.yaml' 
PRETRAINED_WEIGHTS = 'yolo11x-seg.pt'  # ← CHANGED: simplified
OUTPUT_DIR1 = '/workspace/USIS10K/work_dirs/yolo/predictions'  # ← CHANGED: absolute path
OUTPUT_DIR = os.path.join(OUTPUT_DIR1, 'runs', 'segment', 'train')


# ================================
# Fine-Tuning Parameters
# ================================
EPOCHS = 50  # ← CHANGED: 50 → 100
BATCH_SIZE = 8  # ← CHANGED: 16 → 8 (for xlarge model)
IMG_SIZE = 640
DEVICE = "cuda" 

# ================================
# Fine-Tune YOLOv11 Segmentation Model
# ================================
def finetune_yolo():  # ← CHANGED: renamed from train_yolo
    print("Starting YOLOv11 segmentation fine-tuning...")
    
    # Initialize model with pretrained weights
    model = YOLO(PRETRAINED_WEIGHTS)

    # Fine-tune the model
    model.train(
        data=DATASET_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=8,
        
        # ← NEW: Fine-tuning optimization
        optimizer='AdamW',
        lr0=0.001,              # Lower learning rate
        lrf=0.01,
        warmup_epochs=3,
        patience=15,            # Early stopping
        save_period=10,         # Save every 10 epochs
        
        # ← NEW: Underwater-specific augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,             # No vertical flip
        fliplr=0.5,
        mosaic=1.0,
        
        # Output settings
        project=OUTPUT_DIR,
        name='finetune_exp1',   # ← CHANGED: from "exp1"
        exist_ok=True,
        val=True,               # ← NEW: enable validation
        plots=True              # ← NEW: generate plots
    )

    print(f"Fine-tuning complete! Best model: {OUTPUT_DIR}/finetune_exp1/weights/best.pt")

# ================================
# Run the Fine-Tuning Script
# ================================
if __name__ == "__main__":
    finetune_yolo()  # ← CHANGED: function name
