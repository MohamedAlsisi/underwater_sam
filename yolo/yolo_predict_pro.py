import os
import argparse
from ultralytics import YOLO
from pathlib import Path

def run_inference(experiment_name, model_variant = 'finetune_exp1', conf_threshold=0.25):
    """
    Run inference for a specific YOLO experiment.
    
    Args:
        experiment_name (str): Name of the training run (e.g., 'trainl', 'trainm').
        model_variant (str): Sub-folder name (e.g., 'finetune_medium', 'finetune_exp1').
        conf_threshold (float): Confidence threshold for predictions.
    """
    
    # ================================
    # Configuration
    # ================================

    output_name = 'results'



    
    # Define base paths (Container paths)
    BASE_WORK_DIR = '/workspace/USIS10K/work_dirs/yolo'
    TEST_IMAGES_DIR = '/workspace/data/yolo/test/images'
    
    # Construct Model Path
    # Matches your requested structure: .../yolo/{experiment_name}/{model_variant}/weights/best.pt
    # Adjust 'finetune_exp1' if your training script uses a different name
    model_path = os.path.join(BASE_WORK_DIR, 'predictions', 'runs', 'segment', experiment_name, model_variant, 'weights', 'best.pt')
    
    # Construct Output Path
    # Saves to: .../yolo/predictions/runs/segment/{experiment_name}/results/
    output_project = os.path.join(BASE_WORK_DIR, 'predictions', 'runs', 'segment', experiment_name)
    
    
    print(f"\n{'='*60}")
    print(f"Starting Inference for: {experiment_name}")
    print(f"{'='*60}")
    print(f" Model Path:  {model_path}")
    print(f" Input Data:  {TEST_IMAGES_DIR}")
    print(f" Output Dir:  {os.path.join(output_project, output_name)}")
    
    # ================================
    # Validation
    # ================================
    if not os.path.exists(model_path):
        print(f"\n ERROR: Model not found at:\n{model_path}")
        print("Please check your experiment name and folder structure.")
        return

    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"\nERROR: Test images not found at:\n{TEST_IMAGES_DIR}")
        return

    # ================================
    # Inference
    # ================================
    try:
        # Load Model
        print("\n⏳ Loading model...")
        model = YOLO(model_path)
        
        # Run Prediction
        print("⏳ Running predictions...")
        results = model.predict(
            source=TEST_IMAGES_DIR,
            save=True,
            project=output_project,
            name=output_name,
            conf=conf_threshold,
            exist_ok=True,  # Overwrite existing predictions in this folder
            verbose=True
        )
        
        print(f"\nInference Complete!")
        print(f"  Results saved to: {os.path.join(output_project, output_name)}")
        
    except Exception as e:
        print(f"\n An error occurred during inference: {e}")

if __name__ == "__main__":
    # You can change these manually or use command line args
    # Example: python yolo_predict_pro.py --exp trainl --variant finetune_large
    
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    parser.add_argument('--exp', type=str, default='trainl', help='Experiment name (e.g., trainl, trainm)')
    parser.add_argument('--variant', type=str, default='finetune_exp1', help='Training variant folder name')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    run_inference(args.exp, args.variant, args.conf)
