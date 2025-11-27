import os
from ultralytics import YOLO

# ================================
# 🛠️ USER CONFIGURATION
# ================================
# Change these two variables to switch between models/experiments
MODEL_NAME = "trainl"       # Main folder (e.g., trainl, trainm, trainm150)
EXP_NAME   = "finetune_exp1"   # Sub-folder (e.g., finetune_exp1)

# ================================
# ⚙️ AUTOMATED PATHS
# ================================
class Config:
    # Base Directories
    WORKSPACE_DIR = '/workspace/USIS10K/work_dirs/yolo/predictions/runs/segment'
    DATA_YAML = '/workspace/data/yolo/data.yaml'
    TEST_IMAGES_DIR = '/workspace/data/yolo/test/images'
    
    # Dynamic Model Path
    # Structure: .../runs/segment/{MODEL_NAME}/{EXP_NAME}/weights/best.pt
    BASE_MODEL_DIR = os.path.join(WORKSPACE_DIR, MODEL_NAME, EXP_NAME)
    MODEL_PATH = os.path.join(BASE_MODEL_DIR, 'weights', 'best.pt')
    
    # Dynamic Output Paths
    # Structure: .../runs/segment/{MODEL_NAME}/test_results/
    OUTPUT_PROJECT = os.path.join(WORKSPACE_DIR, MODEL_NAME, 'test_results')
    
    # Folder names for this specific run
    OUTPUT_NAME_PRED = 'visual_predictions'
    OUTPUT_NAME_VAL = 'metrics_evaluation'

def test_model():
    print(f"\n{'='*60}")
    print(f"🚀 Starting YOLO Testing for: {MODEL_NAME} / {EXP_NAME}")
    print(f"{'='*60}")
    print(f"📂 Model Path: {Config.MODEL_PATH}")
    print(f"📂 Output Dir: {Config.OUTPUT_PROJECT}\n")

    # 1. Validate Paths
    if not os.path.exists(Config.MODEL_PATH):
        print(f"❌ Error: Model file not found!")
        print(f"   Checked: {Config.MODEL_PATH}")
        return

    # 2. Load Model
    print(f"⏳ Loading model...")
    try:
        model = YOLO(Config.MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Run Validation (Metrics)
    print("\n📊 Calculating Metrics (mAP)...")
    metrics = model.val(
        data=Config.DATA_YAML,
        split='test',
        project=Config.OUTPUT_PROJECT,
        name=Config.OUTPUT_NAME_VAL,
        device='cuda',
        exist_ok=True
    )
    
    # ---------------------------------------------------------
    # PRINT METRICS
    # ---------------------------------------------------------
    print(f"\n{'='*40}")
    print(f"📈 RESULTS: {MODEL_NAME}")
    print(f"{'='*40}")
    print(f"   🎯 mAP (50-95):  {metrics.seg.map:.4f}")
    print(f"   🎯 mAP @ 50:     {metrics.seg.map50:.4f}")
    print(f"   🎯 mAP @ 75:     {metrics.seg.map75:.4f}")
    print(f"{'='*40}\n")

    # 4. Run Inference (Visuals)
    print("🖼️  Generating Visual Predictions...")
    results = model.predict(
        source=Config.TEST_IMAGES_DIR,
        save=True,
        project=Config.OUTPUT_PROJECT,
        name=Config.OUTPUT_NAME_PRED,
        conf=0.25,
        stream=True,  # Memory safe
        exist_ok=True
    )
    
    # Execute generator
    count = 0
    for _ in results:
        count += 1
        if count % 100 == 0:
            print(f"   Processed {count} images...")

    print(f"\n✅ Done! Results saved to:")
    print(f"   Visuals: {os.path.join(Config.OUTPUT_PROJECT, Config.OUTPUT_NAME_PRED)}")
    print(f"   Metrics: {os.path.join(Config.OUTPUT_PROJECT, Config.OUTPUT_NAME_VAL)}")

if __name__ == "__main__":
    test_model()
