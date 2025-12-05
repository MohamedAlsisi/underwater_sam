# USIS10K Underwater Image Segmentation

This project implements underwater salient instance segmentation using MMDetection and YOLO models with Docker containerization for reproducible training and inference.

## 🚀 Quick Start

### Prerequisites
- Docker installed
- NVIDIA GPU with drivers
- nvidia-container-toolkit installed
- CUDA 12.1 compatible GPU

### Verify GPU Access

```bash
# On host machine
nvidia-smi

# Test Docker GPU access
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 📦 Docker Setup

### Build the Image

```bash
docker build -t usis10k-sam .
```

### Run Container (Basic)

```bash
# Simple container without volume mounting
docker run --gpus all --rm -it usis10k-sam bash
```

### Run Container (With Volume Mounting - Recommended)

```bash
# Mount project directory for persistent data
docker run --gpus all --rm -it \
  -v /home/osim-mir/student/mo/sam/USIS10K:/workspace \
  usis10k-sam bash
```

**Explanation:**
- `--gpus all`: Enables GPU access inside container
- `--rm`: Automatically removes container when it exits
- `-it`: Interactive terminal mode
- `-v <host_path>:<container_path>`: Mounts host directory to container (changes persist on host)

### Docker Compose (Alternative Method)

#### Start Services

```bash
docker-compose up -d  # Start in detached mode
```

#### Stop Services

```bash
docker-compose down  # Stop and remove containers
```

#### Enter Running Container

```bash
docker exec -it yolo_training bash
```

#### Configuration Changes

**For configuration changes only (no rebuild needed):**

```bash
# Stop containers
docker-compose down

# Start with new configuration
docker-compose up -d
```

**For image/Dockerfile changes (rebuild required):**

```bash
# Rebuild and restart all services
docker-compose up -d --build

# Or rebuild specific service only
docker-compose up -d --build yolo-training
```

## 🔬 MMDetection Workflow

### 1. Run Testing/Inference

```bash
python tools/test.py \
  project/our/configs/multiclass_usis_train.py \
  checkpoints/multi_class_model.pth \
  --work-dir work_dirs/usis_multiclass_eval \
  --out work_dirs/usis_multiclass_eval/results.pkl
```

**Arguments:**
- `config`: Path to model configuration file
- `checkpoint`: Path to trained model weights (.pth file)
- `--work-dir`: Directory to save evaluation results and visualizations
- `--out`: Path to save predictions as pickle file (required for confusion matrix)

### 2. Generate Confusion Matrix

```bash
python tools/analysis_tools/confusion_matrix.py \
  project/our/configs/multiclass_usis_train.py \
  work_dirs/usis_multiclass_eval/results.pkl \
  work_dirs/confusion_matrix \
  --show \
  --score-thr 0.3 \
  --tp-iou-thr 0.5 \
  --color-theme plasma
```

**Arguments:**
- First argument: Config file used for testing
- Second argument: Path to results.pkl file generated from testing
- Third argument: Output directory for confusion matrix image
- `--show`: Display the plot (optional)
- `--score-thr`: Score threshold to filter detections (default: 0.3)
- `--tp-iou-thr`: IoU threshold for true positive matching (default: 0.5)
- `--color-theme`: Color scheme for heatmap (default: plasma)

**Output:** `confusion_matrix.png` saved in specified directory

## 🎯 YOLO Workflow

### 1. Convert COCO Dataset to YOLO Format

```bash
python coco_yolo.py \
  --data_dir /workspace/data/USIS10K \
  --output_dir /workspace/data/yolo_dir
```

**What it does:**
- Converts COCO annotations to YOLO format
- Creates train/val/test splits
- Generates YOLO-compatible directory structure

### 2. Train YOLO Model

```bash
python yolo_train.py
```

**Training outputs:**
- Model checkpoints saved in `/workspace/USIS10K/work_dirs/yolo/finetune_exp1/weights/`
- Best model: `best.pt`
- Last checkpoint: `last.pt`

### 3. Generate Predictions

```bash
python -c "
from ultralytics import YOLO
model = YOLO('/workspace/USIS10K/work_dirs/yolo/finetune_exp1/weights/best.pt')
model.predict(
    source='/workspace/data/yolo_dir/test/images/',
    save=True,
    project='/workspace/USIS10K/work_dirs/yolo',
    name='predictions'
)
"
```

**Prediction outputs:**
- Annotated images saved in `/workspace/USIS10K/work_dirs/yolo/predictions/`
- Bounding boxes and class labels visualized on images

## 📁 Project Structure

```
USIS10K/
├── checkpoints/                    # Model weights
│   └── multi_class_model.pth
├── configs/                        # MMDetection configs
├── data/                           # Dataset directory
│   ├── USIS10K/                   # Original COCO format
│   └── yolo_dir/                  # Converted YOLO format
├── project/our/                    # Custom project files
│   └── configs/
│       └── multiclass_usis_train.py
├── tools/                          # Training/testing scripts
│   ├── test.py
│   ├── train.py
│   └── analysis_tools/
│       └── confusion_matrix.py
├── work_dirs/                      # Experiment outputs
│   ├── usis_multiclass_eval/
│   │   ├── results.pkl
│   │   └── vis_data/
│   ├── confusion_matrix/
│   └── yolo/
├── coco_yolo.py                    # Dataset conversion
├── yolo_train.py                   # YOLO training script
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker compose config
└── requirements.txt                # Python dependencies
```

## 🛠️ Troubleshooting

### GPU Not Detected

```bash
# Check if nvidia-container-toolkit is installed
docker run --gpus all --rm nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### NumPy Compatibility Error
If you see NumPy 2.x compatibility errors:

```bash
# Edit requirements.txt
numpy<2.0

# Rebuild image
docker build -t usis10k-sam .
```

### Missing results.pkl
The confusion matrix script requires a `.pkl` file. Make sure to run test.py with `--out` flag:

```bash
python tools/test.py ... --out results.pkl
```

## 📊 Key Files Explained

### `.pkl` Files
- **Format:** Python pickle binary format
- **Purpose:** Serializes Python objects (predictions) to disk
- **Contains:** Detection results (bboxes, labels, scores) for all test images
- **Usage:** Loaded by analysis tools without re-running inference

### Config Files
- **MMDetection:** `.py` files in `configs/` and `project/our/configs/`
- **YOLO:** YAML format, auto-generated during conversion

### Checkpoints
- **MMDetection:** `.pth` files (PyTorch weights)
- **YOLO:** `.pt` files (Ultralytics format)

## 📝 Notes

- Always use `--gpus all` when running containers that need GPU access
- Volume mounting (`-v`) is recommended for development to persist changes
- Results are saved in `work_dirs/` by default
- Confusion matrix requires running test.py with `--out` flag first

## 🔗 Dependencies

- PyTorch 2.1.2 + CUDA 12.1
- MMDetection 3.3.0
- MMCV 2.1.0
- Ultralytics YOLO
- NumPy < 2.0 (for PyTorch 2.1 compatibility)

## 📧 Contact

Mohamed Alsisi - [mhmd.alsisi@gmail.com](mailto:mhmd.alsisi@gmail.com)