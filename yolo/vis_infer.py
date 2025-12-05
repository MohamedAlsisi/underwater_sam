# This script is used for inference and visualize the predicted results
from mmdet.apis import DetInferencer
import os
from glob import glob

def vis_infer(
   
    checkpoints="/home/mundus/ymorsy172/USIS10K/work_dirs/USIS10KDataset/huge/best_coco_segm_mAP_epoch_21.pth",
    config='/home/osim-mir/student/mo/sam/USIS10K/project/our/configs/multiclass_usis_train.py',
    data_dir='/home/osim-mir/student/mo/sam/data/USIS10K',
    output_dir='/home/osim-mir/student/mo/sam/USIS10K/work_dirs/yolo/predictions',
    device='cuda:0',
    threshold = 0.85

):
    """
    Function to run the DetInferencer for visual inference with default parameters.

    Args:
    checkpoints (str): Path to the model checkpoint (default: "./pretrain/multi_class_model_with_classes.pth").
    config (str): Path to the configuration file (default: './project/our/configs/multiclass_usis_train.py').
    data_dir (str): Path to the directory containing the test data (default: './data/USIS10K/test/').
    output_dir (str): Path to the output directory where results will be saved (default: './USIS10K/data/vis/test').
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the DetInferencer
    inferencer = DetInferencer(model=config, weights=checkpoints,device=device)
    
        # Handle different input types
    if os.path.isfile(data_dir):
        # Single image
        results = inferencer(data_dir, pred_score_thr=threshold, out_dir=output_dir)
    elif os.path.isdir(data_dir):
        # Directory of images
        image_files = glob(os.path.join(data_dir, '*.[jp][pn][gf]'))  # Match jpg, jpeg, png
        for image_file in image_files:
            results = inferencer(image_file, pred_score_thr=threshold, out_dir=output_dir)
    else:
        raise ValueError(f"Input path {data_dir} is neither a file nor directory")

    print(f"Results saved to {output_dir}")
    return results

    # Perform inference and save the output
    # inferencer(data_dir, out_dir=output_dir)


if __name__ == "__main__":
    vis_infer()


