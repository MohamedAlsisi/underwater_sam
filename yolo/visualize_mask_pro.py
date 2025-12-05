
import cv2
import numpy as np
import os

def visualize_mask(image_path, yolo_results, our_model_results_confidence, output_path):
    # 1. Check which files are missing (Robust Skipping)
    files_to_check = [
        (image_path, "Original Image"),
        (yolo_results, "YOLO Prediction"),
        (our_model_results_confidence, "USIS Prediction")
    ]
    
    missing = [name for path, name in files_to_check if not os.path.exists(path)]
    
    if missing:
        # Print warning and SKIP this image
        print(f"⚠️ Skipping {os.path.basename(image_path) if os.path.exists(image_path) else 'File'} - Missing: {', '.join(missing)}")
        return

    # Load images
    image = cv2.imread(image_path)          # Width: w
    yolo_img = cv2.imread(yolo_results)     # Width: w
    usis_img = cv2.imread(our_model_results_confidence) # Width: 2w (assumed)
    
    if image is None or yolo_img is None or usis_img is None:
        print(f"❌ Error reading images (corrupt file?) for {os.path.basename(image_path)}")
        return

    h, w, _ = image.shape
    
    # 1. Resize YOLO to match Original (w, h)
    if yolo_img.shape != image.shape:
        yolo_img = cv2.resize(yolo_img, (w, h))

    # 2. Create Top Row: [ Original | YOLO ] -> Width: 2w
    row1 = np.concatenate((image, yolo_img), axis=1)
    
    # 3. Resize USIS to match the FULL WIDTH of Row 1 (2w)
    target_width = row1.shape[1] # Should be 2*w
    target_height = h            # Keep height same as original rows
    
    if usis_img.shape[1] != target_width or usis_img.shape[0] != target_height:
        usis_img = cv2.resize(usis_img, (target_width, target_height))

    # 4. Stack Vertically
    # [ Original | YOLO ]
    # [      USIS       ]
    combined = np.concatenate((row1, usis_img), axis=0)

    # Add Titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (255, 255, 255)
    thickness = 3
    
    # Top Left: Original
    cv2.putText(combined, "Original", (50, 80), font, font_scale, color, thickness)
    
    # Top Right: YOLO
    cv2.putText(combined, "YOLO Prediction", (w + 50, 80), font, font_scale, color, thickness)
    
    # Bottom: USIS (Centered-ish title)
    cv2.putText(combined, " GT                         USIS Prediction", (50, h + 80), font, font_scale, color, thickness)

    # Save
    cv2.imwrite(output_path, combined)
    print(f"✅ Saved: {os.path.basename(output_path)}")


if __name__ == "__main__":
    # Configurations
    NUM_IMAGES = 1000
    MODEL_NAME = "trainx200"
    
    # Paths
    BASE_DATA = "/workspace/data/yolo/test"
    YOLO_PRED_DIR = f"/workspace/USIS10K/work_dirs/yolo/predictions/runs/segment/{MODEL_NAME}/test_results/visual_predictions"
    USIS_PRED_DIR = "/workspace/USIS10K/work_dirs/usis_multiclass_eval/20251120_210127/vis_data/vis_image"
    OUTPUT_DIR = "/workspace/USIS10K/work_dirs/comparison"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🔍 Generating comparisons in: {OUTPUT_DIR}")
    
    for i in range(NUM_IMAGES):
        random_id = np.random.randint(0, 1500)
        id_yolo = f"{random_id:05d}"
        id_usis = f"{random_id}"
        
        img_path = f"{BASE_DATA}/images/test_{id_yolo}.jpg"
        yolo_path = f"{YOLO_PRED_DIR}/test_{id_yolo}.jpg"
        usis_path = f"{USIS_PRED_DIR}/test_img_{id_usis}.png"
        
        output_path = f"{OUTPUT_DIR}/compare_{id_yolo}.jpg"
        
        visualize_mask(img_path, yolo_path, usis_path, output_path)
        
    print("\n🎉 Done!")



# import cv2
# import numpy as np
# import os

# def visualize_mask(image_path, mask_path, yolo_results, our_model_results_confidence, output_path):
#     # 1. Check which files are missing
#     files_to_check = [
#         (image_path, "Original Image"),
#         (mask_path, "Ground Truth Mask"),
#         (yolo_results, "YOLO Prediction"),
#         (our_model_results_confidence, "USIS Prediction")
#     ]
    
#     missing = [name for path, name in files_to_check if not os.path.exists(path)]
    
#     if missing:
#         print(f"⚠️ Skipping {os.path.basename(image_path)} - Missing: {', '.join(missing)}")
#         return

#     # Load images
#     image = cv2.imread(image_path)
#     yolo_img = cv2.imread(yolo_results)
#     img_confidence = cv2.imread(our_model_results_confidence)
    
#     if image is None or yolo_img is None or img_confidence is None:
#         print(f"❌ Error reading image files for {os.path.basename(image_path)}")
#         return

#     h, w, _ = image.shape
    
#     # Class Names
#     class_names = ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor']

#     def draw_polygons(img_src, label_path, color, alpha=0.5):
#         img_out = img_src.copy()
        
#         # If label file is empty or missing (sanity check)
#         if not os.path.exists(label_path):
#             return img_out

#         with open(label_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if not parts: continue
                
#                 class_id = int(parts[0])
#                 coords = list(map(float, parts[1:]))

#                 # Denormalize coordinates
#                 points = np.array([(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)], dtype=np.int32)

#                 # Draw polygon
#                 overlay = img_out.copy()
#                 cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
#                 cv2.fillPoly(overlay, [points], color=color)
#                 img_out = cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0)

#                 # Draw Label
#                 text = class_names[class_id] if class_id < len(class_names) else str(class_id)
#                 text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
#                 text_x = int(points[0][0])
#                 text_y = int(points[0][1])
                
#                 # Keep label inside image
#                 text_x = max(0, min(text_x, w - text_size[0]))
#                 text_y = max(text_size[1], min(text_y, h))

#                 box_coords = ((text_x, text_y - 10), (text_x + text_size[0] + 10, text_y + text_size[1] + 5))
#                 cv2.rectangle(img_out, box_coords[0], box_coords[1], (0, 0, 0), -1)
#                 cv2.putText(img_out, text, (text_x + 5, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#         return img_out

#     # Draw Ground Truth on Original
#     image_gt = draw_polygons(image, mask_path, (0, 255, 0), alpha=0.5)

#     # Resize predictions to match original (safety step)
#     if yolo_img.shape != image.shape:
#         yolo_img = cv2.resize(yolo_img, (w, h))
#     if img_confidence.shape != image.shape:
#         img_confidence = cv2.resize(img_confidence, (w, h))

#     # Create 2x2 Grid
#     row1 = np.concatenate((image, image_gt), axis=1)
#     row2 = np.concatenate((yolo_img, img_confidence), axis=1)
#     combined = np.concatenate((row1, row2), axis=0)

#     # Add Titles
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.2
#     color = (255, 255, 255)
#     thickness = 2
    
#     # Top Left: Original
#     cv2.putText(combined, "Original Image", (30, 50), font, font_scale, color, thickness)
#     # Top Right: Ground Truth
#     cv2.putText(combined, "Ground Truth", (w + 30, 50), font, font_scale, color, thickness)
#     # Bottom Left: YOLO
#     cv2.putText(combined, "YOLO Prediction", (30, h + 50), font, font_scale, color, thickness)
#     # Bottom Right: USIS
#     cv2.putText(combined, "USIS Prediction", (w + 30, h + 50), font, font_scale, color, thickness)

#     # Save Result
#     cv2.imwrite(output_path, combined)
#     print(f"✅ Saved: {os.path.basename(output_path)}")


# if __name__ == "__main__":
#     # ================================
#     # Configuration
#     # ================================
#     NUM_IMAGES = 20
#     MODEL_NAME = "trainl"  # Match this to your --exp name
    
#     # Define Paths (Container)
#     BASE_DATA = "/workspace/data/yolo/test"
    
#     # YOLO Results Path (Updated to your structure)
#     # .../predictions/runs/segment/{model_name}/results/
#     YOLO_PRED_DIR = f"/workspace/USIS10K/work_dirs/yolo/predictions/runs/segment/{MODEL_NAME}/results"
    
#     # USIS Results Path
#     USIS_PRED_DIR = "/workspace/USIS10K/work_dirs/usis_multiclass_eval/20251120_210127/vis_data/vis_image"
    
#     # Final Output Path
#     OUTPUT_DIR = "/workspace/USIS10K/work_dirs/comparisons"

#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print(f"🔍 Starting visualization for model: {MODEL_NAME}")
#     print(f"📂 Output Directory: {OUTPUT_DIR}")
    
#     # Run loop
#     # Run loop
#     for i in range(NUM_IMAGES):
#         # Randomly select image ID (0 to 1500)
#         random_id = np.random.randint(0, 1500)
        
#         # YOLO / Original format: "01000" (Zero padded, 5 digits)
#         id_yolo = f"{random_id:05d}"
        
#         # USIS format: "1000" (No padding, just the number)
#         id_usis = f"{random_id}"
        
#         # Define file paths
#         img_path = f"{BASE_DATA}/images/test_{id_yolo}.jpg"
#         mask_path = f"{BASE_DATA}/labels/test_{id_yolo}.txt"
#         yolo_path = f"{YOLO_PRED_DIR}/test_{id_yolo}.jpg"
        
#         # CORRECTED USIS PATH
#         usis_path = f"{USIS_PRED_DIR}/test_img_{id_usis}.png"
        
#         output_path = f"{OUTPUT_DIR}/compare_{MODEL_NAME}_{id_yolo}.jpg"
        
#         visualize_mask(img_path, mask_path, yolo_path, usis_path, output_path)

        
#     print(f"\n🎉 Done! Check {OUTPUT_DIR} for results.")
