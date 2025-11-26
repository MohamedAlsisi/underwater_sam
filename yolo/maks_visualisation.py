import cv2
import numpy as np

def visualize_mask(image_path, mask_path,yolo_imgs, our_model_results_confidence):
    # Load the image
    image = cv2.imread(image_path)
    yolo_img = cv2.imread(yolo_results)
    img_confidence = cv2.imread(our_model_results_confidence)
    h, w, _ = image.shape
    # Define class names
    class_names = ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor']

    def draw_polygons(image, label_path, color, alpha=0.5):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])  # Class ID
                print(class_names[class_id])
                coords = list(map(float, parts[1:]))

                # Denormalize coordinates
                points = np.array([(coords[i] * w, coords[i + 1] * h) for i in range(0, len(coords), 2)], dtype=np.int32)

                # Draw the polygon on the image
                overlay = image.copy()
                cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)
                cv2.fillPoly(overlay, [points], color=color)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # Draw a black box behind the class name
                text = class_names[class_id]  # Get the class name
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]  # Get text size
                text_x = int(points[0][0])  # X-coordinate for text
                text_y = int(points[0][1])  # Y-coordinate for text
                box_coords = ((text_x, text_y - 10), (text_x + text_size[0] + 10, text_y + text_size[1] + 5))
                cv2.rectangle(image, box_coords[0], box_coords[1], (0, 0, 0), -1)  # Black box

                # Write the class name in white on top of the black box
                cv2.putText(image, text, (text_x + 5, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    # Draw polygons for mask and YOLO labels
    image_gt = draw_polygons(image, mask_path, (0, 255, 0, 50), alpha=0.5)  # Green mask with lower opacity
    row1 = np.concatenate((image, image_gt),axis=1)
    row2 = np.concatenate((yolo_img,img_confidence),axis=1)
    compined = np.concatenate((row1, row2),axis=0)


# Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2

    # Label dimensions and positions
    height, width, _ = row1.shape
    cv2.putText(compined, "Image", (50, 50), font, font_scale, color, thickness)
    cv2.putText(compined, "Ground Truth", (width // 2 + 50, 50), font, font_scale, color, thickness)
    cv2.putText(compined, "YOLO", (50, height + 50), font, font_scale, color, thickness)
    cv2.putText(compined, "USIS", (width // 2 + 50, height + 50), font, font_scale, color, thickness)
    cv2.imshow("Image", compined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
for i in range(20):
    img_random = np.random.randint(0, 1500)
    img_random = f"{img_random:05d}"
    # img_random = "00114"
    print(img_random)
    img = f"/home/osim-mir/student/mo/sam/data/yolo_dir/test/images/test_{img_random}.jpg"
    masked = f"/home/osim-mir/student/mo/sam/data/yolo_dir/test/labels/test_{img_random}.txt"
    # YOLO predictions - keep organized under yolo subdirectory
    yolo_results = f"/home/osim-mir/student/mo/sam/USIS10K/work_dirs/yolo/predictions/test_{img_random}.jpg"
    # USIS predictions - separate subdirectory for clarity
    our_model_results_confidence = f"/home/osim-mir/student/mo/sam/USIS10K/work_dirs/usis/predictions/test_{img_random}.jpg"
    visualize_mask(img, masked, yolo_results, our_model_results_confidence)