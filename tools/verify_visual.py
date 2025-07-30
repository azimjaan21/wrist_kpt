import cv2
from pathlib import Path

def visualize_wrist_keypoints(image_path, label_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    h, w = image.shape[:2]

    # Read label file: format class_id x_center y_center w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 11:
            print(f"Invalid label format in {label_path}")
            continue

        # Extract keypoints normalized
        kp1_x, kp1_y, kp1_v = float(parts[5]), float(parts[6]), float(parts[7])
        kp2_x, kp2_y, kp2_v = float(parts[8]), float(parts[9]), float(parts[10])

        # Draw keypoints if visible (v > 0)
        if kp1_v > 0:
            x1 = int(kp1_x * w)
            y1 = int(kp1_y * h)
            cv2.circle(image, (x1, y1), 6, (0, 0, 255), -1)  # Red dot

        if kp2_v > 0:
            x2 = int(kp2_x * w)
            y2 = int(kp2_y * h)
            cv2.circle(image, (x2, y2), 6, (255, 0, 0), -1)  # Blue dot

    cv2.imshow("Wrist Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_folder = Path(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco_wrist\images\train2017")
label_folder = Path(r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\coco_wrist\labels\train2017")

# Pick one image and its label
image_file = image_folder / "000000549532.jpg"  
label_file = label_folder / "000000549532.txt"

visualize_wrist_keypoints(image_file, label_file)
