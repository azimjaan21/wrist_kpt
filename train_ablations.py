import torch
from ultralytics import YOLO

def train_pose_model(
    model_path_or_yaml,
    data_yaml,
    exp_name,
    kpt_shape=[2, 3],
    pretrained=True,
    freeze_backbone=False,
    epochs=100,
    batch=16,
    imgsz=640,
    patience=15,
    workers=4,
    device=None
):
    # Set device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on device: {device}")

    # Load model (pretrained .pt or custom .yaml)
    print(f"[INFO] Loading model: {model_path_or_yaml}")
    model = YOLO(model_path_or_yaml)

    # Set keypoint shape (num_kpts, dimensions)
    print(f"[INFO] Setting keypoint shape to: {kpt_shape}")
    model.model.kpt_shape = kpt_shape

    # Optionally freeze backbone
    if freeze_backbone:
        print("[INFO] Freezing all layers except pose head...")
        for name, param in model.model.named_parameters():
            if 'pose' not in name:
                param.requires_grad = False

    # Start training
    print("[INFO] Starting training...")
    model.train(
        data=data_yaml,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        patience=patience,
        workers=workers,
        device=device,
        project='runs/pose',
        name=exp_name,
        visualize=True,
        task='pose'
    )


if __name__ == "__main__":
    # === EXPERIMENT CONFIGURATION === #
    config = {
        'model_path_or_yaml': 'yolov8s-pose.pt',  # Or your custom YAML like 'yolo11s_pose.yaml'
        'data_yaml': r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml',
        'exp_name': 'abl0_baseline',
        'kpt_shape': [2, 3],  # Wrist-only (left & right, x/y/conf)
        'pretrained': True,
        'freeze_backbone': False,  # True to train head only
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 15,
        'workers': 4
    }

    train_pose_model(**config)
