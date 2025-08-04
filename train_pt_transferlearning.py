import torch
from ultralytics import YOLO

def train_pose_from_pt(
    model_pt,
    data_yaml,
    exp_name,
    kpt_shape=[2, 3],
    epochs=120,
    batch=16,
    imgsz=640,
    patience=30,
    workers=5,
    device=None
):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on {device}, resuming from checkpoint: {model_pt}")

    # Load model from checkpoint (.pt file)
    model = YOLO(model_pt)
    
    # (Optional) Override kpt_shape if you're continuing with different keypoints
    if kpt_shape:
        model.model.kpt_shape = kpt_shape

    print(f"[INFO] Experiment: {exp_name}")
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
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        close_mosaic=10,
        warmup_epochs=3,
        lrf=0.01,
        cos_lr=True,
        pretrained=True,   # ✅ continue from weights
        save=True
    )

if __name__ == "__main__":
    config = {
        'model_pt': r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\wrist_kpt\runs\pose\ab1\weights\best.pt',
        'data_yaml': r'C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml',
        'exp_name': '#trans_[ab1]',
        'kpt_shape': [2, 3],
        'epochs': 50,
        'batch': 16,
        'imgsz': 640,
        'patience': 15,
        'workers': 5,
    }
    train_pose_from_pt(**config)
