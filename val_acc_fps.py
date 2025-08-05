from ultralytics import YOLO

def main():
    model = YOLO("C:/Users\dalab\Desktop/azimjaan21/RESEARCH/wrist_kpt/runs/pose/#trans_[ab2]/weights/best.pt", 
                 task="pose")  

    # Run validation (val dataset and set batch=1 for true FPS)
    metrics = model.val(data= r"C:\Users\dalab\Desktop\azimjaan21\RESEARCH\ablation_yolov8m_seg\data_wrist\data.yaml",
                        imgsz=640, 
                        batch=1,
                        visualize=True,
                        project='evaluation_results',
                        name='TransferL#ab2')  
    # Set batch=1 for single-image FPS

    # speed metrics
    print("Speed metrics (ms per image):", metrics.speed)
    fps = 1000 / sum(metrics.speed.values())
    print(f"Official FPS: {fps:.2f}")

if __name__ == "__main__":
    main()