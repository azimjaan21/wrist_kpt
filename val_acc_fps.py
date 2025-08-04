from ultralytics import YOLO

def main():
    model = YOLO(r"trans_learn_base_pose.engine", 
                 task="pose")  

    # Run validation (val dataset and set batch=1 for true FPS)
    metrics = model.val(data= "coco_wrist/wrist_data.yaml",
                        imgsz=640, 
                        batch=1,
                        visualize=True,
                        project='evaluation_results',
                        name='###trans_learn_base')  
    # Set batch=1 for single-image FPS

    # speed metrics
    print("Speed metrics (ms per image):", metrics.speed)
    fps = 1000 / sum(metrics.speed.values())
    print(f"Official FPS: {fps:.2f}")

if __name__ == "__main__":
    main()