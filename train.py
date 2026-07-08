from ultralytics import YOLO
from multiprocessing import freeze_support
import torch
def main():
    print("=" * 50)
    print("YOLOv8 Fall / Sit / Walk Training Start")
    print("=" * 50)
    print("CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\Users\Lenovo\Desktop\human fall.v2i.yolov8\data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        cache=False,
        name="fall_sit_walk_final",
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.1,
        degrees=10,
        translate=0.15,
        scale=0.5,
        shear=5,
        perspective=0.0005,
        amp=True,
        patience=50,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.0005,
        box=7.5,
        cls=1.0,
        dfl=2.0,
        save=True,
        save_period=10,
        plots=True,
        verbose=True
    )
    print("训练完成")
if __name__ == "__main__":
    freeze_support()
    main()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              