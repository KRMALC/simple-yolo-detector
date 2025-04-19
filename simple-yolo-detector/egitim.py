from ultralytics import YOLO

def main():
    # YOLOv8 modelini yükle
    model = YOLO("yolov8n.pt")  # YOLOv8 Nano önceden eğitilmiş model

    # Modeli eğit
    model.train(
        data="C:/your_PATH_detectıondomates/dataset.yaml",  # YAML dosyasının yolu
        epochs=50,                             # Eğitim epoch sayısı
        imgsz=640,                             # Görüntü boyutu
        batch=16,                              # Batch boyutu
        device=0                               # GPU ID (0) veya -1 (CPU)
    )

if __name__ == "__main__":
    main()
