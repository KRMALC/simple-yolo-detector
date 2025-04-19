import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/detect/train5/weights/best.pt")  # Eğitimde en iyi sonucu veren model

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    # Model ile tahmin yap
    results = model.predict(source=frame, show=False, conf=0.5)

    # Sonuçları görüntü üzerine çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("YOLOv8 Canlı Tespit", annotated_frame)

    # Çıkış için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereyi kapat
cap.release()
cv2.destroyAllWindows()
