# Basit YOLO Formatlı Nesne Tespit Modeli

Bu proje, YOLO formatında etiketlenmiş bir veri kümesi (Roboflow'dan) kullanılarak PyTorch ile sıfırdan yazılmış basit bir CNN modeliyle nesne tespiti yapmayı amaçlamaktadır.
Projede örnek olarak domates tespiti özelinde çalışılmıştır.
Bir kaç katmandan oluştuğu için başarılı bir model değildir.
İndirip denemenize gerek yoktur. Herhangi bir tespit yapmayacaktır ya da çok az tespit yapacaktır.

## Veri Kümesi

- Roboflow üzerinden elde edilen YOLOv8 formatında bir veri seti kullanılmıştır.
- Veri kümesi `train/`, `valid/` ve `test/` klasörlerine ayrılmıştır.
- Etiketler `.txt` dosyalarında, YOLO formatında (class x_center y_center width height) tutulmaktadır.

## Model Yapısı

- Sıfırdan yazılmış basit bir CNN (Convolutional Neural Network)
- Hem sınıf tahmini (class) hem de bounding box (x, y, w, h) tahmini yapmaktadır.
- YOLO mimarisi kadar gelişmiş olmasa da temel çalışma mantığını öğrenmek için uygundur.

## Eğitim Aşaması

Eğitim `pytorchileegit.py` dosyası ile yapılmaktadır. Kullanılan yapılar:
- `CrossEntropyLoss` → Sınıf tahmini kaybı
- `MSELoss` → BBox tahmini kaybı
- `collate_fn` → Farklı sayıda bounding box içeren örnekler için padding işlemi


