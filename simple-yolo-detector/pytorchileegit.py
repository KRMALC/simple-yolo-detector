import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# 1. YOLO Veri Seti Sınıfı
class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        label_path = os.path.join(self.label_dir, self.image_files[index].replace('.jpg', '.txt').replace('.png', '.txt'))

        # Görüntüyü yükle
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Etiketleri yükle
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                data = list(map(float, line.strip().split()))
                class_id, bbox = int(data[0]), data[1:]
                boxes.append([class_id] + bbox)

        # Boxları PyTorch Tensor olarak döndür
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes

# 2. Model Tanımı
class SimpleYoloModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleYoloModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 160 * 160, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes + 4)  # Sınıf + bbox (x, y, w, h)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

# 3. collate_fn Tanımı
def collate_fn(batch):
    images = []
    boxes = []

    for img, bbox in batch:
        images.append(img)
        boxes.append(bbox)

    # Görüntüleri torch tensor'larına çevir
    images = torch.stack(images, 0)

    # Boxları birleştirerek aynı boyutlu bir tensör haline getir
    max_boxes = max([box.shape[0] for box in boxes])  # En büyük box sayısını bul
    padded_boxes = []
    for box in boxes:
        diff = max_boxes - box.shape[0]
        if diff > 0:
            padding = torch.zeros(diff, box.shape[1])  # Sıfırlarla doldur
            padded_boxes.append(torch.cat([box, padding], dim=0))
        else:
            padded_boxes.append(box)

    boxes = torch.stack(padded_boxes, 0)
    return images, boxes

# 4. Veri Seti ve DataLoader Tanımı
train_image_dir = "C:/Your_PATH_/Tomato Detection.v1i.yolov8/train/images"
train_label_dir = "C:/Your_PATH_/Tomato Detection.v1i.yolov8/train/labels"

train_dataset = YoloDataset(image_dir=train_image_dir, label_dir=train_label_dir)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 5. Model, Loss ve Optimizer Tanımı
num_classes = 1  # Domates sınıfı
model = SimpleYoloModel(num_classes=num_classes)

criterion_class = nn.CrossEntropyLoss()  # Sınıf kaybı için
criterion_bbox = nn.MSELoss()  # BBox kaybı için
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Eğitim Döngüsü
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)

        # Çıktıyı böl ve yeniden şekillendir
        pred_classes = outputs[:, :num_classes]  # Sınıf tahminleri (logits)
        pred_bboxes = outputs[:, num_classes:]  # BBox tahminleri

        # Hedefleri böl ve yeniden düzenle
        target_classes = targets[:, :, 0].reshape(-1).long()  # Sınıf hedefleri (1D vektör)
        target_bboxes = targets[:, :, 1:].reshape(-1, 4)      # Bounding box hedefleri (2D)

        # Tahminleri yeniden şekillendir
        pred_classes = pred_classes.view(-1, num_classes)  # Sınıf tahminleri [total_boxes, num_classes]
        pred_bboxes = pred_bboxes.view(-1, 4)              # Bounding box tahminleri [total_boxes, 4]

        # Loss hesapla
        class_loss = criterion_class(pred_classes, target_classes)  # CrossEntropyLoss, logits ile çalışır
        bbox_loss = criterion_bbox(pred_bboxes, target_bboxes)      # MSELoss, bbox'lar için
        loss = class_loss + bbox_loss

        # Geri yayılım ve optimizasyon
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# 7. Model Kaydetme
torch.save(model.state_dict(), "yolo_custom_model.pth")
print("Model eğitimi tamamlandı ve kaydedildi.")
