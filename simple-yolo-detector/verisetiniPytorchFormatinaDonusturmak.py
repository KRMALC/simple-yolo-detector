import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_image_dir = "C:/PATH/Tomato Detection.v1i.yolov8/train/images"
train_label_dir = "C:/PATH/Tomato Detection.v1i.yolov8/train/labels"

class YoloDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform if transform else transforms.ToTensor()

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
