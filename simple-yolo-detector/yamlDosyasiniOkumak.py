import yaml

with open("C:Your_PATH_/detectıondomates/dataset.yaml","r") as file:
    data_config=yaml.safe_load(file)

train_path = data_config["train"]
val_path = data_config["val"]
num_classes = data_config["nc"]
class_names = data_config["names"]

print(f"Train Path: {train_path}")
print(f"Val Path: {val_path}")
print(f"Number of Classes: {num_classes}")
print(f"Class Names: {class_names}")


#dosya degistirilme tarihi
import os
import time

file_path = "yolo_custom_model.pth"
if os.path.exists(file_path):
    modification_time = os.path.getmtime(file_path)
    print(f"{file_path} değiştirilme tarihi: {time.ctime(modification_time)}")
else:
    print(f"{file_path} dosyası bulunamadı.")
