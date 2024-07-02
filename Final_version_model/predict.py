import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model = models.mobilenet_v3_large(weights=None)

# Thay thế lớp cuối cùng của mô hình để phù hợp với số lớp đầu ra mới
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("MobileV3.pth"))
model = model.to('cuda')
model.eval()

def load_and_preprocess_image(image_path):
    img = Image.open(image_path) # Mở ảnh từ đường dẫn

    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Thay đổi kích thước ảnh thành 224x224
        transforms.ToTensor(), # Chuyển đổi ảnh thành tensor
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = transform(img)
    img = img.unsqueeze(0) #thêm một chiều batch

    return img

class_names = ['non_nodule', 'nodule']

def process():
    image_path = input()
    input_image = load_and_preprocess_image(image_path)

    with torch.no_grad():
        output = model(input_image.cuda())

    _, predicted = output.max(dim=1)
    print(class_names[predicted])