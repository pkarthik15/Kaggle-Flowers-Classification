import torch
from model import Resnet50ImageClassification
from PIL import Image
from device import to_device, get_default_device
import torchvision.transforms as t


model_check_point = torch.load('kaggle_image_classification.pth')
classes = model_check_point["classes"]
transform = t.Compose([
    t.ToTensor()
])

model = Resnet50ImageClassification(len(classes))
model.load_state_dict(model_check_point["model_state_dict"])

device = get_default_device()
model = to_device(model, device)

image_path = '/home/karthik/myfiles/Research/DL-PyTorch/datasets/flowers-recognition/test/flowers/rose/12240303_80d87f77a3_n.jpg'
ip = Image.open(image_path)
ip = transform(ip)
op = model.predict(to_device(ip.unsqueeze(0), device))
print(op.item(), classes[op.item()])


