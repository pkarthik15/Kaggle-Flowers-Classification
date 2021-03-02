import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as t
from model import Resnet50ImageClassification
from device import to_device, get_default_device, DeviceDataLoader
from tqdm import tqdm


def fit(model, epochs, lr, train_loader, valid_loader, opt=optim.Adam):
    history = []
    optimizer = opt(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        train_history = []
        for batch in tqdm(train_loader):
            loss, result = model.training_step(batch)
            train_history.append(result)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_result = model.epoch_end(train_history)
        valid_result = model.evaluate(valid_loader)
        epoch_result = {
            'train_loss': train_result['loss'],
            'train_acc': train_result['acc'],
            'val_loss': valid_result['loss'],
            'val_acc': valid_result['acc']
        }
        model.epoch_end_log(epoch, epoch_result)
        history.append(epoch_result)
    return history


data_dir = '/home/karthik/myfiles/Research/DL-PyTorch/datasets/flowers-recognition/flowers/'
transform = t.Compose([
    t.Resize(224),
    t.RandomCrop(224),
    t.ToTensor()
])

image_ds = ImageFolder(data_dir, transform=transform)
val_pct = 0.1
val_size = int(val_pct * len(image_ds))
train_size = len(image_ds) - val_size


train_ds, valid_ds = random_split(image_ds, [train_size, val_size])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=64)

device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


model = Resnet50ImageClassification(len(image_ds.classes))
epoch_history = []
model = to_device(model, device)


epoch_history += fit(model, 5, 1e-4, train_dl, valid_dl, opt=optim.Adam)

model_check_point = {
    "model_state_dict": model.state_dict(),
    "classes": image_ds.classes
}

torch.save(model_check_point, "kaggle_image_classification.pth")












