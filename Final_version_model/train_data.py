import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import time
import os
from tempfile import TemporaryDirectory
import numpy as np

cudnn.benchmark = True

train_folder = './data_split/train'
epochs = 30

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAdjustSharpness(2, p=0.2),
        transforms.RandomAutocontrast(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'val': transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}

dataset = datasets.ImageFolder(train_folder)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])

train_dataset = CustomDataset(train_set, transform=data_transforms['train'])
val_dataset = CustomDataset(val_set, transform=data_transforms['val'])
image_datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = dataset.classes
num_classes = len(class_names)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = 100.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    print(inputs.size())
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_loss = epoch_loss
                        torch.save(model.state_dict(), best_model_params_path)
                    elif epoch_acc == best_acc and epoch_loss < best_loss:
                        best_loss = epoch_loss
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        print(f'Best val Loss: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

# Create EfficientNet model
model = models.mobilenet_v3_large(weights=None)

# Replace last layer to fit with output size
# model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, num_classes)

model = model.to('cuda')

# Define loss-function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=True)

model_conv = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)

torch.save(model_conv.state_dict(), "MobileV3.pth")