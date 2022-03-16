from time import time
import pandas as pd
import cv2
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm




CFG = {
    'TRAIN_PATH': '../datasets/train_images',
    'split': .2,
    'seed': 42,
    'img_size': 128,
    'epochs': 1,
    'train_bs': 16,
    'valid_bs': 32,
    'size': 256,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda'
}


def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            RandomResizedCrop(CFG['size'], CFG['size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
            ])

    elif data == 'valid':
        return Compose([
            Resize(CFG['size'], CFG['size']),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        #print(CFG['TRAIN_PATH'])
        file_path = f"{CFG['TRAIN_PATH']}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    


def prepare_dataloader(df, split):
    train_ = df.loc[split:].reset_index(drop=True)
    valid_ = df.loc[:split].reset_index(drop=True)
        
    train_ds = TrainDataset(train_, transform=get_transforms(data='train'))
    valid_ds = TrainDataset(valid_, transform=get_transforms(data='valid'))
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader



class Network(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epoch, model, optimizer,criterion, train_loader, device):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_loader)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss/len(train_loader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_loader.dataset)    
    return train_loss, train_accuracy

        
def valid(epoch, model, criterion, val_loader, device):
    t = time.time()
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader)):
            data, target = data[0].to(device).float(), data[1].to(device).long()
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(val_loader.dataset)
        val_accuracy = 100. * val_running_correct/len(val_loader.dataset)        
        return val_loss, val_accuracy



def main():
    device = CFG['device']
    train_df = pd.read_csv('../datasets/train.csv')
    split_idx = CFG['split']*len(train_df)

    train_loader, val_loader = prepare_dataloader(train_df, split_idx)
    
    model = Network(train_df.label.nunique())
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(CFG['epochs']):

        train_loss , train_auc = train(epoch, model, optimizer, criterion, train_loader, device)

        val_loss, val_auc = valid(epoch, model, criterion, val_loader, device)
        print(f"=========Epoch: {epoch}/{CFG['epochs']}============")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_auc:.2f}")
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_auc:.2f}')
        if val_auc>validation_auc:
            print("saving the best model")
            torch.save(model.state_dict(),'{}_test'.format(CFG['model_arch']))
            validation_auc = val_auc



if __name__ == '__main__':
    main()