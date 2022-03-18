import pandas as pd
import cv2
import torch
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
    'test_split': .2,
    'val_split': .1,
    'seed': 42,
    'epochs': 1,
    'train_bs': 16,
    'valid_bs': 32,
    'size': 128,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2,
    'verbose_step': 1,
    'device': 'cuda',
    'model_arch':'base_ckpt'
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
    


def prepare_dataloader(df, test_split, val_split):
    test_split = CFG['test_split']*len(df)
    train = df.loc[test_split:,].reset_index(drop=True)
    test_ = df.loc[:test_split,].reset_index(drop=True)
    val_split = CFG['val_split']*len(train)
    train_ = train.loc[val_split:,].reset_index(drop=True)
    valid_ = train.loc[:val_split,].reset_index(drop=True)
    
    
    train_ds = TrainDataset(train_, transform=get_transforms(data='train'))
    valid_ds = TrainDataset(valid_, transform=get_transforms(data='valid'))
    test_ds = TrainDataset(test_, transform=get_transforms(data='valid'))
    
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
    tst_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader, tst_loader


class Network(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # 3 RGB channels, applying kernel size 3, output image size 126, number of output channels 32 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3)
        # 3 RGB channels, applying Max Pooling size (2,2) with stride 2, output image size 62, output channels 32 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 3 RGB channels, applying kernel size 3 with stride 1, output image size 60, output channels 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16*30*30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_class)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, optimizer,criterion, train_loader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for data in tqdm(train_loader):
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

        
def valid(model, criterion, val_loader, device):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            data, target = data[0].to(device).float(), data[1].to(device).long()
            outputs = model(data)
            loss = criterion(outputs, target)
            
            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == target).sum().item()
        
        val_loss = val_running_loss/len(val_loader.dataset)
        val_accuracy = 100. * val_running_correct/len(val_loader.dataset)        
        return model, val_loss, val_accuracy




def main():
    device = CFG['device']
    df = pd.read_csv('../datasets/train.csv')

    train_loader, val_loader, test_loader = prepare_dataloader(df, CFG['test_split'], CFG['val_split'])
    
    model = Network(df.label.nunique())
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    
    validation_auc = 0
    for epoch in range(CFG['epochs']):

        train_loss , train_auc = train(model, optimizer, criterion, train_loader, device)

        model , val_loss, val_auc = valid(model, criterion, val_loader, device)
        print(f"=========Epoch: {epoch+1}/{CFG['epochs']}============")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_auc:.2f}")
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_auc:.2f}')
            
            
            
    _, test_loss, test_auc = valid(model, criterion, test_loader, device)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_auc:.2f}')


if __name__ == '__main__':
    main()