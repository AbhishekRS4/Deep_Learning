import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import split_dataset, get_dataloaders_for_training

import argparse
from tqdm import tqdm



class Network(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # 3 RGB channels, applying kernel size 3, output image size 126, number of output channels 32 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3)
        # 3 RGB channels, applying Max Pooling size (2,2) with stride 2, output image size 62, output channels 32 
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # 3 RGB channels, applying kernel size 3 with stride 1, output image size 60, output channels 16
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(16*78*78, 128)
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
    learning_rate = 1e-4
    weight_decay = 1e-6
    batch_size = 64
    num_epochs = 100
    image_size = 320
    file_logger_train = "train_metrics.csv"
    dir_images = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/images/"
    file_csv = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/image_labels.csv"
    pretrained = 1

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate to use for training")
    parser.add_argument("--weight_decay", default=weight_decay,
        type=float, help="weight decay to use for training")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="batch size to use for training")
    parser.add_argument("--pretrained", default=pretrained,
        type=int, choices=[1, 0], help="weight initialization - 1 [pretrained] or 0 [random]")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="num epochs to train the model")
    parser.add_argument("--image_size", default=image_size,
        type=int, help="image size used to train the model")
    parser.add_argument("--file_logger_train", default=file_logger_train,
        type=str, help="file name of the logger csv file with train losses")
    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full directory path to dataset containing images")
    parser.add_argument("--file_csv", default=file_csv,
        type=str, help="full path to csv file with image ids and labels")

    FLAGS, unparsed = parser.parse_known_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    
    train_x, valid_x, train_y, valid_y, num_classes = split_dataset(FLAGS.file_csv, is_for_train=True)
    num_train_samples = len(train_x)
    num_valid_samples = len(valid_x)
    #print(valid_x[0:5])
    #print(valid_y[0:5])
    train_loader, val_loader = get_dataloaders_for_training(
        train_x, train_y, valid_x, valid_y,
        dir_images=FLAGS.dir_images, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
    )

    print(f"num train samples: {num_train_samples}, num validation samples: {num_valid_samples}")
    print(f"num classes: {num_classes}")

    model = Network(num_classes) 
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(FLAGS.num_epochs):

        train_loss , train_auc = train(model, optimizer, criterion, train_loader, device)

        model , val_loss, val_auc = valid(model, criterion, val_loader, device)
        print(f"=========Epoch: {epoch+1}/{FLAGS.num_epochs}============")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_auc:.2f}")
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_auc:.2f}')
            
            
            
    # _, test_loss, test_auc = valid(model, criterion, test_loader, device)
    # print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_auc:.2f}')


if __name__ == '__main__':
    main()