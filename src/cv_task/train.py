import os
import sys
import time
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from logger_utils import CSVWriter
from models import ResNetImageClassififer
from dataset import split_dataset, get_dataloaders_for_training

def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    num_train_samples = len(train_loader.dataset)
    num_train_batches = len(train_loader)

    for data, label in train_loader:
        data = data.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        train_running_loss += loss.item()
        pred_label = torch.argmax(logits, dim=1)
        train_running_correct += (pred_label == label).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / num_train_batches
    train_accuracy = 100. * train_running_correct / num_train_samples
    return train_loss, train_accuracy

def validate(model, criterion, valid_loader, device):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    num_valid_samples = len(valid_loader.dataset)
    num_valid_batches = len(valid_loader)

    with torch.no_grad():
        for data, label in valid_loader:
            data = data.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            logits = model(data)
            loss = criterion(logits, label)

            valid_running_loss += loss.item()
            pred_label = torch.argmax(logits, dim=1)
            valid_running_correct += (pred_label == label).sum().item()

        valid_loss = valid_running_loss / num_valid_batches
        valid_accuracy = 100. * valid_running_correct / num_valid_samples
    return valid_loss, valid_accuracy

def train_classifier(FLAGS):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    train_x, valid_x, train_y, valid_y, num_classes = split_dataset(FLAGS.file_labels_csv, is_for_train=True)
    num_train_samples = len(train_x)
    num_valid_samples = len(valid_x)
    #print(valid_x[0:5])
    #print(valid_y[0:5])
    train_loader, valid_loader = get_dataloaders_for_training(
        train_x, train_y, valid_x, valid_y,
        dir_images=FLAGS.dir_images, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
    )

    if FLAGS.pretrained:
        file_model = "pretrained"
        dir_model = f"model_{file_model}"
    else:
        file_model = "random"
        dir_model = f"model_{file_model}"

    if not os.path.isdir(dir_model):
        print(f"Creating directory: {dir_model}")
        os.makedirs(dir_model)

    print(f"num train samples: {num_train_samples}, num validation samples: {num_valid_samples}")
    print(f"num classes: {num_classes}")
    model = ResNetImageClassififer(num_classes=num_classes, pretrained=FLAGS.pretrained)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    csv_writer = CSVWriter(
        file_name=os.path.join(dir_model, FLAGS.file_logger_train),
        column_names=["epoch", "loss_train", "loss_valid", "accuracy_train", "accuracy_valid"]
    )

    print("Training of cassava leaf image classification started")
    for epoch in range(1, FLAGS.num_epochs+1):
        time_start = time.time()
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
        valid_loss, valid_acc = validate(model, criterion, valid_loader, device)
        time_end = time.time()
        print(f"Epoch: {epoch}/{FLAGS.num_epochs}, time: {time_end-time_start:.3f} sec.")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.3f}")
        print(f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.3f}\n")
        csv_writer.write_row(
            [
                epoch,
                round(train_loss, 6),
                round(valid_loss, 6),
                round(train_acc, 6),
                round(valid_acc, 6),
            ]
        )
        torch.save(model.state_dict(), os.path.join(dir_model, f"{file_model}_{epoch}.pt"))
    print("Training of cassava leaf image classification complete!!!!")
    csv_writer.close()
    return

def main():
    learning_rate = 1e-4
    weight_decay = 1e-6
    batch_size = 64
    num_epochs = 100
    image_size = 320
    file_logger_train = "train_metrics.csv"
    dir_images = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/images/"
    file_labels_csv = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/image_labels.csv"
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
    parser.add_argument("--file_labels_csv", default=file_labels_csv,
        type=str, help="full path to csv file with image ids and labels")

    FLAGS, unparsed = parser.parse_known_args()
    train_classifier(FLAGS)
    return

if __name__ == "__main__":
    main()
