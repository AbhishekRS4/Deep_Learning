import os
import sys
import time
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from models import ResNetImageClassififer
from dataset import split_dataset, get_dataloader_for_testing
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def test(model, test_loader, device):
    model.eval()
    num_test_samples = len(test_loader.dataset)
    num_test_batches = len(test_loader)
    test_pred_labels = []

    for data, label in test_loader:
        data = data.to(device, dtype=torch.float)
        logits = model(data)
        pred_probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_label = pred_label.detach().cpu().numpy()

        test_pred_labels.append(pred_label)
    return np.array(test_pred_labels)

def test_classifier(FLAGS):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA device not found, so exiting....")
        sys.exit(0)

    test_x, test_y, num_classes = split_dataset(FLAGS.file_labels_csv, is_for_train=False)
    num_test_samples = len(test_x)

    test_loader = get_dataloader_for_testing(
        test_x, test_y, dir_images=FLAGS.dir_images, image_size=FLAGS.image_size, batch_size=1,
    )

    print("\ntest set details")
    print(f"num test samples: {num_test_samples}")
    print(f"num classes: {num_classes}")

    print("\ntesting started")
    print(f"model file used : {FLAGS.file_model}")
    model = ResNetImageClassififer(num_classes=num_classes, pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(FLAGS.file_model))
    test_pred_labels = test(model, test_loader, device)

    test_acc = accuracy_score(test_y, test_pred_labels)
    test_cm = confusion_matrix(test_y, test_pred_labels)
    test_f1 = f1_score(test_y, test_pred_labels, average="weighted")
    print("***************************")
    print("test set evaluation metrics")
    print("***************************")
    print(f"accuracy score : {100*test_acc:.4f}")
    print(f"f1 score : {100*test_f1:.4f}")
    print("confusion matrix")
    print(test_cm)
    print("\ntesting completed")
    return

def main():
    image_size = 320
    dir_images = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/images/"
    file_model = "model_pretrained/pretrained_10.pt"
    file_labels_csv = "/home/abhishek/Desktop/deep_learning/cassava_image_classification_dataset/image_labels.csv"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image_size", default=image_size,
        type=int, help="image size used to train the model")
    parser.add_argument("--dir_images", default=dir_images,
        type=str, help="full directory path to dataset containing images")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="full path to model file")
    parser.add_argument("--file_labels_csv", default=file_labels_csv,
        type=str, help="full path to csv file with image ids and labels")

    FLAGS, unparsed = parser.parse_known_args()
    test_classifier(FLAGS)
    return

if __name__ == "__main__":
    main()
