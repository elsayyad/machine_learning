import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

from collections import OrderedDict

import time
import glob
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models.densenet import model_urls

from PIL import Image
import argparse

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    if title:
        ax.set_title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def fetch_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    class_to_idx = train_data.class_to_idx

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    return train_loader, valid_loader, test_loader, class_to_idx

def save_model(model_checkpoint, save_dir_path, file_name = 'checkpoint-cmd-final.pth'):
    #time = time.strftime("-%Y-%m-%d-%H:%M.cp")
    path = save_dir_path + file_name
    torch.save(model_checkpoint, path)


def train_model(train_loader, valid_loader, test_loader, arch, hidden_layer, learning_rate, dropout, epochs, device, save_dir_path, class_to_idx):


    device = "cuda" if torch.cuda.is_available() and device=="cuda" else "cpu"
    #model_init(hidden_layer, dropout, learning_rate, device)

    ## Create Model ##

    supported_models = {
        'vgg16': (models.vgg16, 25088),
        'vgg16_bn': (models.vgg16_bn, 25008),
        'densenet121': (models.densenet121, 1024),
        'resnext50': (models.resnet50,2048)
    }

    supported_models_result = supported_models.get(arch, None)
    if supported_models_result == None:
        print("\nModel architecture not found. Supported models are: \n{}".format('\n'.join(supported_models.keys())))

    #model_urls['densenet121'] = model_urls['densenet121'].replace('https://', 'http://')
    model = supported_models_result[0](pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier  = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(supported_models_result[1], hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layer, 102)),
                          ('dropout', nn.Dropout(p=dropout)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.class_to_idx = class_to_idx
    model.to(device);

    start_time = time.time()

    steps = 0
    running_loss = 0
    print_every = 5
    train_losses, valid_losses = [], []

    for epoch in range(epochs):

        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



                train_losses.append(running_loss/len(train_loader))
                valid_losses.append(valid_loss/len(valid_loader))

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}")

                running_loss = 0
                model.train()


                model_checkpoint = {'model':supported_models_result[0],
                                    'model_base_name':arch,
                                    'model_input':supported_models_result[1],
                                    'model_hidden_layer':hidden_layer,
                                    'model_state_dict':model.state_dict(),
                                    'model_class_to_idx':model.class_to_idx,
                                    'optimizer':optimizer,
                                    'optim_state_dict':optimizer.state_dict(),
                                    'dropout':dropout
                                   }
                save_model(model_checkpoint, save_dir_path)

    total_time = time.time() - start_time
    total_time = time.strftime('%H:%M:%S', time.localtime(total_time))
    print("\n** Total Elapsed Training Runtime: ", total_time)

def main():

    parser = argparse.ArgumentParser(description='Training a Nureal Network Model')

    parser.add_argument('data_dir', type=str, help='Data directory path for Images with subdirs(Train, Valid, Test)')
    parser.add_argument('--save_dir_path', type=str, help='Directory to save the checkpoint in')
    parser.add_argument('--arch', type=str, help='Archictecture of neural network model. Default: densenet121')
    parser.add_argument('--hidden_layer', type=int, help='Hidden layer variable. Default=250')
    parser.add_argument('--dropout', type=float, help='Dropout float variable. Default=0.4')
    parser.add_argument('--gpu', action='store_true', help='Use GPU (cuda) if available otherwise CPU')
    parser.add_argument('--learning_rate', type=float,help='Floar variable for learning rate. Default: 0.01')
    parser.add_argument('--epochs', type=int, help='Epochs integer number. Default: 3')

    args, _ = parser.parse_known_args()

    data_dir = args.data_dir

    save_dir_path = './'
    if args.save_dir_path:
        save_dir_path = args.save_dir_path

    arch = 'densenet121'
    if args.arch:
        arch = args.arch

    hidden_layer = 250
    if args.hidden_layer:
        hidden_layer = args.hidden_layer

    dropout = 0.4
    if args.dropout:
        dropout = args.dropout

    learning_rate = 0.01
    if args.learning_rate:
        learning_rate = args.learning_rate

    epochs = 3
    if args.epochs:
        epochs = args.epochs

    device = "cpu"
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
            device = "cuda"
        else:
            cuda = False
            device = "cpu"
            print("No GPU found, switching to cpu ..")

    train_loader, valid_loader, test_loader, class_to_idx =  fetch_data(data_dir)

    train_model(train_loader, valid_loader, test_loader,
                arch = arch,
                hidden_layer = hidden_layer,
                learning_rate = learning_rate,
                dropout=dropout,
                epochs=epochs,
                device=device,
                save_dir_path=save_dir_path,
                class_to_idx=class_to_idx)



if __name__ == '__main__':
    main()
