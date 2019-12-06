import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.0005')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()


def train(model, criterion, optimizer, dataloaders, epochs, gpu):

    steps = 0
    print_every = 10
    
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs, labels) in enumerate(dataloaders[0]):
            steps += 1
            
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            else:
                model.cpu() # use a CPU if user says anything other than "gpu"
                
            # zeroing parameter gradients
            optimizer.zero_grad() 
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss=0
                accuracy=0

                for ii, (images, labels) in enumerate(dataloaders[1]):
                    optimizer.zero_grad()

                    if gpu == 'gpu':
                        images, labels = images.to('cuda') , labels.to('cuda') 
                        model.to('cuda:0')
                    else:
                        pass # just use inputs as is     
                        
                    with torch.no_grad():
                        output = model.forward(images)
                        valid_loss += criterion(output,labels).item()
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()

                valid_loss = valid_loss/ len(dataloaders[1])
                accuracy = accuracy/ len(dataloaders[1])

                print(f"epochs: {e+1}, \
                Training Loss: {round(running_loss/print_every,3)} \
                Valid Loss: {round(valid_loss,3)} \
                Valid Accuracy: {round(float(accuracy),3)}")

                running_loss = 0
                
                

def main():
    print("Start training")  
    args = parse_args()
    
    data_dir = 'ImageClassifier/flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_data.class_to_idx
    gpu = args.gpu # get the gpu settings
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir 
    save_checkpoint(path, model, optimizer, args, classifier)


if __name__ == "__main__":
    main()