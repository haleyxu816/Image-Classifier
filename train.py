import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

parser = argparse.ArgumentParser(prog='train', usage='train a new network on a dataset and save the model as a checkpoint')
parser.add_argument("data_dir", type=str)
parser.add_argument("--save_dir", dest="save_dir",default='none', type=str)
parser.add_argument("--arch", dest="arch",default='vgg11', type=str)
parser.add_argument("--learning_rate", dest="learning_rate", default=0.001,type=float)
parser.add_argument("--hidden_units", dest="hidden_units",default=4096,type=int)
parser.add_argument("--epochs", dest="epochs", type=int,default=25)
parser.add_argument('--gpu', default=True,dest='gpu')
options = parser.parse_args()

def train_save_network(data_dir,save_dir,arch,learning_rate,hidden_units,epochs,gpu):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    valid_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_data=datasets.ImageFolder(valid_dir, transform=valid_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    
    if arch=='vgg11':
        model= models.vgg11(pretrained=True)
    if arch=='vgg13':
        model= models.vgg13(pretrained=True)
    if arch=='vgg16':
        model= models.vgg16(pretrained=True)
    if arch=='vgg19':
        model= models.vgg19(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1',nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    if gpu == True:
        device='cuda'
    else:
        device='cpu'
    epochs = epochs
    print_every = 40
    steps = 0
    
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()        
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        equality = (labels.data == ps.max(dim=1)[1])
                        accuracy += equality.type(torch.FloatTensor).mean()
                             
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
   
                running_loss = 0

    if save_dir=='none':
        pass
    else:
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {'input_size': 25088,
              'output_size': 102,
              'hidden_layers': [hidden_units],
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx,
              'optimizer_state_dict':optimizer.state_dict}

        torch.save(checkpoint, save_dir)

train_save_network(options.data_dir,options.save_dir,options.arch,options.learning_rate,options.hidden_units,options.epochs,options.gpu)
    
