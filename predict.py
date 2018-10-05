import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from torchvision import transforms
from PIL import Image
import json

parser = argparse.ArgumentParser(prog='predict', usage='predict flower name from an image')
parser.add_argument("image_path", type=str)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--top_k",dest=topk,default=5, type=int)
parser.add_argument("--category_names",dest=category_names,default='cat_to_name.json',type=str)
parser.add_argument('--gpu', default=True,dest='gpu')

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = models.vgg11(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(OrderedDict([
                          ('fc1',nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.state_dict=checkpoint['optimizer_state_dict']
    return model

model = load_checkpoint(checkpoint)

def process_image(image):    
    im = Image.open(image)
    if im.size[0] > im.size[1]:
        im.thumbnail((10000, 256))
    else:
        im.thumbnail((256, 10000))
    pil_image=transforms.CenterCrop(224)(im)
    np_image = np.array(pil_image)/225
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=np_image-mean/std
    np_image=np_image.transpose((2,0,1))
    
    return np_image
if gpu == True:
    device='cuda'
else:
    device='cpu'
    
def predict(image_path, model, topk):
    im=process_image(image_path)
    im = torch.from_numpy(im).type(torch.FloatTensor) 
    im.unsqueeze_(0)
    model.to(device)
    model.eval()
    with torch.no_grad():
        im=im.to(device)
        output = model.forward(im)
    ps = torch.exp(output)
    ps_k=ps.topk(topk)
    
    return ps_k

im = process_image(image_path)
probs, classes_idx=predict(image_path,model,topk)


with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

classes_idx=classes_idx.numpy()
def get_keys(d, value):
    return [k for k,v in d.items() if v == value]
classes=list()
for i in range(topk):
    classes.append(get_keys(model.class_to_idx,classes_idx[0][i]))
for i in range(topk):
    classes[i]=cat_to_name[classes[i][0]]
    
print(classes,probs)
 