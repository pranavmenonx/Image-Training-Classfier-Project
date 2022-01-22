#Author: Pranav Menon
#imports
#args
import argparse
#torch library
import torch
#nn
from torch import nn
#optim
from torch import optim
#variable
from torch.autograd import Variable
#datasets, transforms, models
from torchvision import datasets, transforms, models
#Image Folder
from torchvision.datasets import ImageFolder
#F
import torch.nn.functional as F
#Image
from PIL import Image
#OrderedDict
from collections import OrderedDict
#time
import time
#np
import numpy as np
#plt
import matplotlib.pyplot as plt


#save_checpoint method
def save_checkpoint(path, model, optimizer, args, classifier):
    #checkpoint
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    #torch save
    torch.save(checkpoint, path) # the path will be user defined, if not it autosets to checkpoint.pth

#parser method
def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    #data directory
    parser.add_argument('data_dir', action='store')
    #type of training
    parser.add_argument('--arch', dest='arch', default='vgg13', choices=['vgg13', 'densenet121'])
    #learning rate
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    #hidden units
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    #epochs
    parser.add_argument('--epochs', dest='epochs', default='20')
    #gpu
    parser.add_argument('--gpu', action='store', default='gpu')
    #save directory
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

#train method
def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 10
    for e in range(epochs):
        running_loss = 0
        #0 means to train
        for ii, (inputs, labels) in enumerate(dataloaders[0]): 
            steps += 1 
            if gpu == 'gpu':
                model.cuda()
                # use cuda
                inputs, labels = inputs.to('cuda'), labels.to('cuda') 
            else:
                model.cpu()
            optimizer.zero_grad()
            #forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            #backward pass
            loss.backward()
            optimizer.step()
            #runningloss
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valloss = 0
                accuracy=0
                # 1 is validation
                for ii, (inputs2,labels2) in enumerate(dataloaders[1]):  
                        optimizer.zero_grad()
                        
                        if gpu == 'gpu':
                            # use cuda
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda') 
                            # use cuda
                            model.to('cuda:0') 
                        else:
                            pass 
                        with torch.no_grad():    
                            outputs = model.forward(inputs2)
                            valloss = criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                #valloss calc
                valloss = valloss / len(dataloaders[1])
                #accuracy
                accuracy = accuracy /len(dataloaders[1])
                #Epoch, Training Loss, Validation Loss, Accuracy
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(valloss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
            
def main():
    
    args = parse_args()   
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(val_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]
   
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
    #neuralnetwork loss
    criterion = nn.NLLLoss()
    #optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    #epochs
    epochs = int(args.epochs)
    #convert
    class_index = image_datasets[0].class_to_idx
    # get the gpu settings
    gpu = args.gpu 
    #call train
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    #convert
    model.class_to_idx = class_index
    # get the new save location 
    path = args.save_dir 
    #checkpoint!
    save_checkpoint(path, model, optimizer, args, classifier)

#calling the main method
#During this part of the project for both the train.py and predict.py 
#section, I referenced this github link https://github.com/kwahid, 
#which helped me understand how to do a lot of it. After looking at
#his code I know fully understand how to do this :D

if __name__ == "__main__":
    main()
