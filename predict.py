#Author: Pranav Menon
#imporse
#arg
import argparse
#torch library
import torch
#variable
from torch.autograd import Variable
#transforms and models
from torchvision import transforms, models
#F
import torch.nn.functional as F
#np
import numpy as np
#Image
from PIL import Image
#json
import json
#os
import os
#random
import random


#method to load the checkpoint data
def load_checkpoint(filepath):
    #loading it to a variable checkpoint
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#method to load the files
def load_cat_names(filename):  
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

#method to parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    #checkpoint
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    #top k number of plants
    parser.add_argument('--top_k', dest='top_k', default='3')
    # default filepath to primrose image 
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') 
    #category
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    #gpu
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

#method to process the actual image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # use Image  
    img_pil = Image.open(image) 
    #final adjustments
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = adjustments(img_pil)
    
    return image

#method the predict what flower it is
def predict(image_path, model, topk=3, gpu='gpu'):
    ''' Get probability values (indeces) and respective flower classes. 
    '''
    
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1) # use F 
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()} # from reviewer advice
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

def main(): 
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    img_path = args.filepath
    probs, classes = predict(img_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs
    print('File selected: ' + img_path)
    
    print(labels)
    print(probability)
    # this prints out top numbers classes
    #and the proabbalililtes
    i=0 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probability[i]))
        
        i += 1 

if __name__ == "__main__":
    main()
