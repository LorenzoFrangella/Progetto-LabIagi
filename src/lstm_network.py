from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
from re import A
import random
import csv
import pandas as pd

import torch
from torch import nn
import matplotlib.pyplot as plt

PATH = r"./"

# DEFINIAMO DEGLI IPERPARAMETRI PER LA NOSTRA RETE
POINTS = 13
HIDDEN_SIZE = 16
NUM_LAYERS = 1 

def findFiles(path): return glob.glob(path)

## FUNZIONE PER CARICARE I DATI DAI FILE CSV

def csv_to_list_of_tensor(file):

    list_of_tensor = []

    with open(file,'r') as csvfile:
        rowread = csv.reader(csvfile)
        i=0
        for row in rowread:
            if i==0:
                i+=1
                continue
            floats = [float(x) for x in row]
            tensor_output = torch.Tensor(floats)
            list_of_tensor.append(tensor_output)
    return list_of_tensor

def csv_to_tensor(filecsv):
    train = pd.read_csv(filecsv)
    train_tensor = torch.tensor(train.values)
    return train_tensor


all_gestures = []

#print(os.walk('./'))

rootdir = './data'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        category = d.strip().split('/')[-1]
        all_gestures.append(category)

#print(all_gestures)
num_gestures = len(all_gestures)



def get_random_video():
    random_index = random.randrange(0,num_gestures)
    random_gesture = all_gestures[random_index]
    list_of_videos = glob.glob('./data/'+random_gesture+ '/'+ '*.csv')
    num_videos = len(list_of_videos)
    random_index = random.randrange(0,num_videos)
    return random_gesture,list_of_videos[random_index]



def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_gestures[category_idx]

def random_training_example():
    category, sequence = get_random_video()
    li = all_gestures.index(category)
    category_tensor = torch.tensor([all_gestures.index(category)], dtype=torch.long)
    #print(category_tensor)
    sequence_tensor = csv_to_tensor(sequence)
    return category,sequence,category_tensor,sequence_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

#print(random_training_example())

#### costruiamo la nostra rete neurale

class RNN(nn.Module):

    # creiamo una rete neurale ricorrente per classificare le gesture
    # tipo della rete: lstm

    def __init__(self, input_size, hidden_size, output_size):

        num_layers = NUM_LAYERS
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size,hidden_size = hidden_size,num_layers = num_layers,proj_size=output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x, (h_n, c_n) = self.lstm(x)
        x = self.softmax(x.double())
        return x


rnn = RNN(13*3,13,11)


tensor_input = random_training_example()[3]


print(tensor_input)

tensor_input = tensor_input.double()

test = torch.randn(30,39)

print(tensor_input)
print(test)

#output = rnn(test)

#print(output)