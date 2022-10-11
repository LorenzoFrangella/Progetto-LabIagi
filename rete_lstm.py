from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
from re import A
from tkinter import HIDDEN
import unicodedata
import string
import random
import time
import math
import csv
import pandas as pd

import torch
from torch import nn
import matplotlib.pyplot as plt

PATH = r"./"

# DEFINIAMO DEGLI IPERPARAMETRI PER LA NOSTRA RETE
POINTS = 33
HIDDEN_SIZE = 128

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

print(all_gestures)
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
    sequence_tensor = csv_to_tensor(sequence)
    return category_tensor, sequence_tensor


#PRIMO TIPO DI RETE NEURALE TROVATA ONLINE CHE USA LSTM

def timeSince(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=in_features, hidden_size=hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.mlp_1 = nn.Linear(hidden_dim, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        x, (hn, cn) = self.lstm(x)
        
        x = torch.transpose(hn, 0, 1)
        x = self.flatten(x)
        x = self.softmax(self.mlp_1(x))
        return x



if __name__ == "__main__":
    # Build the category_lines dictionary, a list of lines per category

    all_categories = all_gestures
    n_categories = len(all_categories)


    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')
    

    print('# categories:', n_categories, all_categories)

    learning_rate = 0.005

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    rnn = RNN(POINTS*3, HIDDEN_SIZE, n_categories).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate, amsgrad=True)


    ##### SE NON ESISTE UN FILE RNN.PTH VUOL DIRE CHE LA NOSTRA RETE NON è STATA ALLENATA
    ##### QUINDI INIZIAMO CON LA FASE DI TRAINING

    if not os.path.exists(PATH + r"\RNN.pth"):
        n_iters = 1000000
        print_every = 5000
        plot_every = 500
        all_losses = []
        total_loss = 0  # Reset every plot_every iters

        start = time.time()

        for iter in range(1, n_iters + 1):
            input_line_tensor, target_line_tensor = random_training_example()
            
            hidden = torch.zeros(1, HIDDEN_SIZE)

            # training
            rnn.zero_grad()
            
            #batch_category = category_tensor.expand(input_line_tensor.shape[0], category_tensor.shape[1])
            input_data = input_line_tensor

            out = rnn(input_data.to(device))
            loss = criterion(out, target_line_tensor.to(device))
            loss.backward()
            optimizer.step()

            total_loss += loss

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

            if iter % plot_every == 0:
                all_losses.append(total_loss.to(torch.device("cpu")).detach().numpy() / plot_every)
                total_loss = 0
                fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                ax.plot(all_losses)
                fig.savefig('losses.png')  # save the figure to file
                plt.close(fig)

        plt.figure()
        plt.plot(all_losses)
        plt.show()

        torch.save(rnn.state_dict(), PATH + r"\RNN.pth")
    else:
        rnn.load_state_dict(torch.load(PATH + r"\RNN.pth"))