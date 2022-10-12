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

# THESE ARE THE PARAMETERS FOR MY NEURAL NETWORK
# POINTS VARIABLE IS THE SIZE OF THE INPUT OF MY NEURAL NETWORK

POINTS = 33
HIDDEN_SIZE = 11

def findFiles(path): return glob.glob(path)

## FUNCTION TO LOAD DATAS FROM A CSV IN A LIST OF TENSORS

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

#FUNCTION TO LOAD DATA FROM CSV TO TENSOR

def csv_to_tensor(filecsv):
    train = pd.read_csv(filecsv)
    train_tensor = torch.tensor(train.values)
    return train_tensor


# ARRAY TO STORE THE NAME OFF ALL THE GESTURES

all_gestures = []

#THE CODE UNDER THIS COMMENT IS TO LOAD THE GESTURES NAME FROME THE DATA DIRECTORY
#USING THE FOLDERS INSIDE THIS AS NAMES OF THE GESTURES CAUSE THESE ARE DIVIDED 
#IN DIFFERENT DIRECTORIES FOR ANY CLASS

rootdir = './data'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        category = d.strip().split('/')[-1]
        all_gestures.append(category)

print(all_gestures)
num_gestures = len(all_gestures)


#THIS FUNCTION IS USED TO GET A RANDOM FILE FROM THE DATA DIR AS INPUT FOR OUR NEURAL NETWORK

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

#THIS FUNCTION GET AS INPUT THE RANDOM VIDEO FROM get_random_video() FUNCTION AND RETURNS 
# 1) THE CATEGORY WHICH THE FILE BELONGS TO AS A STRING
# 2) THE INPUT FILE NAME
# 3) A TENSOR WHICH IS A ONE HOT VECTOR FOR THE CATEGORY
# 4) THE INPUT DATA AS TENSOR (THAT IS A MATRIX)

def random_training_example():
    category, sequence = get_random_video()
    li = all_gestures.index(category)
    category_tensor = torch.tensor([all_gestures.index(category)], dtype=torch.long)
    #print(category_tensor)
    sequence_tensor = csv_to_tensor(sequence)
    return category,sequence,category_tensor,sequence_tensor


# SETTING THE CORRECT GRAPHIC ACCELERATION
device = "cuda" if torch.cuda.is_available() else "cpu"


print('using device: ',device)

##### SECONDO TIPO DI RETE TROVATA ONLINE CHE USA RNN 

class RNN(nn.Module):

    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.float()

        #combined = torch.cat((input_tensor.to(device), hidden_tensor.to(device)), 1)
        output,hidden = self.rnn(input_tensor,hidden_tensor)
        
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    


rnn = RNN(POINTS*3, HIDDEN_SIZE, num_gestures).to(device)



#criterion = nn.CrossEntropyLoss()
criterion = nn.NLLLoss()


learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):

    hidden = (torch.zeros(1,num_gestures),torch.zeros(1,num_gestures))
    skip_frame = 0
    for i in range(line_tensor.size()[0]):
        if skip_frame==0:
            output, hidden = rnn(line_tensor[i], hidden)
            skip_frame=0
        else:
            skip_frame=skip_frame-1
        

    loss = criterion(output.to(device), category_tensor.to(device)).to(device)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

if not os.path.exists(PATH + r"RNN.pth"):
    print('modello non allenato')

    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example()
        
        output, loss = train(line_tensor, category_tensor)
        current_loss += loss 
        
        if (i+1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
            
        if (i+1) % print_steps == 0:
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")
            print("valore dell output ",output[0])
    
    torch.save(rnn.state_dict(), PATH + r"RNN.pth")

else:
    rnn.load_state_dict(torch.load(PATH + r"RNN.pth",map_location=device))
    
        
    
plt.figure()
plt.plot(all_losses)
plt.show()

def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = csv_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i].to(device), hidden.to(device))
        
        guess = category_from_output(output)
        #print(guess)
        return(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break
    if sentence == "test":
        score = 0
        number_of_tests = 0
        print('start testing the result')
        for gesture in all_gestures:
            list_of_videos = glob.glob('./data/'+gesture+ '/'+ '*.csv')
            for elem in list_of_videos:
                solution = gesture
                predicted_value = predict(elem)
                if predicted_value == solution:
                    score +=1
                number_of_tests +=1
                print('test number: ',number_of_tests,' it must be: ', solution, ' it is: ', predicted_value)
        print('test ultimated, accuracy on the train dataset: ',score/number_of_tests)

    else: 
        print(predict(sentence))
    