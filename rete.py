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
POINTS = 33
HIDDEN_SIZE = 1

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
    #print(category_tensor)
    sequence_tensor = csv_to_tensor(sequence)
    return category,sequence,category_tensor,sequence_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"


##### SECONDO TIPO DI RETE TROVATA ONLINE CHE USA RNN 

class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.float()

        combined = torch.cat((input_tensor.to(device), hidden_tensor.to(device)), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
# Inizializziamo la rete neurale ricorrente con i parametri giusti ovvero la dimensione dell input:33 punti per 3 coordinate ciascuno,
# la dimensione delle nostro stato nascosto per ora abbiamo scelto 128, e infine la dimensione dell output che dovrà essere una lista
# con dimensione uguale al numero di gesture che vogliamo riconoscere

rnn = RNN(POINTS*3, HIDDEN_SIZE, num_gestures).to(device)


#
criterion = nn.CrossEntropyLoss()

#criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    
    skip_frame =0
    for i in range(line_tensor.size()[0]):
        if skip_frame==0:
            output, hidden = rnn(line_tensor[i], hidden)
            skip_frame=4
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
            print("valore dell output ",output)
    
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
    

'''

#PRIMO TIPO DI RETE NEURALE TROVATA ONLINE CHE USA LSTM

def timeSince(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        super(RNN, self).__init__()

        self.lstm = nn.LSTM(input_size=in_features, HIDDEN_SIZE=hidden_dim, num_layers=1, batch_first=True, dropout=0.2)
        self.flatten = nn.Flatten()
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

    all_categories = [os.listdir('./data/light/')]
    n_categories = len(all_categories)


    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')
    

    print('# categories:', n_categories, all_categories)

    learning_rate = 0.005


    assert torch.cuda.is_available(), "Notebook is not configured properly!"
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
            category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
            hidden = torch.zeros(1, HIDDEN_SIZE)

            # training
            rnn.zero_grad()

            batch_category = category_tensor.expand(input_line_tensor.shape[0], category_tensor.shape[1])
            input_data = torch.cat((batch_category.unsqueeze(1), input_line_tensor),2)

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
'''