from ctypes import pointer
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import torch
import torch.nn as nn
import os

PATH = r"./"

heading_fout = ['x0','y0','z0',
                'x1','y1','z1',
                'x2','y2','z2',
                'x3','y3','z3',
                'x4','y4','z4',
                'x5','y5','z5',
                'x6','y6','z6',
                'x7','y7','z7',
                'x8','y8','z8',
                'x9','y9','z9',
                'x10','y10','z10',
                'x11','y11','z11',
                'x12','y12','z12',
                'x13','y13','z13',
                'x14','y14','z14',
                'x15','y15','z15',
                'x16','y16','z16',
                'x17','y17','z17',
                'x18','y18','z18',
                'x19','y19','z19',
                'x20','y20','z20',
                'x21','y21','z21',
                'x22','y22','z22',
                'x23','y23','z23',
                'x24','y24','z24',
                'x25','y25','z25',
                'x26','y26','z26',
                'x27','y27','z27',
                'x28','y28','z28',
                'x29','y29','z29',
                'x30','y30','z30',
                'x31','y31','z31',
                'x32','y32','z32',]


# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray

def input_to_tensor(list):
    new_row = []
    for elem in list:
        new_row.append(elem.x)
        new_row.append(elem.y)
        new_row.append(elem.z)
    data = torch.tensor(new_row)
    return data


    
### Inizializzare il programma di riconoscimento delle gesture che poi a sua volta avvierà la fotocamera

all_gestures = []
rootdir = './data'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        category = d.strip().split('/')[-1]
        all_gestures.append(category)

print(all_gestures)

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(PATH + r"RNN.pth"):
    quit('No configuration file for the model')

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_gestures[category_idx]

class RNN(nn.Module):

    
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
    
num_gestures = len(all_gestures)
print(num_gestures)


rnn = RNN(99, 11, num_gestures).to(device)
rnn.load_state_dict(torch.load(PATH + r"RNN.pth",map_location=device))
clip =0

cap = cv2.VideoCapture(0)
counter = 27
hidden_state=rnn.init_hidden()
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks != None:

        points = list(results.pose_landmarks.landmark)
        tensore = input_to_tensor(points)

        if counter == 0:
            print('counter scaduto',clip)
            clip = clip+1
            result,hidden_state = rnn(tensore,hidden_state)
            print(category_from_output(result))
            hidden_state= rnn.init_hidden()
            counter = 30
        else:
            result,hidden_state = rnn(tensore,hidden_state)
            counter = counter-1

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(2) & 0xFF == 27:
      break
cap.release()

