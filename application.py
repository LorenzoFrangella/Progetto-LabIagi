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
    print('probability: ', output[0][category_idx].item())
    return all_gestures[category_idx],output[0][category_idx].item()

class RNN(nn.Module):

    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
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
      break
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
            categoria, probabilità = category_from_output(result)
            if probabilità>=0.60:
                print(categoria)
            else:
                print('no gesture detected')
            hidden_state= rnn.init_hidden()
            counter = 27
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

