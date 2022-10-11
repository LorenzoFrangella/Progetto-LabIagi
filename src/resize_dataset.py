import csv 
import os 
import glob
import pandas as pd

PATH = r'./'

all_gestures=[]

to_remove = ['x1','y1','z1',
              'x2','y2','z2',
              'x3','y3','z3',
              'x4','y4','z4',
              'x5','y5','z5',
              'x6','y6','z6',
              'x7','y7','z7',
              'x8','y8','z8',
              'x9','y9','z9',
              'x10','y10','z10',
              'x17','y17','z17',
              'x18','y18','z18',
              'x19','y19','z19',
              'x20','y20','z20',
              'x21','y21','z21',
              'x22','y28','z28',
              'x29','y29','z29',
              'x30','y30','z30',
              'x31','y31','z31',
              'x32','y32','z32']

rootdir = './data'
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        category = d.strip().split('/')[-1]
        all_gestures.append(category)

print(all_gestures)

for gesture in all_gestures:
    list_of_videos = glob.glob('./data/'+gesture+ '/'+ '*.csv')
    for video in list_of_videos:
        data = pd.read_csv(video)
        data.drop(to_remove, inplace=True, axis=1)
        data.to_csv(video,index=False)
