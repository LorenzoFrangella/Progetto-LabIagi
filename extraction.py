# Importazione delle librerie necessarie

from bdb import Breakpoint
from operator import truediv
from re import I
import cv2
import mediapipe as mp
import pandas as pd
import os
from os import walk
from os import walk
import csv

# definizione delle variabili necessarie a mediapipe per effettuare il tracciamento dei punti

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# funzione che prende in ingresso un video e operando l'estrazione dei punti con mediapipe
# ci restituisce un file di tipo csv che viene salvato nella cartella ./input. Il file csv
# contiene al suo interno una riga per ogni frame del video che abbiamo processato, per ogni
# riga abbiamo 33 punti che rappresetano il mapping del corpo.
# Ogni punto è salvato in una struttura dati organizzata in tale modo:
#
# x: floatValue (0,1)
# y: floatValue (0,1)
# z: floatValue (0,1)
# Visibility: floatValue (0,1)
#
# tutti i valori sono dei valori in percentuale e le coordinate indicano la posizione all'interno 
# dell'immagine con valori percentuali


def feature_extraction(file):
    # caricamento del video come input
    cap = cv2.VideoCapture(file)
    # operazione sul nome del file per ottenere il nome del file di destinazione
    filename = file.strip().split('/')[-1].strip().split('.')[0]
    heading = range(33)
    with open('./input/'+filename+'.csv','w') as csvfile:
        write = csv.writer(csvfile)
        write.writerow(heading)
        # parte di codice che usa mediapipe per l'estrazione delle features
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break;
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # salvataggio dei punti estratti come lista di punti
                results = pose.process(image)
                if results.pose_landmarks != None:
                    points = list(results.pose_landmarks.landmark)
                    write.writerow(points)
            cap.release()
    return

#F è la lista dei file presenti nella cartella raw
'''
raw = []
for (dirpath, dirnames, filenames) in walk('./data/raw'):
    raw.extend(filenames)
    break




convertiti =[]



with open('elenco.txt','r') as f:
    lines = f.readlines()
    print(lines)
    for elem in lines:
        nome_file = elem.replace('\n',"")
        convertiti.append(nome_file)
        print(nome_file)

for file in raw:
    if file in convertiti:
        print('il file',file,' è già nella lista dei convertiti')
    else:
        feature_extraction('./data/raw/'+file)
        with open('elenco.txt','a') as f:
            f.write(file+'\n')

extracted = []
for (dirpath, dirname, filenames) in walk('./data/extracted'):
    extracted.extend(filenames)
    break
print(extracted)

'''
# heading del file csv finale dove i punti sono salvati in una lista di 99 elementi così organizzati
# [x0,y0,z0, ... ,x32,y32,z32]


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
'''
for file in extracted:
    if file == '.DS_Store':
        continue
    with open('./data/light/'+file,'w') as fout:
        write = csv.writer(fout)
        write.writerow(heading_fout)
        with open('./data/extracted/'+file,'r') as f1:
            csvreader = csv.reader(f1)
            i=0
            for row in csvreader:
                if i==0:
                    i+=1
                    continue
                new_row = []
                for elem in row:
                    step = elem.strip().split('\n')
                    x = float(step[0].replace('x: ',''))
                    y = float(step[1].replace('y: ',''))
                    z = float(step[2].replace('z: ',''))
                    new_row.append(x)
                    new_row.append(y)
                    new_row.append(z)
                write.writerow(new_row)

'''

raw = []
for (dirpath, dirnames, filenames) in walk('./input'):
    raw.extend(filenames)
    break
print(raw)

for file in raw:
    feature_extraction('./input/'+file)
    os.remove('./input/'+file)

extracted = []
for (dirpath, dirname, filenames) in walk('./input'):
    extracted.extend(filenames)
    break
print(extracted)

for file in extracted:
    if file == '.DS_Store':
        continue
    with open('./output/'+file,'w') as fout:
        write = csv.writer(fout)
        write.writerow(heading_fout)
        with open('./input/'+file,'r') as f1:
            csvreader = csv.reader(f1)
            i=0
            for row in csvreader:
                if i==0:
                    i+=1
                    continue
                new_row = []
                for elem in row:
                    step = elem.strip().split('\n')
                    x = float(step[0].replace('x: ',''))
                    y = float(step[1].replace('y: ',''))
                    z = float(step[2].replace('z: ',''))
                    new_row.append(x)
                    new_row.append(y)
                    new_row.append(z)
                write.writerow(new_row)
