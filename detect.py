import tkinter as tk 
from tkinter import Message, Text 
import os 
import shutil 
import csv 
import numpy as np 
from PIL import Image, ImageTk 
import pandas as pd 
import datetime 
import time 
import tkinter.ttk as ttk 
import tkinter.font as font 
from pathlib import Path 
from mtcnn.mtcnn import MTCNN
from numpy import expand_dims
import cv2
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import pickle



    
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        detector = MTCNN()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = detector.detect_faces(img)
        for loc in face_locations:
            x1, y1, width, height = loc['box']
            cv2.rectangle(frame, (x1,y1), (x1+width,y1+height), (80,18,236), 2)
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face,(160,160))
            face = face.astype('float32')
            mean, std = face.mean(), face.std()
            face = (face - mean) / std

        

        cv2.imshow('Détection de visage',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()