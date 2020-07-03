from numpy import expand_dims
import cv2
from numpy import expand_dims
import pickle
import cv2
#from google.colab.patches import cv2_imshow
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np


model = load_model('facenet_keras.h5')
#print("FaceNet Model Loaded")

emb=[]
names=["Angelina_Jolie","Bill_Gates","John_Snow","Muhammad_Ali","Zinedine_Zidane","elton_john"]

cap = cv2.VideoCapture(0)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
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
    
          face = expand_dims(face, axis=0)
    
          embed = model.predict(face)
          emb.append(embed)
          filename = 'finalized_model.sav'
          prediction_model = pickle.load(open(filename, 'rb'))
          yhat_class = prediction_model.predict(embed)
    
          yhat_prob = prediction_model.predict_proba(embed)
    
    
          class_index = yhat_class[0]
  #print("Index",class_index)
          class_probability = yhat_prob[0,class_index] * 100
          print('Prediction Probablity:%.3f' %(class_probability))
    #setting threshold based on probability
          if(class_probability>99.5):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,names[class_index],(x1-10,y1-10), font, 1, (200,255,155)) 
            cv2.imshow('image',frame)
          else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'unknown',(x1-10,y1-10), font, 1, (200,255,155)) 
            cv2.imshow('image',frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break  
cap.release()
cv2.destroyAllWindows()