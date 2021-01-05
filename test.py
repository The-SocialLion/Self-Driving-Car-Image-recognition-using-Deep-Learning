from PIL import ImageOps
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image# used for preproccesing 
model = load_model('SDC.h5')
print("Loaded model from disk")

classs = { 1:"Left",
           2:"Center",
           3:"Right"          
           }


Img=64
def classify(img_file):
    test_image=image.load_img(img_file)
    test_image=ImageOps.grayscale(test_image)
    test_image = test_image.resize((64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    test = np.array(test_image).reshape(-1,Img,Img,1)
    result = model.predict_classes(test)[0]
    sign = classs[result + 1]
    print("The decision is ",sign)
    
print("Obtaining Images & its Labels..............")
path='D:/python/dl programs/Self Driving Car/data/test'
files=[]
print("Dataset Loaded")
# r=root,d=directories,f=files
for r,d,f in os.walk(path):
    for file in f:
        if '.jpeg' or '.jpg' or '.png' or '.JPG' in file:
            files.append(os.path.join(r,file))
for f in files:
    classify(f)
    print('\n')
