import cv2
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

model_path="vgg16_image_classification_model.h5"
model=load_model(model_path)

with open("class_names.json") as f:
    class_dict=json.load(f)

test_image_path=r"Data"

image_obj=cv2.imread(test_image_path)
test_image=cv2.resize(image_obj,(224, 224))

#==========Convert image to numpy array and normalize==========
test_image=img_to_array(test_image)/255
#=======================change dimention 3D to 4D==============
test_image=np.expand_dims(test_image,axis=0)

result=model.predict(test_image)

pred=np.argmax(result,axis=1)

predicted_val=int(pred)

predicted_disease=class_dict[str(predicted_val)]
print("Disease : ",predicted_disease)
