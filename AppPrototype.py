##########################################################################
import numpy as np
import pandas as pd
import os
pd.set_option('display.max_columns', None)
from tensorflow.keras.models import load_model
import cv2
from googletrans import Translator, constants
import playsound
import time
from gtts import gTTS
#########################################################################

# UNPICKLE FILES
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# IMPORT LABELS
meta = unpickle('cifar-100-python/meta')
Superclass_label = pd.DataFrame(meta[b'coarse_label_names']).rename(columns={0: 'Superclass_label'})
SL = Superclass_label['Superclass_label']

##########################################################################
# SOME PRESET PARAMETERS FOR THE WEBCAM
frameWidth = 32
frameHeight = 32
brightness = 180
threshold = 0.7 # THIS IS THE PROBABILITY THRESHOLD THAT MUST BE MET
font = cv2.FONT_ITALIC

# INDICATE LANGUAGE TO TRANSLATE TO
trans_lang = 'fr'
####################################################

# SET UP CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT TRAINED MODEL
model = load_model('images/CNN_model4.h5')

# scale image
def preprocessing(img):
    img = img/255
    return img

# Once the model identifies the class number, this is to get the corresponding class name.
def getclassname(classno):
    if classno == 0:
        return 'aquatic mammals'
    elif classno == 1:
        return 'fish'
    elif classno == 2:
        return 'flowers'
    elif classno == 3:
        return 'food containers'
    elif classno == 4:
        return 'fruits and vegetables'
    elif classno == 5:
        return 'household electrical devices'
    elif classno == 6:
        return 'household furniture'
    elif classno == 7:
        return 'insects'
    elif classno == 8:
        return 'large carnivores'
    elif classno == 9:
        return 'large man made outdoor things'
    elif classno == 10:
        return 'large natural outdoor scenes'
    elif classno == 11:
        return 'large omnivores and herbivores'
    elif classno == 12:
        return 'medium mammals'
    elif classno == 13:
        return 'non-insect invertebrates'
    elif classno == 14:
        return 'people'
    elif classno == 15:
        return 'reptiles'
    elif classno == 16:
        return 'small mammals'
    elif classno == 17:
        return 'trees'
    elif classno == 18:
        return 'vehicles'
    elif classno == 19:
        return 'vehicles 2'

# TRANSLATE CLASS NAME INTO A DESIRED LANGUAGE
def translate(class_):
    translator = Translator()
    translation = translator.translate(class_, dest=trans_lang)
    print(translation.text, translation.dest)
    return translation.text, translation.dest

# TO AUDIBLY DESCRIBE THE CLASS NAME
def speak(text):
    tts = gTTS(text=text, lang=trans_lang)
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

# INITIATE THE CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    # CAPTURE IMAGE FROM WEBCAM
    success, imgOriginal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 3)
    cv2.putText(imgOriginal, "Class:", (20,35), font, 0.75, (0,0,225), 2, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.putText(imgOriginal, "Probability:", (20, 75), font, 0.75, (225, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)

    # PREDICT IMAGE
    predictions = model.predict(img)

    # PICK THE PREDICTION WITH HIGHEST PROBABILITY
    classIndex = np.argmax(model.predict(img), axis=-1)
    probabilityValue = np.amax(predictions)

    # PREDICTION WITH HIGHEST PROBABILITY MUST BE ABOVE THRESHOLD TO BE DISPLAYED
    if probabilityValue > threshold:
        # PRINT PREDICTION AND CORRESPONDING PROBABILITY
        cv2.putText(imgOriginal, str(classIndex)+" "+ str(getclassname(classIndex)), (120, 35), font, 0.75, (0, 0, 225), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(imgOriginal, str(round(probabilityValue*100,2)) + "%", (180, 75), font, 0.75, (225, 0, 0), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(imgOriginal, str(translate(getclassname(classIndex))) ,(180, 115), font, 0.75, (0, 0, 225), 2, cv2.FONT_HERSHEY_SIMPLEX)
        speak(translate(getclassname(classIndex))[0])

    # IF HIGHEST PROBABILITY IS LOWER THAN THRESHOLD
    else:
        cv2.putText(imgOriginal, "Not sure", (120, 35), font, 0.75, (0, 0, 225), 2, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(imgOriginal, str(translate('Not sure')),(180, 115), font, 0.75, (0, 0, 225), 2, cv2.FONT_HERSHEY_SIMPLEX)
        speak(translate('Not sure')[0])

    # DISPLAY
    cv2.imshow("Result", imgOriginal)

    # PRESS 'q' TO STOP WEBCAM
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


