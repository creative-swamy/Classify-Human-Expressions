##########################################################################################################
#########Copy data from ck+ to individual emotion labeled folders#######################################
print("Preparing images ...")
import glob
from shutil import copyfile
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

participants = glob.glob(r"Emotion_labels\Emotion\*")

for x in participants: #All the folders
    part = "%s" %x[-4:]
    for sessions in glob.glob("%s\*" %x): #All the files under a folder
        for files in glob.glob("%s\*" %sessions):
            current_session = sessions[-3:]
            file = open(files, 'r')
            emotion = int(float(file.readline())) 
            #neutral image location from ck+
            sourcefile_emotion = glob.glob(r"extended-cohn-kanade-images\cohn-kanade-images\%s\%s\*" %(part, current_session))[-1]
            #emotion image location from ck+
            sourcefile_neutral = glob.glob(r"C:extended-cohn-kanade-images\cohn-kanade-images\%s\%s\*" %(part, current_session))[0]
            #where to save the data: neutral and emotion
            dest_neut = r"stored_Set\neutral\%s" %sourcefile_neutral[-21:]
            dest_emot = r"stored_Set\%s\%s" %(emotions[emotion], sourcefile_emotion[-21:])
            copyfile(sourcefile_neutral, dest_neut)
            copyfile(sourcefile_emotion, dest_emot)
############################################################################################################
##########################Detect faces using cascade classifier and cut images to 350*350
import cv2

faceDet = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(r"haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier(r"haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier(r"haarcascade_frontalface_alt_tree.xml")

def detect_faces(emotion):
    files = glob.glob(r"stored_Set\%s\*" %emotion)
    filenumber = 0
    for f in files:
        frame = cv2.imread(f)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else :
            facefeatures= ""
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            #print("face found in file: %s" %f)
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite(r"final_dataset\%s\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass 
        filenumber += 1 
for emotion in emotions:
    detect_faces(emotion) 
#############################################################################################################
############################prepare test and training data
import numpy as np
import random
data = {}
def get_files(emotion):
    files = glob.glob(r"final_dataset\%s\*"%emotion)
    random.shuffle(files)
    training  = files[:int(len(files)*.80)]
    prediction = files[-int(len(files)*.20):]    
    return training, prediction
def makeset():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    
    for emotion in emotions:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray)
            training_labels.append(emotions.index(emotion))
        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))
    np.save(r'train_images.npy', [a for a in training_data])
    np.save(r'train_labels.npy', [a for a in training_labels])
    np.save(r'test_images.npy', [a for a in prediction_data])
    np.save(r'test_labels.npy', [a for a in prediction_labels])
makeset()
print("Data preparation done")
##########################################################################################################33