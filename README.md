# Classify-Human-Expressions
Train a CNN model using ck++ data and test using test samples
####################################################
Feature extraction and data preparation
#####################################################
Step 1: Load prepare_data.ipynb
Step 2: Download Emotion_labels.zip and extended-cohn-kanade-images from ck++ website : http://www.consortium.ri.cmu.edu/ckagree/
Step 3: Extract the data in one folder
Step 4: Create two foders "final_dataset" and "stored_Set"
Step 5: With in each folder create subfolders with names : anger, contempt, disgust, fear, happy, neutral, sadness and surprise.
Step 6: Run prepare_data.ipynb

Results:

The final images will be stored in the foder: stored_Set
train_images.npy, train_labels.npy, test_images.npy, test_labels.npy files which can be used to train and test Machine learning models will be 
created in the same folder.



######################################################
Run main CNN model code
#####################################################
Step 1: Load .npy files created before.
Step 2: Load CNN_traintest.ipynb file.
Step 3: Run CNN_traintest.ipynb
