import random 
import pickle
import json
import numpy as np
import pandas as pd
import nltk 
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

path = r"C:\Users\shriyansh\OneDrive\Desktop\Machine_Learning_Project's\AI-Customer-Support-Chatbot\intent.json"

with open(path, "r", encoding= "utf-8") as f:
    intents = json.loads(f.read())


words = []
classes = []
documents = []
ignoreletters = ["!","?",".",","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        WordList = nltk.word_tokenize(pattern)
        words.extend(WordList)
        documents.append(WordList,intent["tag"])
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in  ignoreletters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordpatterns = document[0]
    wordpatterns = [lemmatizer.lemmatize(word.lower())  for word in wordpatterns]
    for word in words:
        bag.append(1) if word in wordpatterns else bag.append(0)


    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

train_x = training[:,: len(words)]
train_y = training[:,len(words) :]


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,input_shape = (len(train_x[0]),), activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation = "relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]),activation = "Softmax"))


sgd = tf.keras.optimizers.SGD(learning_rate = 0.01, momentum = 0.09, nesterov = True)


model.complie(loss = "categorical_crossentropy",optimizers = sgd, metrics = ["accuracy"])


hist = model.fit(
    np.array(train_x),np.array(train_y), epochs = 240, batch_size = 5,verbose = 1
)



model.save("Chatbot_Model.h5",hist)
if hist is not None:
    print("Model Downloaded Successfully")

