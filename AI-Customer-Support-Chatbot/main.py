import random 
import pickle
import json
import numpy as np
import pandas as pd
import nltk 
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

path = r"D:\Machine_Learning_Projects\AI-Customer-Support-Chatbot\intent.json"

with open(path, "r", encoding= "utf-8") as f:
    intent = json.loads(f.read())