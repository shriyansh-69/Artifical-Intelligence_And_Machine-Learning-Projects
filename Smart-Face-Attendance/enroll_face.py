import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ResourceWarning)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import cv2
import pickle
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import os
import warnings


embedder = FaceNet()
detector = MTCNN()

os.makedirs("embeddings", exist_ok=True)

DB_path = "embeddings/face_db.pkl"

try :
    with open(DB_path,"rb") as f:
        face_db = pickle.load(f)
except:
    face_db = {}


name = input("Enter The Person:- ").strip()
embeddings = []

cap = cv2.VideoCapture(0)
count = 0
max_images = 5

print("[INFO] Press 'C' to capture | ESC to exit")

while True:
    ret,frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)


    if len(faces) == 1:
        x,y,w,h = faces[0]['box']
        x,y = abs(x),abs(y)

        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)


        cv2.putText(
            frame,
            f"captured: {count}/{max_images} | Press 'C' ",
            (20,40),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            0.8,
            (255,0,0),
            2
        )

        cv2.imshow("Enroll Face",frame)

        key = cv2.waitKey(1) & 0xFF
 
        if key == ord('c') and len(faces) == 1:
            face = rgb[y: y+h, x:x+w]
            face = cv2.resize(face,(160,160))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis = 0)


            embedding = embedder.model.predict(face, verbose=0)[0]
            embeddings.append(embedding)

            count += 1
            print(f"[INFO] Image {count} captured. Change pose.")

        if key == 27 or count >= max_images:
            break

cap.release()
cv2.destroyAllWindows()

face_db[name] =  embeddings
with open(DB_path, "wb") as f:
    pickle.dump(face_db, f)


print("[INFO] Enrollment completed successfully")
        
import sys
sys.exit(0)
