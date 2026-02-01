import os 
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


import cv2
import pickle
import numpy as np
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet
import csv

embedder = FaceNet()
detector = MTCNN()


db_path = "embeddings/face_db.pkl"

with open(db_path,"rb") as f:
    face_db =  pickle.load(f)

os.makedirs("attendance", exist_ok= True)
ATT_FILE  =  "attendance/attendance.csv"

if not os.path.exists(ATT_FILE):
    with open(ATT_FILE, "w", newline="") as f :
        writer = csv.writer(f)
        writer.writerow(["Name","Date","Time"])


marked_today = set()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot Acess Webcam")
    exit


def already_marked_today(name,date):
    if not os.path.exists(ATT_FILE):
        return False
    
    with open(ATT_FILE,"r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == date:
                return True
            
    return False

print("[INFO] Face Recongition Started | Press ESC To Exit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    # MTCNN needs RGB, not GRAY
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    for face_data in faces:
        x, y, w, h = face_data["box"]
        x, y = abs(x), abs(y)

        face = rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        embedding = embedder.model.predict(face, verbose=0)[0]


        min_dist = 999
        identity = "Unknown"

        # Compare with database
        for name, db_embeddings in face_db.items():
            for db_emb in db_embeddings:
                dist = np.linalg.norm(embedding - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    identity = name

        if min_dist > 0.9:
            identity = "Unknown"

        # Mark Attendance (NO DUPLICATES)
        if identity != "Unknown" and identity not in marked_today:
            now = datetime.now()
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")

            if not already_marked_today(identity, date):
                with open(ATT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([identity, date, time])

                marked_today.add(identity)
                print(f"[INFO] Attendance marked for {identity}")


        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            identity,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (231, 34, 10),
            2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Program terminated")
