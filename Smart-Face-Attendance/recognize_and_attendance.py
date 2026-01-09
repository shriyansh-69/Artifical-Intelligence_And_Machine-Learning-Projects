import cv2
import os 
import pickle
import csv
from datetime import datetime


cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
trainer_path = "trainer/trainer.yml"
labels_path = "trainer/labels.pkl"
attendance_dir =  "attendance"
attendance_file = os.path.join(attendance_dir,"attendance.csv")

confidence_threshold = 70

os.makedirs(attendance_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError("Haar Cascade file not found")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

with open(labels_path, "rb") as f:
    label_map = pickle.load(f)

labels = {k: v for k, v in label_map.items()}

def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    # Create file + header if not exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])

    # Check duplicate (same person, same date)
    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0 and row[0] == name and row[1] == date_str:
                return False   # ✅ explicit

    # Write attendance
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])

    print(f"[INFO] Attendance marked for {name}")
    return True

marked_people = set()   # ✅ track marked persons

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot access webcam")

print("[INFO] Face Recognition Started")
print("[INFO] Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (280, 280))

        label, confidence = recognizer.predict(face_img)

        if confidence < confidence_threshold:
            name = labels.get(label, "Unknown")

            if name not in marked_people:
                marked = mark_attendance(name)

                if marked:
                    marked_people.add(name)
                    print(f"[INFO] Attendance stored for {name}")

                    cv2.putText(
                        frame,
                        "Attendance Completed",
                        (70, 220),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 0),
                        3
                    )

        # draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        "Next person please | ESC to exit",
        (40, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Program terminated")
