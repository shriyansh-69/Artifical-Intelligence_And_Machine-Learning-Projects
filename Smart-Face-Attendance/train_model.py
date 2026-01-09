import cv2 
import os
import numpy as np
import pickle 


dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"
label_path = "trainer/labels.pkl"

os.makedirs("trainer",exist_ok=True)

recognizer =  cv2.face.LBPHFaceRecognizer_create()

face_sample = []
face_label = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path,person_name)

    if not os.path.isdir(person_dir):
        continue

    label_map[current_label] = person_name

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        if img_path.endswith(".jpg"):
            gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

            if gray_img is None:
                continue

            face_sample.append(gray_img)
            face_label.append(current_label)

    current_label += 1


recognizer.train(face_sample, np.array(face_label))


recognizer.save(trainer_path)

with open(label_path, "wb") as f:
    pickle.dump(label_map, f)

print("[INFO] Training completed successfully!")
print(f"[INFO] Model saved at: {trainer_path}")
print(f"[INFO] Labels saved at: {label_path}")
 