import cv2  # pyright: ignore[reportMissingImports]
import time
import os

dataset_path = "dataset"
cascade_path = "haarcascade/haarcascade_frontalface_default.xml"
image_count = 20
face_size = (250,250)

# Classifier Loaded

face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError("Haar cascade file not found!")

# Person's Input Name 

person_name = input("Enter The Person Name: ").strip()

person_dir = os.path.join(dataset_path,person_name)

if not os.path.exists(person_dir):
    os.makedirs(person_dir)


#  Add Timer After Name Input
print("\n[INFO] Face capture will start in 5 seconds...")
for i in range(5, 0, -1):
    print(f"starting in {i}")
    time.sleep(1)

print("[INFO] Camera started. Please look at the camera.")


# Start Webcam 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot Access Webcam")

print("[INFO] Starting Face Capture")
print("[INFO] Look at the camera and move your head slightly")

count = 0


while True:
    ret,frame = cap.read()
    if not ret:
        print("[ERROR] Failed To Grab Frame")

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )


    # Instruction text
    cv2.putText(
        frame,
        f"Images: {count}/{image_count} | Press 'C' to Capture | ESC to Exit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # Proceed ONLY if exactly one face detected
    if len(faces) == 1:
        (x, y, w, h) = faces[0]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            count += 1

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img,face_size )

            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)

            print(f"[INFO] Image {count} captured. Change pose.")
            time.sleep(0.2)  # prevents double capture

    else:
        cv2.putText(
            frame,
            "Ensure ONLY ONE face is visible",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Capture", frame)

    # Exit conditions
    if cv2.waitKey(1) & 0xFF == 27:
        break

    if count >= image_count:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("[INFO] Face capture completed successfully!")
print(f"[INFO] {count} images saved in '{person_dir}'")