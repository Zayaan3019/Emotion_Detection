import cv2
import numpy as np
from tensorflow.keras.models import load_model

def live_emotion_detection(model_path, target_size=(48, 48)):
    """
    Detect emotions in real-time using a webcam.
    """
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
            face = np.expand_dims(face, axis=-1) / 255.0
            face = np.expand_dims(face, axis=0)

            emotion = model.predict(face)
            emotion_label = np.argmax(emotion)
            cv2.putText(frame, str(emotion_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
