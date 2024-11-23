import cv2
import numpy as np
from keras.models import model_from_json, Sequential
from keras.saving import register_keras_serializable

register_keras_serializable()(Sequential)

class EmotionDetector:
    def __init__(self, model_json_path, model_weights_path, haarcascade_path):
        self.model = self.load_model(model_json_path, model_weights_path)
        self.face_cascade = cv2.CascadeClassifier(haarcascade_path)
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
    def load_model(self, model_json_path, model_weights_path):
        with open(model_json_path, "r") as json_file:
            model_json = json_file.read()
        # Pass the custom_objects parameter with the registered class
        model = model_from_json(model_json, custom_objects={'Sequential': Sequential})
        model.load_weights(model_weights_path)
        return model

    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def predict_emotion(self, image):
        img = self.extract_features(image)
        pred = self.model.predict(img)
        return self.labels[pred.argmax()]

    def start_webcam(self):
        webcam = cv2.VideoCapture(0)
        while True:
            ret, frame = webcam.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                emotion = self.predict_emotion(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            cv2.imshow("Emotion Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    model_json_path = "emotiondetection.json"
    model_weights_path = "emotiondetection.h5"
    
    detector = EmotionDetector(model_json_path, model_weights_path, haarcascade_path)
    detector.start_webcam()


