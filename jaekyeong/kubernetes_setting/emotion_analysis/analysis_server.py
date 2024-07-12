import cv2
import numpy as np
import dlib
from math import hypot
from keras.models import load_model

class analysis:
    # Initialise models
    def __init__(self):
        self.emotion_model = load_model('/app/e_model/emotion_recognition.h5')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "/app/e_model/shape_predictor_68_face_landmarks.dat")
        self.faceCascade = cv2.CascadeClassifier(
            '/app/e_model/haarcascade_frontalface_default.xml')
        self.x = 0
        self.y = 0
        self.emotion = 5
        self.size = 0
        self.frame_count = 0

# Function for finding midpoint of 2 points

    def midpoint(self, p1, p2):
        return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

# Function for eye size

    def get_blinking_ratio(self, frame, eye_points, facial_landmarks):
        left_point = (facial_landmarks.part(
            eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
        right_point = (facial_landmarks.part(
            eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
        center_top = self.midpoint(facial_landmarks.part(
            eye_points[1]), facial_landmarks.part(eye_points[2]))
        center_bottom = self.midpoint(facial_landmarks.part(
            eye_points[5]), facial_landmarks.part(eye_points[4]))
        hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)
        hor_line_lenght = hypot(
            (left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
        ver_line_lenght = hypot(
            (center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        ratio = ver_line_lenght / hor_line_lenght
        return ratio

# Gaze detection function
    def get_gaze_ratio(self, frame, eye_points, facial_landmarks, gray):
        # Gaze detection
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(
                                        eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(
                                        eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x,
                                     facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x,
                                     facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        up_side_threshold = threshold_eye[0: int(height/2), 0: int(width / 2)]
        up_side_white = cv2.countNonZero(up_side_threshold)
        down_side_threshold = threshold_eye[int(height/2): height, 0: width]
        down_side_white = cv2.countNonZero(down_side_threshold)
        lr_gaze_ratio = (left_side_white+10) / (right_side_white+10)
        ud_gaze_ratio = (up_side_white+10) / (down_side_white+10)
        return lr_gaze_ratio, ud_gaze_ratio

# Main function for analysis

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = self.detector(gray)
        benchmark = []
        for face in faces:
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            f = gray[x:x1, y:y1]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            landmarks = self.predictor(gray, face)
            left_point = (landmarks.part(36).x, landmarks.part(36).y)
            right_point = (landmarks.part(39).x, landmarks.part(39).y)
            center_top = self.midpoint(landmarks.part(37), landmarks.part(38))
            center_bottom = self.midpoint(
                landmarks.part(41), landmarks.part(40))
            hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
            ver_line = cv2.line(frame, center_top,
                                center_bottom, (0, 255, 0), 2)
            left_eye_ratio = self.get_blinking_ratio(frame,
                                                     [36, 37, 38, 39, 40, 41], landmarks)

            gaze_ratio_lr, gaze_ratio_ud = self.get_gaze_ratio(frame,
                                                               [36, 37, 38, 39, 40, 41], landmarks, gray)

            benchmark.append([gaze_ratio_lr, gaze_ratio_ud, left_eye_ratio])
            emotion = self.detect_emotion(gray)
            ci = self.gen_concentration_index()
            emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                        3: 'Sad', 4: 'Surprised', 5: 'Neutral'}
            self.x = gaze_ratio_lr
            self.y = gaze_ratio_ud
            self.size = left_eye_ratio
        return ci

# Function for detecting emotion

    def detect_emotion(self, gray):
        # Dictionary for emotion recognition model output and emotions
        emotions = {0: 'Angry', 1: 'Fear', 2: 'Happy',
                    3: 'Sad', 4: 'Surprised', 5: 'Neutral'}

        # Face detection takes approx 0.07 seconds
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(100, 100),)
        if len(faces) > 0:
            for x, y, width, height in faces:
                cropped_face = gray[y:y + height, x:x + width]
                test_image = cv2.resize(cropped_face, (48, 48))
                test_image = test_image.reshape([-1, 48, 48, 1])

                test_image = np.multiply(test_image, 1.0 / 255.0)

                probab = self.emotion_model.predict(test_image)[0] * 100

                label = np.argmax(probab)
                probab_predicted = int(probab[label])
                predicted_emotion = emotions[label]
                self.frame_count = 0
                self.emotion = label

        self.frame_count += 1

    def gen_concentration_index(self):
        weight = 0
        emotionweights = {0: 0.25, 1: 0.3, 2: 0.6,
                          3: 0.3, 4: 0.6, 5: 0.9}

        gaze_weights = 0

        if self.size < 0.2:
            gaze_weights = 0
        elif self.size > 0.2 and self.size < 0.3:
            gaze_weights = 1.5
        else:
            if self.x < 2 and self.x > 1:
                gaze_weights = 5
            else:
                gaze_weights = 2

        concentration_index = 0
        concentration_index = (
            emotionweights[self.emotion] * gaze_weights) / 4.5
        if concentration_index > 0.65:
            return 1
        elif concentration_index > 0.25 and concentration_index <= 0.65:
            return 1
        else:
            return 0
