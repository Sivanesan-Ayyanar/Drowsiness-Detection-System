import cv2
import dlib
from scipy.spatial import distance
import pygame
import time


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


def play_alarm():
    pygame.mixer.init()
    pygame.mixer.music.load("Alarm.wav")
    pygame.mixer.music.play()


def reset_alarm():
    pygame.mixer.music.stop()


def eyes_closed(ear):
    global start_time
    global closed_start_time
    global eyes_closed_duration

    if ear < 0.23:
        if not closed_start_time:
            closed_start_time = time.time()
        eyes_closed_duration = time.time() - closed_start_time
        if eyes_closed_duration >= 3:
            return True
    else:
        closed_start_time = None
    return False


pygame.init()

cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

alarm_played = False
start_time = time.time()
closed_start_time = None
eyes_closed_duration = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 30), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 30), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)

        if eyes_closed(EAR):
            if not alarm_played:
                play_alarm()
                alarm_played = True
                print("Eyes Closed - Alarm Played")
            cv2.putText(frame, "Eyes Closed!!!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            reset_alarm()
            alarm_played = False

    cv2.imshow("Drive Careful", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
