import cv2
import numpy as np
import pickle

rect_width, rect_height = 108, 49

cap = cv2.VideoCapture('carParkingVideo.mp4')

with open('CarParkPosition', 'rb') as f:
    posList = pickle.load(f)
frame_counter = 0


def check_free_space(img_projection):
    space_count = 0
    for pos in posList:
        x, y = pos
        crop = img_projection[y:y + rect_height, x:x + rect_width]
        count = cv2.countNonZero(crop)
        if count < 900:
            space_count += 1
            color = (0, 0, 255)
            thick = 5
        else:
            color = (255, 0, 0)
            thick = 2

        cv2.rectangle(img, pos, (x + rect_width, y + rect_height), color, thick)
    cv2.rectangle(img, (45, 30), (250, 75), (180, 0, 180), -1)
    cv2.putText(img, f'Free Space: {space_count}/{len(posList)}', (50, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)


while True:
    _, img = cap.read()
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    Threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    blur = cv2.medianBlur(Threshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(blur, kernel, iterations=1)
    check_free_space(dilate)

    cv2.imshow("Car Parking Image ---> ", img)
    cv2.waitKey(5)
