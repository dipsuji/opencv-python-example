import cv2
import mediapipe as mp


def live_hand_video_capture():
    hands_sol = mp.solutions.hands
    hands = hands_sol.Hands()
    sol_draw = mp.solutions.drawing_utils
    fingers_coordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
    thumb_coordinate = (4, 3)
    video_capture = cv2.VideoCapture(0)

    while video_capture.isOpened():
        up_count = 0
        success, img = video_capture.read()
        converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(converted_image)
        hand_no = 0
        list_lim = []

        if results.multi_hand_landmarks:
            for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                list_lim.append((cx, cy))
            for hand_in_frame in results.multi_hand_landmarks:
                sol_draw.draw_landmarks(img, hand_in_frame, hands_sol.HAND_CONNECTIONS)
            for point in list_lim:
                cv2.circle(img, point, 5, (0, 75, 255), cv2.FILLED)
            for coordinate in fingers_coordinates:
                if list_lim[coordinate[0]][1] < list_lim[coordinate[1]][1]:
                    up_count += 1
            if list_lim[thumb_coordinate[0]][0] > list_lim[thumb_coordinate[1]][0]:
                up_count += 1
            cv2.putText(img, str(up_count), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (75, 0, 255), 10)

        cv2.imshow("Hand finger Tracking", img)

        if cv2.waitKey(1) == 113:
            break


live_hand_video_capture()
