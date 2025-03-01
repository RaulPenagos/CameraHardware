import cv2
import numpy as np
import imutils

def main():
    cap = cv2.VideoCapture(1)

    cap.set(3, 640)
    cap.set(4, 480)

    while True:

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # to find contours

        lower_blue = np.array([90, 60, 0])
        upper_blue = np.array([121, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)


        for c in cnts:

            area = cv2.contourArea(c)

            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # to find centroids, we look for the moments
            M = cv2.moments(c)

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "centroid", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            cv2.imshow('frame', frame)

        print('area is ...', area)
        print('centroid is ...', cx, cy)

        k = cv2.waitKey(1000)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()