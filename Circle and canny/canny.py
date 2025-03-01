import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.image import imread


def search_border4():

    # fig = cv2.imread('./biblia.jpg')[: , :, 0]
    fig = imread('./biblia.jpg')[: , :, 0]


    bordeCanny = cv2.Canny(fig, 100, 200)

    cv2.imshow('Original', fig)
    cv2.imshow('Canny', bordeCanny)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    x, y = np.array([]), np.array([])
    for i,f in enumerate(bordeCanny):
        for j,px in enumerate(f):
            if px != 0:
                x = np.append(x, j)
                y = np.append(y, np.abs(i-800))

    plt.plot(x,y , 'or')

    # plt.gca().invert_yaxis()
    plt.show()




def canny_trackbar_tests():

    def callback(x):
        print(x)

    img = cv2.imread('fiducial_test.png', 0) #read image as grayscal


    canny = cv2.Canny(img, 100, 200) 

    cv2.namedWindow('image') # make a window with name 'image'
    cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
    cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image

    while(1):
        numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: #escape key
            break
        l = cv2.getTrackbarPos('L', 'image')
        u = cv2.getTrackbarPos('U', 'image')

        canny = cv2.Canny(img, l, u)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(): 
    # canny_trackbar_tests()
    search_border4()







if __name__ == "__main__":
    main()