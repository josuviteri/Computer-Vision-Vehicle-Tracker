import cv2 as cv


image = cv.imread("media/jupiter-moon-whole.tif")


cv.imshow("Og image", image, )
cv.waitKey(0)
cv.destroyAllWindows()