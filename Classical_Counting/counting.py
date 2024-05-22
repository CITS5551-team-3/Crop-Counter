import cv2
import numpy as np


im = cv2.imread('Test_Images/NGRDI.jpg')
imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh  = cv2.threshold(imgrey, 10, 255, cv2.THRESH_BINARY)
kernel1= np.ones((4,4), np.uint8)
kernel2 = np.ones((3,3), np.uint8)

eroded = cv2.erode(thresh, kernel1, iterations = 2)
dilated = cv2.dilate(eroded, kernel2, iterations = 8)
#edged = cv2.Canny(imgrey, 30, 200)

cv2.imwrite('Test_Images/Dilated.jpg', dilated)
cv2.imwrite('Test_Images/Eroded.jpg', eroded)

combined = dilated

contours, heirachy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))
cv2.drawContours(combined, contours, -1, (225, 255, 0), 10)
cv2.imwrite('Test_Images/Contours.jpg', combined)
