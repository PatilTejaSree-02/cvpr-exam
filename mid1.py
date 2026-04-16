1.
2. Write programs for the following
a) Loading and displaying an image.
import cv2
img = cv2.imread("image.jpg")
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
b) Reading and writing video files.
import cv2
cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(25) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
c) Image enhancement. 
import cv2
import numpy as np
img = cv2.imread("image.jpg", 0)

linear = img * 2
log = np.uint8(40 * np.log(1 + img))
power = np.uint8((img / 255) ** 0.5 * 255)
cv2.imshow("Linear", linear)
cv2.imshow("Log", log)
cv2.imshow("Power", power)
cv2.waitKey(0)
cv2.destroyAllWindows()


3. Perform the following operations on an image
   a) Resize 
import cv2
img = cv2.imread("image.jpg")
resized = cv2.resize(img, (200,200))
cv2.imshow("Resize", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
  b) Rotation   
import cv2
img = cv2.imread("image.jpg")
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
c) Flipping 
import cv2
img = cv2.imread("image.jpg")
cv2.imshow("Flip H", cv2.flip(img,1))
cv2.imshow("Flip V", cv2.flip(img,0))
cv2.waitKey(0)
cv2.destroyAllWindows()
d) Cropping
import cv2
img = cv2.imread("image.jpg")
crop = img[50:200, 50:200]
cv2.imshow("Crop", crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
4. Displaying an image with different Spatial resolutions and Intensity resolutions.
import cv2
img = cv2.imread("image.jpg", 0)
# Sampling (Spatial Resolution)
cv2.imshow("Resize", cv2.resize(img,(100,100)))
cv2.imshow("Slice", img[::5,::5])
# Quantization (Intensity Resolution)
cv2.imshow("L=2", (img//128)*128)
cv2.imshow("L=4", (img//64)*64)
cv2.imshow("L=8", (img//32)*32)
cv2.waitKey(0)
cv2.destroyAllWindows()
5. Apply Nearest Neighbour, Bilinear and Bicubic Interpolations on an image for increasing its size.
import cv2
img = cv2.imread("image.jpg")
cv2.imshow("NN", cv2.resize(img,None,fx=2,fy=2,interpolation=0))
cv2.imshow("BL", cv2.resize(img,None,fx=2,fy=2,interpolation=1))
cv2.imshow("BC", cv2.resize(img,None,fx=2,fy=2,interpolation=2))
cv2.waitKey(0)
cv2.destroyAllWindows()
6. Apply different Intensity level transformations on an image.
import cv2
import numpy as np
img = cv2.imread("image.jpg", 0)
contrast = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
intensity = (img//64)*64
bit7 = img & 128
cv2.imshow("Contrast", contrast)
cv2.imshow("Intensity", intensity)
cv2.imshow("Bit Plane", bit7)
cv2.waitKey(0)
cv2.destroyAllWindows()

7.Implement Histogram Calculation and Equalization for a given image
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("image.jpg", 0)
eq = cv2.equalizeHist(img)
cv2.imshow("Original Image", img)
cv2.imshow("Equalized Image", eq)
plt.figure()
plt.title("Original Histogram")
plt.plot(cv2.calcHist([img],[0],None,[256],[0,256]))
plt.figure()
plt.title("Equalized Histogram")
plt.plot(cv2.calcHist([eq],[0],None,[256],[0,256]))
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()




