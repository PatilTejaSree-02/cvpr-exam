  1.SMOOTH AN IMAGE USING GAUSSIAN AND MEDIAN FILTERS

import cv2

img = cv2.imread("img.jpg")

gaussian = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
median = cv2.medianBlur(img, 5)

cv2.imshow("Original Image", img)
cv2.imshow("Gaussian Filtered Image", gaussian)
cv2.imshow("Median Filtered Image", median)

cv2.waitKey(0)
cv2.destroyAllWindows()


2.MORPHOLOGICAL OPERATIONS-EROSION,DILATION,OPENING,CLOSING

import cv2
import numpy as np

img = cv2.imread("img.png", cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("Original", img)
cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()










3.EDGE DETECTION-LAPLACIAN,SOBEL,PREWITT,ROBERT,1-D GRADIENT


import cv2
import numpy as np

img = cv2.imread("flower.jpg", cv2.IMREAD_GRAYSCALE)

# ------------------ SOBEL ------------------
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# ------------------ PREWITT ------------------
prewitt_x = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
prewitt_y = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])

p_x = cv2.filter2D(img, cv2.CV_64F, prewitt_x)
p_y = cv2.filter2D(img, cv2.CV_64F, prewitt_y)
prewitt = cv2.magnitude(p_x, p_y)

# ------------------ ROBERTS ------------------
roberts_x = np.array([[1,0],[0,-1]])
roberts_y = np.array([[0,1],[-1,0]])

r_x = cv2.filter2D(img, cv2.CV_64F, roberts_x)
r_y = cv2.filter2D(img, cv2.CV_64F, roberts_y)
roberts = cv2.magnitude(r_x, r_y)

# ------------------ 1D GRADIENT ------------------
gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1], [1]])

g_x = cv2.filter2D(img, cv2.CV_64F, gx_kernel)
g_y = cv2.filter2D(img, cv2.CV_64F, gy_kernel)
grad1d = cv2.magnitude(g_x, g_y)

# ------------------ LAPLACIAN ------------------
laplacian = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow("Original", img)
cv2.imshow("Sobel", sobel)
cv2.imshow("Prewitt", prewitt)
cv2.imshow("Roberts", roberts)
cv2.imshow("1D Gradient", grad1d)
cv2.imshow("Laplacian", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
4. RGB TO CMY AND HSV

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv

img = cv2.imread("flower.jpg")

# Convert to 0–1 range (for HSV)
if img.dtype == np.uint8:
    img = img / 255.0

cmy = 1 - img
hsv = rgb_to_hsv(img)

plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cmy)
plt.title("CMY")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(hsv)
plt.title("HSV")
plt.axis("off")

plt.tight_layout()
plt.show()














5.K MEANS CLUSTERING
import cv2  
import numpy as np   
import matplotlib.pyplot as plt 

img = cv2.imread("shinchan.png")  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

Z = np.float32(img.reshape((-1, 3)))  

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  

plt.figure(figsize=(12, 8))

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("org")
plt.axis("off") 

for i, k in enumerate([2,4,6,8]):  
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    centers = np.uint8(centers)  
    res = centers[labels.flatten()] 
    res = res.reshape(img.shape)  

    plt.subplot(2,3,i+2)
    plt.imshow(res)
    plt.title(f"k={k}")
    plt.axis("off")  

plt.tight_layout()
plt.show()



6.NAIVE BAYES 

import cv2, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

paths = ["pic.jpg", "pic.jpg", "pic1.jpg", "pic.jpg"]
labels = np.array(["cat", "dog", "cat", "dog"])

data = [cv2.resize(cv2.imread(p, 0), (64,64)).flatten() for p in paths]

model = GaussianNB().fit(data, labels)
pred = model.predict(data)

print("Pred:", pred)
print("Acc:", accuracy_score(labels, pred))
OUTPUT 
Pred: ['dog' 'dog' 'cat' 'dog']
Acc: 0.75

7.HOUGH TRANSFORM

import cv2
import numpy as np

img = cv2.imread("input.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
if lines is not None:
    for line in lines[:5]:
        rho, theta = line[0]   # unpacking ρ & θ correctly

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
    print("No lines detected")

cv2.imshow("detected lines", img)
cv2.waitKey(0)
cv2.destroyAllWindows()








8. PCA
import cv2,
 numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("shinchan.png"), cv2.COLOR_BGR2RGB) gray = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))  
mean, eig = cv2.PCACompute(gray, mean=None, maxComponents=50)  recon = cv2.PCABackProject(cv2.PCAProject(gray, mean, eig), mean, eig) 

plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(recon, cmap='gray')
plt.title("PCA Reduced")
plt.axis("off")

plt.tight_layout()
plt.show()


9. BIT PLANE CODING

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("image.jpg", 0)

# Bit-plane extraction (MSB - highest bit plane)
bit_plane = (img >> 7) & 1
bit_plane = bit_plane * 255

plt.imshow(bit_plane, cmap='gray')
plt.title("Higher Bit Plane (MSB)")
plt.axis("off")
plt.show()








10. HANDWRITTEN RECOGNITION

10.import cv2, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier

X,y = fetch_openml('mnist_784', return_X_y=True)
model = KNeighborsClassifier(3).fit(X/255.0, y.astype(int))

img = cv2.imread("digit.png",0)

if img is None:
    print("Image not found"); exit()

img = cv2.resize(img,(28,28))

if np.mean(img) > 127:
    img = 255 - img  
_, img = cv2.threshold(img,100,255,cv2.THRESH_BINARY)


img = img/255.0

pred = model.predict(img.reshape(1,784))[0]

cv2.imshow("Processed",img)
print("Predicted:",pred)
cv2.waitKey(0)

