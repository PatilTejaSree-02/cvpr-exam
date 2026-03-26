

1.import cv2

img = cv2.imread("img.jpg")

g = cv2.GaussianBlur(img,(5,5),0)
m = cv2.medianBlur(img,5)

cv2.imshow("G", g)
cv2.imshow("M", m)
cv2.waitKey(0)



2.import cv2, numpy as np

img = cv2.imread("img.png",0)
k = np.ones((5,5),np.uint8)

cv2.imshow("E", cv2.erode(img,k))
cv2.imshow("D", cv2.dilate(img,k))
cv2.imshow("O", cv2.morphologyEx(img,cv2.MORPH_OPEN,k))
cv2.imshow("C", cv2.morphologyEx(img,cv2.MORPH_CLOSE,k))

cv2.waitKey(0)



3.import cv2, numpy as np

i = cv2.imread("img.jpeg",0)

if i is None:
    print("Image not found")
    exit()

s = cv2.magnitude(cv2.Sobel(i,cv2.CV_64F,1,0,3),
                  cv2.Sobel(i,cv2.CV_64F,0,1,3))

p = cv2.magnitude(cv2.filter2D(i,cv2.CV_64F,np.array([[-1,0,1]]*3)),
                  cv2.filter2D(i,cv2.CV_64F,np.array([[1,1,1],[0,0,0],[-1,-1,-1]])))

r = cv2.magnitude(cv2.filter2D(i,cv2.CV_64F,np.array([[1,0],[0,-1]])),
                  cv2.filter2D(i,cv2.CV_64F,np.array([[0,1],[-1,0]])))

g = cv2.magnitude(cv2.filter2D(i,cv2.CV_64F,np.array([[-1,1]])),
                  cv2.filter2D(i,cv2.CV_64F,np.array([[-1],[1]])))

l = cv2.Laplacian(i,cv2.CV_64F)

cv2.imshow("S",s); cv2.imshow("P",p)
cv2.imshow("R",r); cv2.imshow("G",g)
cv2.imshow("L",l)

cv2.waitKey(0)




4.import cv2

img = cv2.imread("flower.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cmy = 255 - img

cv2.imshow(“Original”,img)
cv2.imshow("HSV", hsv)
cv2.imshow("CMY", cmy)
cv2.waitKey(0)



5.
import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.cvtColor(cv2.imread("img.jpeg"), cv2.COLOR_BGR2RGB)
Z = np.float32(img.reshape((-1,3)))

plt.subplot(2,3,1)
plt.imshow(img); plt.title("org"); plt.axis("off")

for i,k in enumerate([2,4,6,8]):
    _,l,c = cv2.kmeans(Z,k,None,(1,10,1),10,0)
    res = c[l.flatten()].reshape(img.shape).astype('uint8')

    plt.subplot(2,3,i+2)
    plt.imshow(res); plt.title(f"k={k}"); plt.axis("off")

plt.show()




6.import cv2, numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# images + labels
paths = ["pic.jpeg","pic1.jpeg",'pic.jpeg','pic.jpeg']
labels = ["cat", "dog", "cat", "dog"]

# convert images → data
data = [cv2.resize(cv2.imread(p,0),(50,50)).flatten() for p in paths]

# train + predict
m = GaussianNB().fit(data,labels)
pred = m.predict(data)

# output
print("Pred:",pred)
print("Acc:",accuracy_score(labels,pred))



7.import cv2, numpy as np

img = cv2.imread("input.jpg")
g = cv2.cvtColor(img, 0)

e = cv2.Canny(g,50,150)
lines = cv2.HoughLines(e,1,np.pi/180,120)

if lines is not None:
    print("Lines detected")
    
    for l in lines[:5]:
        r,t = l[0]
        a,b = np.cos(t), np.sin(t)
        x,y = a*r, b*r
        cv2.line(img,(int(x+1000*-b),int(y+1000*a)),
                      (int(x-1000*-b),int(y-1000*a)),(0,0,255),2)
else:
    print("No lines detected")

cv2.imshow("Hough", img)
cv2.waitKey(0)



8.import cv2, numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img.jpeg")

g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')

m,e = cv2.PCACompute(g,None,50)
p = cv2.PCAProject(g,m,e)
r = cv2.PCABackProject(p,m,e)

plt.subplot(1,2,1)
plt.imshow(g,cmap='gray'); plt.title("Original"); plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(r,cmap='gray'); plt.title("PCA"); plt.axis("off")

plt.show()



9.import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img.jpeg",0)

bit = (img>>7)&1

plt.imshow(bit,'gray')
plt.title("MSB")
plt.axis("off")
plt.show()



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


