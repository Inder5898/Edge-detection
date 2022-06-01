#Carny Edge Detection
import cv2,sklearn
import numpy as np
from matplotlib import pyplot as plt

img_Canny = cv2.cv2.imread(r'C:\Users\surps\OneDrive\Desktop\documents\project/2.jpg',1)
edges_Canny = cv2.cv2.Canny(img_Canny,100,200)

plt.subplot(121),plt.imshow(img_Canny,cmap = 'binary')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges_Canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
#from sklearn.metrics import mean_absolute_error

#mean_absolute_error(img_Canny, edges_Canny)

#Sobel Edge Detection
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('2.jpg',0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()


#Wavelet Domain
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

# Load image
import cv2,sklearn
import numpy as np
from matplotlib import pyplot as plt
#original = pywt.data.camera()
original = cv2.cv2.imread(r'C:\Users\surps\OneDrive\Desktop\documents\project/2.jpg',0)
#edges_Canny = cv2.cv2.Canny(img_Canny,100,200)

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
plt.subplot(122),plt.imshow(edges_Canny,cmap = 'gray')
plt.title('Edge Image using wavelet domain'), plt.xticks([]), plt.yticks([])

plt.show()
