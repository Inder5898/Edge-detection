import cv2,os,glob,sklearn,argparse,imutils,pywt,pywt.data 
import numpy as np
from matplotlib import pyplot as plt 
from math import log10, sqrt
from sklearn import metrics
from skimage.measure import compare_ssim



def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
#READ IMAGES
img_dir = r"C:/Users/Hp/Desktop/new paper/original_img" 
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1,1) 
    data.append(img) 

wave=[]
for f1 in files: 
    img = cv2.imread(f1,0) 
    wave.append(img) 
#APPLY 
img_dir_c = r"C:/Users/Hp/Desktop/new paper/generated_img"
p=f'canny_img.jpg'
p=((img_dir_c+'/')+p)
cannylist=[]

p1='sobel_8u.jpg'
p1=((img_dir_c+'/')+p1)

p2='sobelx8u.jpg'
p2=((img_dir_c+'/')+p2)
sx8u=[]
s8u=[]

mae=[]
psnr=[]
ssi=[]

mae_c8=[]
psnr_c8=[]
ssi_c8=[]

mae_c64=[]
psnr_c64=[]
ssi_c64=[]

i=0
for f1 in data:
    #Canny APPLIED---------------------------------------------
    edges_Canny = cv2.cv2.Canny(f1,100,200)

    #SOBEL APPLIED---------------------------------------------
    sobelx8u = cv2.Sobel(f1,cv2.CV_8U,1,0,ksize=1)
    
    #SOBEL64f APPLIED------------------------------------------
    sobelx64f = cv2.Sobel(f1,cv2.CV_64F,1,0,ksize=1)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    sobel_8u = cv2.Sobel(sobel_8u,cv2.CV_8U,1,0,ksize=1)

    #append to list -------------------------------------------
    cannylist.append(edges_Canny)
    sx8u.append(sobelx8u)
    s8u.append(sobel_8u)
    
    #SAVE_IMAGE------------------------------------------------
    cv2.imwrite(p,cannylist[i])
    cv2.imwrite(p1,s8u[i])
    cv2.imwrite(p2,sx8u[i])
    
    #READ_IMAGE------------------------------------------------
    cannylist_img = cv2.imread(r"C:/Users/Hp/Desktop/new paper/generated_img/canny_img.jpg",0) 
    s8u_img = cv2.imread(r"C:/Users/Hp/Desktop/new paper/generated_img/sobel_8u.jpg",0) 
    sx8u_img = cv2.imread(r"C:/Users/Hp/Desktop/new paper/generated_img/sobelx8u.jpg",0) 

    '''#3D TO 2D------------------------------------------------- above change 0 to 1
    a1,a2,a3=cannylist_img.shape
    cannylist_img=cannylist_img.reshape(a1,(a2*3))
    
    b1,b2,b3=s8u_img.shape
    s8u_img=s8u_img.reshape(b1,(b2*3))
    
    b1,b2,b3=sx8u_img.shape
    sx8u_img=sx8u_img.reshape(b1,(b2*3))'''

    #Plot----------------------------------------------------
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1,4,1),plt.imshow(f1,cmap = 'binary')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,4,2),plt.imshow(edges_Canny,cmap = 'gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,4,3),plt.imshow(sobel_8u,cmap = 'gray') 
    plt.title('Sobel 64f'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(1,4,4),plt.imshow(sobelx8u,cmap = 'gray')
    plt.title('Sobel (normal)'), plt.xticks([]), plt.yticks([])
    plt.show()


    #CALCULATE ERRORS------------------------------------------
    
    
    # canny sobel64f
    value = PSNR(s8u_img, cannylist_img)
    psnr_c64.append(value)
    mae_c64.append(metrics.mean_absolute_error(s8u_img,cannylist_img))
    (score, diff) = compare_ssim(s8u_img, cannylist_img, full=True)
    diff = (diff * 255).astype("uint8")
    ssi_c64.append(score*100)
    
    
    
    #canny sobel
    value = PSNR(sx8u_img, cannylist_img)
    psnr_c8.append(value)
    mae_c8.append(metrics.mean_absolute_error(sx8u_img,cannylist_img))
    (score, diff) = compare_ssim(sx8u_img, cannylist_img, full=True)
    diff = (diff * 255).astype("uint8")
    ssi_c8.append(score*100)
    
    
    
    #sobel sobel
    value = PSNR(s8u_img, sx8u_img)
    psnr.append(value)
    mae.append(metrics.mean_absolute_error(s8u_img,sx8u_img))
    (score, diff) = compare_ssim(s8u_img,sx8u_img, full=True)
    diff = (diff * 255).astype("uint8")
    ssi.append(score*100)
    
    
    #Plot Bar Graph---------------------------------------------------
    
    langs = ['Canny-Sobel 64f', 'Canny-Sobel(normal)', 'Sobel-Sobel64f']
    students = [mae_c64[i],mae_c8[i],mae[i]]
    plt.bar(langs,students)
    plt.title("Mean Absolute Error")
    plt.ylabel("MAE Values")
    plt.show()  
    
    
    langs = ['Canny-Sobel 64f', 'Canny-Sobel(normal)', 'Sobel-Sobel64f']
    students = [psnr_c64[i],psnr_c8[i],psnr[i]]
    plt.bar(langs,students)
    plt.title("PSNR")
    plt.ylabel("PSNR Values")
    plt.show()  
    
    
    langs = ['Canny-Sobel 64f', 'Canny-Sobel(normal)', 'Sobel-Sobel64f']
    students = [ssi_c64[i],ssi_c8[i],ssi[i]]
    plt.bar(langs,students)
    plt.title("Structure Similarity Index")
    plt.ylabel("SSI Values")
    plt.show()  
     
    
    #Wavelet Domain-----------------------------------------------------

    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(wave[i], 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    plt.title("Wavelet on Original Image")
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

    
    #ON CANNY
    coeffs2 = pywt.dwt2(cannylist_img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    plt.title("Wavelet on Canny Image")
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
   
    fig.tight_layout()
    plt.show()
    
    
    #ON SOBEL64f
    coeffs2 = pywt.dwt2(s8u_img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    plt.title("Wavelet on SOBEL64f Image")
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
   
    fig.tight_layout()
    plt.show()


    #ON SOBEL
    coeffs2 = pywt.dwt2(sx8u_img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    plt.title("Wavelet on SOBEL(normal) Image")
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
   
    fig.tight_layout()
    plt.show()
    
    
    i+=1