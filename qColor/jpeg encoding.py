import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

B=8 # blocksize (In Jpeg the
img1 = cv2.imread("images/cat.jpg", cv2.CV_LOAD_IMAGE_UNCHANGED)
h,w=np.array(img1.shape[:2])/B * B
img1=img1[:h,:w]
#Convert BGR to RGB
img2=np.zeros(img1.shape,np.uint8)
img2[:,:,0]=img1[:,:,2]
img2[:,:,1]=img1[:,:,1]
img2[:,:,2]=img1[:,:,0]
plt.imshow(img2)

