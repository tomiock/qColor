import matplotlib.image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def rgb2ycbcr(im):
    xform = np.array([	[.299, .587, .114],
                        [-.1687, -.3313, .5],
                        [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([	[1, 0, 1.402],
                        [1, -0.34414, -.71414],
                        [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

#img = Image.open('images/cat.jpeg')
#img = img.resize((64, 64), Image.ANTIALIAS)
#img = img.convert("1")
#img = matplotlib.image.pil_to_array(img)[:,:,:3]
#plt.imshow(img)
#plt.show()
#
#img = rgb2ycbcr(img)
#print(img)
#plt.imshow(img)
#plt.show()
#
#img = ycbcr2rgb(img)
#plt.imshow(img)
#plt.show()


