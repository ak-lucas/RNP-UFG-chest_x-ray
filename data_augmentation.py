import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt
from skimage.transform import rotate, rescale
import cv2
import numpy as np
import scipy.misc
import sys
from skimage import exposure

folder_n = sys.argv[1] 
for filename in os.listdir(folder_n):
  img = mpimg.imread(os.path.join(folder_n, filename))
  if img is not None:
    if img.shape[-1] == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rotate(img, angle=np.random.randint(-3,3), mode='constant')[50:-50,50:-50]
    img = rescale(img, scale=1.1, mode='constant')
    img = exposure.equalize_hist(img)
    scipy.misc.imsave(folder_n + 'aug_' + filename, img)
  else:
    print "algo deu errado"
    exit(1)

folder_n = sys.argv[2] 
for filename in os.listdir(folder_n):
  img = mpimg.imread(os.path.join(folder_n, filename))
  if img is not None:
    if img.shape[-1] == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = rotate(img, angle=np.random.randint(-3,3), mode='constant')[50:-50,50:-50]
    img = rescale(img, scale=1.1, mode='constant')
    img = exposure.equalize_hist(img)
    scipy.misc.imsave(folder_n + 'aug_' + filename, img)
  else:
    print "algo deu errado"
    exit(1)
