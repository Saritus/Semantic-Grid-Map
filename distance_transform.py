from scipy import ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

inputImage='fr79_better_training_map.png'

im = Image.open(inputImage)
pix = im.load()
width, height = im.size

stepwidth = 1
stepheight = 1

overall = np.zeros((width, height), dtype=np.uint8)

for i in range(0, width, 1):
    for j in range(0, height, 1):
        overall[i][j] = 1 - np.array_equal(pix[i, j],[0,0,0])* 1

dt = ndimage.distance_transform_edt(overall)

#print overall
#print dt

plt.imshow(overall)
plt.show()

plt.imshow(dt)
plt.show()