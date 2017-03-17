# -*- coding: cp1252 -*-
# imports
from PIL import Image
import glob, os
import numpy as np
import sys
#import theano
import pickle
from scipy import ndimage
import matplotlib.pyplot as plt

# set global variables
inputImage='fr79_better_training_map.png'
outputDir='walls_py'
width=50
height=50
step_width=4
step_height=4
innerstep=5

# create folder if necessary
if not os.path.exists(outputDir):
    os.makedirs(outputDir)

# open image and get size
im = Image.open(inputImage)
pix = im.load()
print (pix[0, 0])
#im.show() to show input image
im_width, im_height = im.size
print ("width = " + str(im_width) + " _ height = " + str(im_height))

#                        Schwarz    Weiß             Gelb           Braun        Blau
#color_table = np.array([[0, 0, 0], [255, 255, 255], [255, 207, 0], [127, 0, 0], [0, 223, 255]])
color_table = []

def get_color_code(pixel):
    for index in range(len(color_table)):
        if np.array_equal(pixel, color_table[index]):
            return index
    color_table.append(pixel)
    sys.stderr.write("color_table" + str(color_table) + "\n")
    #print (color_table)
    #print ((color_table)-1)
    return len(color_table)-1

def create_pkl_file(train_x, train_y, test_x, test_y, filename='autoencoder.pkl'):
    train_set_x = np.array(train_x, dtype='float64')
    train_set_y = np.array(train_y, dtype='float64')
    train_set = train_set_x, train_set_y
    test_set_x = np.array(test_x, dtype='float64')
    test_set_y = np.array(test_y, dtype='float64')
    test_set = test_set_x, test_set_y
    data_set = [train_set, train_set, test_set, np.array(color_table, dtype='uint8')]
    f = open(filename, 'wb')
    pickle.dump(data_set, f, protocol=2)
    f.close()

def create_dt(pixelarray):
    overall = np.zeros((im_width, im_height), dtype=np.uint8) # create empty array
    for i in range(0, im_width, 1):
        for j in range(0, im_height, 1):
            overall[i][j] = 1 - np.array_equal(pixelarray[i, j],[0,0,0])
    return ndimage.distance_transform_edt(overall) # DT

# Matlab
#for i=1:step_width:(size(mapImage, 2))
#    for j=1:step_height:(size(mapImage, 1))
#        if ((size(mapImage,2)-i) < width) || ...
#                ((size(mapImage,1)-j) < height)
#            continue
#        end
#        tmp=mapImage( ...
#                j:innerstep:min(j+height,size(mapImage,1)), ...
#                i:innerstep:min(i+width,size(mapImage,2)));
#        imshow(tmp)
#        drawnow()
#        fileName=fullfile(outputDir,['localMap_',num2str(i,'%04u'),'_', num2str(j,'%04u'),'.png'])
#        imwrite(tmp,fileName)
#    end
#end

# Python
#labels = np.zeros((im_width, im_height), dtype=np.uint8)
train_set_x = []
train_set_y = []
test_set_x = []
test_set_y = []
for x in range(0, im_width, step_width):
    if (im_width-x) < width:
        break
    for y in range(0, im_height, step_height):
        if (im_height-y) < height:
            break
        localMapBW = np.zeros((height/innerstep+1, width/innerstep+1), dtype=np.uint8)
        localMapCol = np.zeros((height/innerstep+1, width/innerstep+1, 3), dtype=np.uint8)
        for indexi, i in enumerate(range(x, min(x+width+1, im_width), innerstep)):
            for indexj, j in enumerate(range(y, min(y+height+1, im_height), innerstep)):
                localMapBW[indexj, indexi] = np.array_equal(pix[i, j],[0,0,0])* 255
                localMapCol[indexj, indexi] = pix[i, j]
        img = Image.fromarray(localMapBW, 'L')
        img = img.convert('1')
        img.save(outputDir + '/' + "localMap_" + str(x+1).zfill(4) + "_" + str(y+1).zfill(4) + ".png")
        #img = Image.fromarray(localMapCol, 'RGB')
        #img.save(outputDir + '/' + "localMapCol_" + str(x+1).zfill(4) + "_" + str(y+1).zfill(4) + ".png")
        #labels[x+1,y+1] = get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])
        test_set_x.append(localMapBW.flatten())
        test_set_y.extend([get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])])
        #if (get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])!=0) & (get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])!=1):
        #if x<=363:
        if (get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])!=1):
            train_set_x.append(localMapBW.flatten())
            #train_set_y.extend([get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])])
            train_set_y.extend([1])
        else:
            train_set_x.append(localMapBW.flatten())
            train_set_y.extend([0])
            #train_set_x.append(np.rot90(localMapBW,1).flatten())
            #train_set_y.extend([get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])])
            #train_set_x.append(np.rot90(localMapBW,2).flatten())
            #train_set_y.extend([get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])])
            #train_set_x.append(np.rot90(localMapBW,3).flatten())
            #train_set_y.extend([get_color_code(localMapCol[height/innerstep/2, width/innerstep/2])])
    print (x)
print (len(train_set_y))
print (len(test_set_y))
create_pkl_file(train_set_x, train_set_y, test_set_x, test_set_y)

from pylab import imshow, show, cm
def label_to_color(array, color_table):
    imagearray = []
    for i in range(len(array)):
        imagearray.append(color_table[array[i]])
    imagearray = np.array(imagearray, dtype='uint8')
    return imagearray

imagearray = label_to_color(train_set_y, color_table)
imagearray = imagearray.reshape(
    175,  # first image dimension (vertical)
    73,  # second image dimension (horizontal)
    3
)
image = Image.fromarray(imagearray)
imshow(image, cmap=cm.gray)
show()
#
#imagearray = label_to_color(train_set_y, color_table)
#imagearray.shape
#imagearray = imagearray.reshape(
#    651+49,  # first image dimension (vertical)
#    240+49,  # second image dimension (horizontal)
#    3
#)
#image = Image.fromarray(imagearray)
#imshow(image, cmap=cm.gray)
#show()
#image.save('full_' + inputImage + '.png',"PNG")
