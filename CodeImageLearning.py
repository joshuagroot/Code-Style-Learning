

import tensorflow as tf
from functools import partial

import os
import sys
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import statistics
import numpy as np
import copy
np.set_printoptions(threshold=sys.maxsize)

import struct
import imghdr
import imageio
from skimage.transform import resize
import random

import keras

from PIL import Image

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
            return height, width

    


def getListOfFiles(dirName, fileType):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath, fileType)
        else:
            if fileType == 'vector':
                if '.vector' in fullPath:
                    allFiles.append(fullPath)
            elif fileType == 'npy':
                if '_reshaped' in fullPath:
                    allFiles.append(fullPath)   
    return allFiles

def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))

def find_last(text, pattern):
    return text.rfind(pattern, 0)

# https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
class My_Custom_Generator(keras.utils.Sequence) :
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    # tempArr = np.array(dtype=np.uint8)

    # for file_name in batch_x:
    #     tempArr.append(np.load(file_name), )
    return np.array([
            np.load(file_name)
               for file_name in batch_x]), np.array(batch_y)



# plt.figure()
# plt.plot([], label='train_error')
# plt.plot([], label = 'val_error')
# plt.xlabel('Epoch')
# plt.ylabel('Mean Abs Error')
# plt.legend(loc='lower right')
# plt.show()
picturesFiles = getListOfFiles('allSubmissions2', 'npy')
# exit()

print(picturesFiles[0])

print(len(picturesFiles))
pictures = []
labels = []
count = 0
legitFiles = []
fullMarks = []
fullMarksLabels = []
otherMarks = []
otherMarksLabels = []

for i in picturesFiles:
    tag = find_last(i, '_')
    # print(i[:tag])
    # exit()
    if os.path.isfile(i[:tag] + '.vector'):

        current = open(i[:tag] + '.vector')

        mark = float(current.readline())
        if mark >= 0 and mark <= 2:
            labels.append(mark)
            # pictures.append(imageio.imread(i))
            # pictures.append(np.load(i))
            legitFiles.append(i)
        if mark == 2:
            fullMarks.append(i)
            fullMarksLabels.append(mark)
        elif mark < 2 and mark >= 0:
            otherMarks.append(i)
            otherMarksLabels.append(mark)
        current.close()

print(len(legitFiles))

print('trucnating averages')
avgRow = 463
avgCol = 543

batch_size = 32

# pictures = np.array(pictures, dtype=np.uint8)
# print(type(pictures))
# print(pictures.shape)

balancedFiles = otherMarks
balancedLabels = otherMarksLabels
pictures = legitFiles

fullRange = random.sample(range(len(otherMarks)), len(balancedFiles))
for i in fullRange:
    balancedFiles.append(fullMarks[i])
    balancedLabels.append(fullMarksLabels[i])

# pictures = balancedFiles
# labels = balancedLabels

print(len(fullRange))
labels = np.array(labels)
print(len(fullMarks))
print(len(otherMarks))
print(len(balancedFiles))

fullCount = 0
otherCount = 0
for i in labels:
    if i == 2:
        fullCount +=1
    else:
        otherCount += 1

print(fullCount, otherCount)
print(len(pictures), len(labels))
# exit()
# print('LEN TEST: ', int(len(legitFiles)*0.2))
# test_x = np.array(legitFiles[:int(len(legitFiles)*0.2)])
# print('test x shape: ', test_x.shape)
# test_label = np.array(labels[:int(len(labels)*0.2)], dtype=np.uint8)
# # print(test_label.shape)

# print('LEN TRAIN ', int(len(legitFiles)*0.8))
# train_x = np.array(legitFiles[int(len(legitFiles)*0.2):])
# # del legitFiles
# # print('train x shape: ', train_x.shape)
# train_label = np.array(labels[int(len(labels)*0.2):], dtype=np.uint8)

# my_training_batch_generator = My_Custom_Generator(train_x, train_label, batch_size)
# my_validation_batch_generator = My_Custom_Generator(test_x, test_label, batch_size)


predictionsMean = round(mean_absolute_error(labels, [statistics.mean(labels)]*len(labels)),3)
predictionsMedian = round(mean_absolute_error(labels, [statistics.median(labels)] *len(labels)),3)
predictionsMode = round(mean_absolute_error(labels, [statistics.mode(labels)] *len(labels)),3)

print('LEN TEST: ', int(len(pictures)*0.2))
test_x = np.array(pictures[:int(len(pictures)*0.2)])
print('test x shape: ', test_x.shape)
test_label = np.array(labels[:int(len(labels)*0.2)], dtype=np.uint8)
# print(test_label.shape)

print('LEN TRAIN ', int(len(pictures)*0.8))
train_x = np.array(pictures[int(len(pictures)*0.2):])
# del balancedFiles
# print('train x shape: ', train_x.shape)
train_label = np.array(labels[int(len(labels)*0.2):], dtype=np.uint8)

my_training_batch_generator = My_Custom_Generator(train_x, train_label, batch_size)
my_validation_batch_generator = My_Custom_Generator(test_x, test_label, batch_size)

model = models.Sequential()
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), input_shape=(avgRow, avgCol, 3)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(60, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Conv2D(60, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(60, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Conv2D(60, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(20, activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Dense(10, activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Dense(1))



model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae','mse'])

model.summary()
# exit()

print(model)


# history = model.fit(train_x, train_label, epochs=20, validation_data=(test_x, test_label), batch_size=1)
history = model.fit(my_training_batch_generator,
                             steps_per_epoch=int(len(train_x)/batch_size),
                             epochs=15,
                             validation_data=my_validation_batch_generator,
                             validation_steps=int(len(test_x)/batch_size))


plt.figure()
plt.plot(history.history['mae'], label='train_error')
plt.plot(history.history['val_mae'], label = 'val_error')
plt.plot([predictionsMean] * len(history.history['mae']), 'g--', label='Mean')
plt.plot([predictionsMedian] * len(history.history['mae']), 'r--', label='Median/Mode')
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.legend(loc='lower left')
plt.savefig('MAE.png')
plt.show()


plt.figure()
plt.plot(history.history['mse'], label='train_error')
plt.plot(history.history['val_mse'], label = 'val error')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
# plt.show()
plt.savefig('MSE.png')


# testFiles = getListOfFiles('allSubmissions2', 'npy')
# print(testFiles)


# # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
# testLabels = []
# testPictures = []

# for i in testFiles:
#     tag = find_second_last(i, '.')
#     print(i[:tag]+'.vector')
#     # try:

#     if os.path.isfile(i[:tag] + '.vector'):

#         current = open(i[:tag] + '.vector')

#         mark = float(current.readline())
#         if mark >= 0:
#             testLabels.append(mark)
#             # print(current.readline())
#             # print(np.load(i))
#             testPictures.append(np.load(i))
#             count += 1

#         current.close()

# print(testLabels)
# count = 0

# reshapedTestPictures = []

# while count < len(testPictures):
#     # if len(pictures[count]) < 5:
#     #     count+=1
#     #     continue
#     new = np.empty(shape=(avgRow, avgCol))
#     # print(pictures[count].shape)
#     inner = 0
#     while inner < len(testPictures[count]) and inner < avgRow:
#     # while inner < avgRow:
#         innerLoop = inner%len(testPictures[count])
#         # print(innerLoop)
#         if len(testPictures[count][inner]) > avgCol:
#             # print(len(testPictures[count][inner]), inner)
#             new[inner] = testPictures[count][inner][:avgCol]
#         else:
#             # temp = np.zeros(avgCol-len(testPictures[count][inner]))
#             # print(temp)
#             # exit()
#             new[inner] = np.concatenate((testPictures[count][inner], np.zeros(avgCol-len(testPictures[count][inner]))))
#             # new[inner] = np.concatenate(testPictures[count][inner], np.zeros(avgCol-len(testPictures[count][inner])))
#             # new[inner] = np.concatenate(testPictures[count][inner], np.empty([0] * (avgCol-len(testPictures[count][inner]))))
#         inner += 1
#         # else

#     count += 1
#     # print('appending')
#     reshapedTestPictures.append(copy.deepcopy(new))
#     # break
#     # while inner < len(pictures[count]):

# # Drawing the pictures and saving them
# # for i, p in enumerate(reshapedTestPictures):
# #     imgplot = plt.imshow(p)
# #     # plt.show()
# #     # exit()
# #     plt.savefig(testFiles[i]+'reshapedDouble.png')
# #     plt.close()

# reshapedTestPictures = np.reshape(np.array(reshapedTestPictures), (len(reshapedTestPictures), avgRow, avgCol, 1))
# testLabels = np.array(testLabels)
# print(testLabels)
# print(testFiles)

# results = model.evaluate(reshapedTestPictures,  testLabels)
# print(results)

# predictions = model.predict(reshapedTestPictures)
# print(predictions)
# # print(test_loss)
# # print(test_acc)

# # plt.show()

# # print(pictures[0])
# # print(len(pictures[0]))
# # print(len(pictures[0][0]))

# # https://pypi.org/project/Code2pdf/
# # https://pypi.org/project/pdf2image/