

import tensorflow as tf
from functools import partial

import os
import sys
from tensorflow.keras import datasets, layers, models
from tensorflow.python import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import statistics
import numpy as np
import copy
import random
np.set_printoptions(threshold=sys.maxsize)
from PIL import Image

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
                # if 'color.npy' in fullPath and '.png' not in fullPath and 'Comment' not in fullPath:
                if fullPath.endswith('color.npy'):
                # if 'Comments' in fullPath and '.png' not in fullPath:
                    allFiles.append(fullPath)
    return allFiles

def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))
# vectorFiles = getListOfFiles('allSubmissions2', 'vector')
picturesFiles = getListOfFiles('allSubmissions2', 'npy')

# print(len(vectorFiles))
print(len(picturesFiles))
# random.shuffle(picturesFiles)

pictures = []
labels = []
count = 0
bad = 0
badFiles = []
removed = 0
fullMarks = []
fullMarksLabels = []
otherMarks = []
otherMarksLabels = []
# print(len(picturesFiles))
print(picturesFiles[0])
# exit()
legitFiles = []
for i in picturesFiles:
    tag = find_second_last(i, '.')

    # The mark is stored in the vector file
    if os.path.isfile(i[:tag] + '.vector'):

        current = open(i[:tag] + '.vector')

        mark = float(current.readline())
        if mark == 2:
            labels.append(mark)
            pictures.append(np.load(i))
            legitFiles.append(i)
            # img = Image.fromarray(np.load(i))

            # img.show(np.load(i))

            fullMarks.append(np.load(i))
            fullMarksLabels.append(mark)
        elif mark >= 0 and mark < 2:
            labels.append(mark)
            pictures.append(np.load(i))
            legitFiles.append(i)
            otherMarks.append(np.load(i))
            otherMarksLabels.append(mark)

        current.close()
    else:
        removed += 1

maxRow = 0
maxCol = 0

avgRow = 0
avgCol = 0
rowLengths = []
colLengths = []

print(statistics.mean(fullMarksLabels))
print(statistics.mean(otherMarksLabels))

fullRange = random.sample(range(len(fullMarks)), len(otherMarks))

for i in fullRange:
    otherMarks.append(fullMarks[i])
    otherMarksLabels.append(fullMarksLabels[i])

# pictures = otherMarks
# labels = otherMarksLabels

print(len(pictures))
print(len(labels))

print(statistics.mean(labels))

for i in pictures:
    rowLengths.append(len(i))
    colLengths.append(len(i[0]))
    if len(i) > maxRow:
        maxRow = len(i)
    if len(i[0]) > maxCol:
        maxCol = len(i[0])

avgRow = statistics.mean(rowLengths)

plt.hist(rowLengths, 2)
# plt.show()
# exit()

avgCol = statistics.mean(colLengths)
print(avgRow)
print(avgCol)
print(len(rowLengths))
print(len(colLengths))

avgRow = int(avgRow)
avgCol = int(avgCol)
stdRow = int(statistics.stdev(rowLengths))
stdCol = int(statistics.stdev(colLengths))


zeroesRows = np.array([0] * avgRow)

reshapedPictures = []

# Reshape all inputs to be the same dimensions
while count < len(pictures):
    new = np.empty(shape=(avgRow, avgCol))

    inner = 0
    while inner < len(pictures[count]) and inner < avgRow:

        innerLoop = inner%len(pictures[count])
        # print(innerLoop)
        if len(pictures[count][inner]) > avgCol:
            new[inner] = pictures[count][inner][:avgCol]
        else:
            new[inner] = np.concatenate((pictures[count][inner], np.zeros(avgCol-len(pictures[count][inner]))))
        inner += 1

    count += 1
    # print('appending')
    reshapedPictures.append(copy.deepcopy(new))

# print(pictures[0])

print(len(pictures))
print(len(reshapedPictures))
print(np.array(reshapedPictures).shape)
print(len(picturesFiles))
print(removed)

# Drawing the pictures and saving them
# for i, p in enumerate(reshapedPictures):
#     imgplot = plt.imshow(p)
#     # plt.show()
#     # exit()
#     plt.savefig(legitFiles[i]+'reshapedDouble.png')
#     plt.close()
# exit()

reshapedPictures = np.reshape(np.array(reshapedPictures), (len(reshapedPictures), avgRow, avgCol, 1))
# reshapedPictures.reshape(len(reshapedPictures), avgRow, avgCol, 1)
print('reshapedPictures shape: ', reshapedPictures.shape)
print(type(reshapedPictures))
labels = np.array(labels)
assert not np.any(np.isnan(labels))

predictionsMean = round(mean_absolute_error(labels, [statistics.mean(labels)]*len(labels)),3)
predictionsMedian = round(mean_absolute_error(labels, [statistics.median(labels)] *len(labels)),3)
predictionsMode = round(mean_absolute_error(labels, [statistics.mode(labels)] *len(labels)),3)


print('LEN TEST: ', int(len(reshapedPictures)*0.1))
test_x = np.array(reshapedPictures[:int(len(reshapedPictures)*0.1)])
print('test x shape: ', test_x.shape)
test_label = labels[:int(len(labels)*0.1)]
print('CHECKING LABELS: ', all(i >= 0 and i <= 2 for i in test_label))

print(test_label.shape)

print('LEN TRAIN ', int(len(reshapedPictures)*0.9))
train_x = reshapedPictures[int(len(reshapedPictures)*0.1):]
print('train x shape: ', train_x.shape)
train_label = np.array(labels[int(len(labels)*0.1):])
print('CHECKING LABELS: ', all(i >= 0 and i <= 2 for i in train_label))

print(train_label.shape)

model = models.Sequential()
initializer = tf.keras.initializers.GlorotNormal()
regularizer = tf.keras.regularizers.l2(l=0.0)
model.add(layers.Conv2D(20, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), input_shape=(avgRow, avgCol, 1), kernel_initializer=initializer) )
model.add(layers.Conv2D(20, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))

# model.add(layers.BatchNormalization()) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))
# model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))
# model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())

model.add(layers.Dense(20, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))
# model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=initializer))
# model.add(layers.Dropout(0.1))
model.add(layers.Dense(1))


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae','mse'])

model.summary()



print(model)

print('LAYER LOOP')

for layer in model.layers:
    print(layer.output_shape)



# # exit()
# print(train_x.shape)
# print(test_x.shape)
# print(train_label.shape)
# print(test_label.shape)
# # print(len(train_x[0][1]))
history = model.fit(train_x, train_label, epochs=50, validation_data=(test_x, test_label), batch_size=32)

plt.style.use('seaborn-whitegrid')
plt.figure()
plt.plot(history.history['mae'], label='train_error')
plt.plot(history.history['val_mae'], label = 'val error')
plt.plot([predictionsMean] * len(history.history['mae']), 'g--', label='Mean')
plt.plot([predictionsMedian] * len(history.history['mae']), 'r--', label='Median/Mode')


plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.legend(loc='lower left')
plt.show()


plt.figure()
plt.plot(history.history['mse'], label='train_error')
plt.plot(history.history['val_mse'], label = 'val error')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()


testFiles = getListOfFiles('Bad Indentation Examples', 'npy')
print(testFiles)

testLabels = []
testPictures = []

for i in testFiles:
    tag = find_second_last(i, '.')
    print(i[:tag]+'.vector')
    # try:

    if os.path.isfile(i[:tag] + '.vector'):

        current = open(i[:tag] + '.vector')

        mark = float(current.readline())
        if mark >= 0:
            testLabels.append(mark)
            # print(current.readline())
            # print(np.load(i))
            testPictures.append(np.load(i))
            count += 1

        current.close()

print(testLabels)
count = 0

reshapedTestPictures = []

while count < len(testPictures):
    # if len(pictures[count]) < 5:
    #     count+=1
    #     continue
    new = np.empty(shape=(avgRow, avgCol))
    # print(pictures[count].shape)
    inner = 0
    while inner < len(testPictures[count]) and inner < avgRow:
    # while inner < avgRow:
        innerLoop = inner%len(testPictures[count])
        # print(innerLoop)
        if len(testPictures[count][inner]) > avgCol:
            # print(len(testPictures[count][inner]), inner)
            new[inner] = testPictures[count][inner][:avgCol]
        else:
            # temp = np.zeros(avgCol-len(testPictures[count][inner]))
            # print(temp)
            # exit()
            new[inner] = np.concatenate((testPictures[count][inner], np.zeros(avgCol-len(testPictures[count][inner]))))
            # new[inner] = np.concatenate(testPictures[count][inner], np.zeros(avgCol-len(testPictures[count][inner])))
            # new[inner] = np.concatenate(testPictures[count][inner], np.empty([0] * (avgCol-len(testPictures[count][inner]))))
        inner += 1
        # else

    count += 1
    # print('appending')
    reshapedTestPictures.append(copy.deepcopy(new))
    # break
    # while inner < len(pictures[count]):

# Drawing the pictures and saving them
# for i, p in enumerate(reshapedTestPictures):
#     imgplot = plt.imshow(p)
#     # plt.show()
#     # exit()
#     plt.savefig(testFiles[i]+'reshapedDouble.png')
#     plt.close()

reshapedTestPictures = np.reshape(np.array(reshapedTestPictures), (len(reshapedTestPictures), avgRow, avgCol, 1))
testLabels = np.array(testLabels)
print(testLabels)
print(testFiles)

results = model.evaluate(reshapedTestPictures,  testLabels)
print(results)

predictions = model.predict(reshapedTestPictures)
print(predictions)
# print(test_loss)
# print(test_acc)

# plt.show()

# print(pictures[0])
# print(len(pictures[0]))
# print(len(pictures[0][0]))

# https://pypi.org/project/Code2pdf/
# https://pypi.org/project/pdf2image/