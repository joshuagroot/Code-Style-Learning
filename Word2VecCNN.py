import tensorflow as tf
from functools import partial

import os

from sklearn.metrics import mean_absolute_error, mean_squared_error


from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import statistics
import numpy as np
import copy

def getListOfFiles(dirName):
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
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if '.vector' in fullPath:
                allFiles.append(fullPath)
                
    return allFiles

def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))

fileVectors = {}
fileMarks = {}

train_x = []
train_label = []
test_x = []
test_label = []
zeroes = [0] * 100

fileNames = getListOfFiles('allSubmissions2')

count = 0
max = 0
lengths = []
print(len(fileNames))
badFiles = 0

for i in fileNames:
    f = open(i, 'r')
    lines = f.readlines()
    
    if float(lines[0]) < 0 or float(lines[0]) > 2:
        badFiles +=1
        continue
        
    if len(lines)-1 > max:
        max = len(lines)-1

    vectorList = []
    verts = []
    # print(x)

    count += len(lines)-1
    if len(lines)-1 > max:
        max = len(lines)-1
    for line in lines[1:]:
        # print(len([float(val) for val in line.split()]))
        verts.append([float(val) for val in line.split()])
        vectorList.extend(verts)
        verts = []

    fileMarks[i] = float(lines[0])
    fileVectors[i] = vectorList
    train_x.append(vectorList)
    train_label.append(float(lines[0]))
    lengths.append(len(vectorList))
    count+=1

# print(train_label)

print(badFiles)
# exit()
deviation = int(statistics.stdev(lengths))
mean = int(statistics.mean(lengths))
remove = []
print('--------')
print(len(train_x[0][1]))
print('--------')

plt.style.use('seaborn-whitegrid')
plt.hist(lengths)
plt.xlabel('File length')
plt.ylabel('No. of files')
zeroesCount = 0
# plt.show()
for i in range(len(train_label)):
    # print(t_label[i])
    if train_label[i] == 0.0:
        zeroesCount += 1
        remove.append(i)
    # elif len(train_x[i]) > (mean+deviation)*2:
    #     remove.append(i)
    # elif len(train_x[i]) < deviation-mean:
    #     remove.append(i)

print("ZEROES COUNT: ", zeroesCount)
print(max)
print(count)
lengths.sort()
# print(lengths)
print("standard dev: ", int(statistics.stdev(lengths)))
print("Mean: ", statistics.mean(lengths))
print("Median: ", statistics.median(lengths))

# print(len(train_x))
count = 0
for i in remove:
    # print(i)
    del train_x[i-count]
    del train_label[i-count]
    count+=1


lengths = []

for i in train_x:
    lengths.append(len(i))

print(count)
lengths.sort()
# print(lengths)
print("standard dev: ", int(statistics.stdev(lengths)))
print("Mean: ", statistics.mean(lengths))
print("Median: ", statistics.median(lengths))

plt.hist(lengths)
plt.xlabel('File length')
plt.ylabel('No. of files')
# plt.show()

for i in fileVectors:
    # print(i)
    if len(fileVectors[i]) > mean:
        del fileVectors[i][mean:]
        # print(len(fileVectors[i]))
    else:
        while len(fileVectors[i]) < mean:
            fileVectors[i].append(copy.deepcopy(zeroes))

print('---------')
# for i in train_x:
#     print(len(i))
            
# print(remove)
# print(len(train_x))


predictionsMean = round(mean_absolute_error(train_label, [statistics.mean(train_label)]*len(train_label)),3)
predictionsMedian = round(mean_absolute_error(train_label, [statistics.median(train_label)] *len(train_label)),3)
predictionsMode = round(mean_absolute_error(train_label, [statistics.mode(train_label)] *len(train_label)),3)

print('MEAN: ' , predictionsMean)
print('MEDIAN: ' , predictionsMedian)
print('MODE: ' ,  predictionsMode)

test_x = train_x[int(len(train_x)*0.8):]
test_label = train_label[int(len(train_label)*0.8):]

train_x = train_x[:int(len(train_x)*0.8)]
train_label = train_label[:int(len(train_label)*0.8)]

# print(len(test_x))
# print(len(train_x))
# print('--------')
# print('TEST LABELS')
# print(test_label)
# print(len(test_label))

# print(list(fileVectors.values())[0])



s = [1 for n in range (len(lengths))]
y = [0]*len(lengths)

model = models.Sequential()
model.add(layers.Conv2D(20, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01), input_shape=(mean, 100, 1)))
model.add(layers.Conv2D(20, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
# model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
# model.add(layers.BatchNormalization()) 

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Conv2D(40, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.01)))
# model.add(layers.BatchNormalization()) 

model.add(layers.Flatten())
model.add(layers.Dense(32, activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Dense(32, activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Dense(10, activation=partial(tf.nn.leaky_relu, alpha=0.01)))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae','mse'])

# print(train_x[0][2])

train_x = np.array(train_x)
test_x = np.array(test_x)
train_label = np.array(train_label)
test_label = np.array(test_label)

print(train_x.shape)
print(len(train_x))
print(len(train_x[0]))
# print(train_x[0])

        
print(len(train_x[0][1]))
print(train_x.shape)

print(test_x.shape)
print(len(test_x))
print(len(test_x[0]))
print(len(test_x[0][1]))
train_x = train_x.reshape(len(train_x), len(train_x[0]), len(train_x[0][1]), 1)
# train_x = train_x.reshape(-1, len(train_x[0]), len(train_x[0][1]), 1)
test_x = test_x.reshape(-1, len(test_x[0]), len(test_x[0][1]), 1)
print(test_x.shape)
# train_label = train_label.reshape(-1, len(train_label[0]), len(train_label[0][1]), 1)
# train_x = train_x.reshape(-1, len(train_x[0]), len(train_x[0][1]), 1)
model.summary()
# exit()

print(train_x.shape)
print(test_x.shape)
print(train_label.shape)
# print(len(train_x[0][1]))
history = model.fit(train_x, train_label, epochs=15, 
                    validation_data=(test_x, test_label))

plt.figure()
plt.plot(history.history['mae'], label='train_error')
plt.plot(history.history['val_mae'], label = 'val error')
plt.xlabel('Epoch')
plt.plot([predictionsMean] * len(history.history['mae']), 'g--', label='Mean')
plt.plot([predictionsMedian] * len(history.history['mae']), 'r--', label='Median/Mode')
plt.legend(loc='lower left')
plt.ylabel('Mean Abs Error')

plt.figure()
plt.plot(history.history['mse'], label='train_error')
plt.plot(history.history['val_mse'], label = 'val error')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
# plt.ylim([0.5, 1])

# test_loss, test_acc = model.evaluate(test_x,  test_label, verbose=2)

plt.show()


# print(fileNames)