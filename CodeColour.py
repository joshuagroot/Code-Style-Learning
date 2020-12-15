import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import statistics
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # print(dirName)
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            # print(fullPath)
            if fullPath.endswith('.cpp') or fullPath.endswith('.h'):

                allFiles.append(fullPath)
                
    return allFiles

files = getListOfFiles("Bad Indentation Examples")
# print(files)
print(len(files))
# print(files[0])
count = 0
good = 0
badFiles = []
rowLengths = []
colLengths = []

for fileName in files:
    try:
        ogFile = open(fileName, 'r')
        # ogFile = open(fileName[:len(fileName)-4])
        print(ogFile)
        lines = ogFile.readlines()
        
        max = 0
        # Find the 'widest' line
        for i in lines:
            size = len(i) + (i.count('\t')*4)
            if size > max:
                max = size

        print(max)
        output = np.array([0] * max)
        output = np.expand_dims(output, axis=0)
        currentLine = np.array([])
        comment = False
        variableNames = []

        for i in lines:
            # print(repr(i))
            prevToken = ''

            prevPos = ''
            slashCount = 0
            currentToken = ''
            for pos, item in enumerate(i):
                
                # End of block comment
                if item == '/':
                    if len(i) > 1 and prevPos == '*':
                        comment = False
                        currentLine = np.append(currentLine, 1)
                        continue
                    else:
                        slashCount+=1
                        if slashCount == 2:
                            # print('two slash', i[len(currentLine)-1], len(currentLine)-1)
                            # print(i)
                            if prevPos == '/':
                                currentLine[-1] = 1
                            else:
                                slashCount = 1      
                # Start of block comment
                elif item == '*':
                    # print('COMMENT BLOCK', i[len(currentLine)-1], slashCount)
                    if prevPos == '/' and slashCount == 1:
                        currentLine[-1] = 1
                        comment = True
                        # print("TRUE")
                # Tab
                elif item == '\t':
                    zeroes = np.array([0] * 4)
                    currentLine = np.append(currentLine, zeroes)

                # Space
                elif item == ' ':
                    print(currentToken)
                    # Testing variable names
                    if prevToken == 'string':
                        print(currentToken)
                        variableNames.append(currentToken)
                    prevToken = currentToken
                    currentToken = ''
                    currentLine = np.append(currentLine, 0)
                # If we have double slashes on this line, this token is part of a comment
                elif item != '\n':
                    if slashCount >= 2 or comment:
                        currentLine = np.append(currentLine, 1)
                    # Else it is a code token
                    else:
                        currentToken += item
                        currentLine = np.append(currentLine, 2)

                prevPos = item
            
            if len(currentLine) < max:
                currentLine = np.append(currentLine, np.array([0] * (max-len(currentLine))))
            output = np.vstack((output, copy.deepcopy(currentLine)))
            rowLengths.append(len(output))
            colLengths.append(len(output[0]))
            currentLine = np.array([])
            print(variableNames)
        np.save(fileName+'.colorComments', output)
        imgplot = plt.imshow(output)
        
        plt.savefig(fileName+'Comments.png')
        plt.close()

        good+=1
        
    except:
        count+=1
        badFiles.append(fileName)

print(badFiles)
print(count)
print(good)

print(statistics.mean(rowLengths))
print(statistics.mean(colLengths))
print(statistics.stdev(rowLengths))
print(statistics.stdev(colLengths))