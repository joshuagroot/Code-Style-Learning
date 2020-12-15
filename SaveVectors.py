

import os
import glob

import gensim 
import re
from gensim.models import KeyedVectors

def getListOfFiles(dirName, files):
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
            allFiles = allFiles + getListOfFiles(fullPath,files)
        else:
            if '.txt' in fullPath and files == 'txt':
                allFiles.append(fullPath)
            if 'feedback' in fullPath and files == 'feedback':
                allFiles.append(fullPath)
            if 'override' in fullPath and files == 'override':
                allFiles.append(fullPath)
                
    return allFiles

def find_second_last(text, pattern):
    return text.rfind(pattern, 0, text.rfind(pattern))

fileNames = getListOfFiles('AllSubmissions2NoComments', 'txt')
feedback = getListOfFiles('AllSubmissions2NoComments', 'feedback')
override = getListOfFiles('AllSubmissions2NoComments', 'override')


kv2 = KeyedVectors.load('Vectors/AllSubWindow30MinCount10.kv')
count = 0
max = 0
outputFiles = []
# print(fileNames)
for i in fileNames:

    current = []
    print(i)
    if '.cpp.txt' in i or '.h.txt' in i: 
        count+=1
        sample = open(i, 'r')
        print(i)
        s = sample.read()
        sample.close()
        temp = s.split('\n')

        for j in list(temp):
            if '\\n' in j:
                temp.remove(j)
        
        for j in list(temp):
            if j in kv2:
                current.append(kv2[j])
        
        # count += len(current)
        finalDir = i.rfind('/')
        finalDot = find_second_last(i, '.')
        if (i[finalDot+1] == 'h'):
            finalDot+=2
        else:
            finalDot+=4
        outputDir = i[:finalDir+1]
        outputFile = i[:finalDot]+'.vector'
        
        outputFiles.append(outputFile)

        cId = i.find('C')
        # print(i)
        # print(cId)
        cId = i[:find_second_last(i, '/')]
        # cId = i[cId:i[cId:].find('/')]
        # print(cId)

        reg = re.compile(cId +"/feedback.+")
        overrideReg = re.compile(cId+"/override.+")


        feedbackFiles = list(filter(reg.match, feedback))
        overrideFiles = list(filter(overrideReg.match, override))

        # print(feedbackFiles)

        # print(feedbackFiles)
        # print('trying')
        overrideMark = 0
        feedbackFile = ''
        if len(feedbackFiles) > 0:
            feedbackFile = open(feedbackFiles[len(feedbackFiles)-1], 'r')
        print(feedbackFile)

        # print(feedbackContents.read())
        feedbackContents = ''
        try:
            feedbackContents = feedbackFile.read().lower()
        except:
            feedbackContents = ''
        print(feedbackContents)
        if len(overrideFiles) > 0:
            overrideFile = open(overrideFiles[len(overrideFiles)-1], 'r')
            overrideContents = overrideFile.read()
            if len(overrideContents) > 0:
                try:
                    overrideMark = float(overrideContents)
                except:
                    overrideMark = 0

        tempStr = feedbackContents[:feedbackContents.find('\n')]
        
        mark = -1
        # if (float(tempStr) > 0):
        #     mark = float(tempStr)
        # tempStr = ''
        # if 'commenting/style' in feedbackContents:
        #     print('design')

        if len(tempStr) > 0 and (float(tempStr) == 6 or overrideMark == 6):
            # print('good')
            mark = 2
        elif 'style/commenting' in feedbackContents:
            tempStr = feedbackContents[feedbackContents.find('style/commenting')+len('style/commenting')+2:]
        elif 'commenting/style' in feedbackContents:
            # print('should be here')
            tempStr = feedbackContents[feedbackContents.find('commenting/style')+len('commenting/style')+2:]
        elif 'style/comments' in feedbackContents:
            tempStr = feedbackContents[feedbackContents.find('style/comments')+len('style/comments')+2:]
        elif  'comments/style' in feedbackContents:
            tempStr = feedbackContents[feedbackContents.find('comments/style')+len('comments/style')+2:]
        elif 'style' in feedbackContents:
            # old
            tempStr = feedbackContents[feedbackContents.find('style')+7:]

        if len(tempStr) > 0:
            print(tempStr)
            tempStr = tempStr[:tempStr.find('\n')]
            if '/' in tempStr:
                tempStr = tempStr[:tempStr.find('/')]
            # tempStr = feedbackContents[feedbackContents.find('Style'):]
            if len(tempStr) < 4:
                try:
                    mark = float(tempStr)
                    if mark < 2:
                        count +=1
                except:
                    print(tempStr)
                    print('bad file')
        print(mark)
        if mark != -1:
            f = open(outputFile, 'w')
            # print(mark)
            f.write(str(mark) + '\n')
            count += len(current)
            if len(current) > max:
                max = len(current)
            for ele in current:
                for item in ele:
                    f.write(str(item) + ' ')
                f.write('\n')
                # print(str(ele)+'\n')
            f.close()
        # break


# print(count)
print(max)
# print(outputFiles)
# print(outputFiles)