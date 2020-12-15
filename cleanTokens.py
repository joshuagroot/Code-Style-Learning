import sys
import re

def getTokenPos(input):
    tokens = {}
    if ',' in input:
        tokens[','] = input.find(',')
    if '->' in input:
        tokens['->'] = input.find('->')
    if '(' in input:
        tokens['('] = input.find('(')
    if ')' in input:
        tokens[')'] = input.find(')')
    if '[' in input:
        tokens['['] = input.find('[')
    if ']' in input:
        tokens[']'] = input.find(']')
    if '.' in input:
        tokens['.'] = input.find('.')
    if ';' in input:
        tokens[';'] = input.find(';')
    if ':' in input:
        tokens[':'] = input.find(':')
    if '#' in input:
        tokens['#'] = input.find('#')
    if '~' in input:
        tokens['~'] = input.find('~')
    if '<' in input:
        tokens['<'] = input.find('<')
    if '>' in input:
        tokens['>'] = input.find('>')
    if len(tokens) == 0:
        return -1, ''
    
    val = min(tokens, key=tokens.get)

    return tokens[val], val


def removeComments(input):
    comment = False
    commentStart = 0
    newList = []
    for n, i in enumerate(input):
        
        if '//' in i:
            # print('n: ', n, i)
            input[n] = '//'
            newList.append('//')
            commentStart = n
            # print('comment')
            comment = True
        elif i == 'newline':
            newList.append(i)
            comment = False
        else:
            if not comment:
                newList.append(i)
                # n-=1

    return newList

def removeStrings(input):
    quote = False
    newList = []

    for n, i in enumerate(input):
        if i == '"':
            newList.append(i)
            if not quote:
                newList.append('String')
            quote = not quote
        else:
            if not quote:
                newList.append(i)

    return newList


# print(sys.argv[1])
sample = open(sys.argv[1], "r") 
s = sample.readlines() 
sample.close()

spaceCount = 0


for n, i in enumerate(s):
    s[n] = s[n].replace("\\r\\n", "\\n")
    i = i.replace("\\r\\n", "\\n")
    # s[n] = s[n].replace("\r", "")

    
    s[n] = s[n][1:len(i)-2]
    # print('current: ', s[n])
    s[n] = s[n].replace('\\r', '')

    i = i[1:len(i)-2]
    i = i.replace('\\r', '')

    split = re.split(r'\\', i)
    # print('split: ', split)
    pos = 0
    spaceCount = 0
    for splitPos, j in enumerate(split):
        # print('J: ', j)
        if len(j) > 1 and (j[0] == 't' or j[0] =='n') and j != 'newline':

            if s[n].find(j) == -1:
                split[splitPos] = j[0]
                split.insert(splitPos+1, j[1:])
                j = j[0] 
                # print('new split: ', split)
    
        spaces = re.split(' ', j)
        # print('spaces: ', spaces, ' count: ', spaceCount)
        if spaces == ['', '']:
            spaceCount+=1
            continue
        if len(spaces) >= 1:
            
            for spacePos, space in enumerate(spaces):
                # print(getTokenPos(space))
                # print(spaceCount, ' ', space)
                if space != 't' and space != '':

                    if spaceCount > 0:
                        if pos == 0:
                            s[n] = str(spaceCount)+'space'
                        else:
                            s.insert(n+pos, '"'+str(spaceCount)+'space'+'"\n')
                        pos+=1
                        spaceCount = 0

                    if pos == 0:
                        if space == 'n':
                            # print('zero newline')
                            s[n] = 'newline'
                        else:
                            if len(space) > 1 and space[0] == '"':
                                s[n] = space[0]
                                pos+=1
                                s.insert(n+pos, '"'+space[1:]+'"\n')
                                # print('QUOTE: ', s[n+pos])
                            # elif len(space) > 2 and (space[0:2] == '//' or space[0:2] == '/*'):
                            #     s[n] = space[0:2]
                            #     pos +=1
                            #     s.insert(n+pos, '"'+space[2:]+'"\n')
                            elif len(space) > 2 and space[0:2] == '/*':
                                s[n] = space[0:2]
                                pos +=1
                                s.insert(n+pos, '"'+space[2:]+'"\n')
                            else:
                                s[n] = space

                            space = s[n]

                            if len(space) > 2 and space[len(space)-2:] == '*/':
                                s[n] = space[:len(space)-2]
                                pos+=1
                                s.insert(n+pos, '"'+space[len(space)-2:]+'"\n')

                            space = s[n]
                            # print('space: ', space)
                            foundPos, val = getTokenPos(space)
                            if foundPos != -1 and len(space) > 1:
                                # print('comma')
                                if len(space[:foundPos]) > 0:
                                    # print('stuff')
                                    s[n] = space[:foundPos]
                                    pos+=1
                                    s.insert(n+pos, '"'+val+'"\n')
                                    pos+=1
                                    s.insert(n+pos,'"'+space[foundPos+len(val):]+'"\n')
                                else:
                                    pos+=1
                                    s.insert(n+pos, '"'+space[foundPos+len(val):]+'"\n')
                                    # space = space[space.find(',')+1:]
                                    s[n] = val
                                    # print('s[n] ', s[n])
                                # pos+=1

                                # if len(space[space.find(',')+1:]) > 0:
                                #     print(space[space.find(',')+1:])
                                #     spaces.insert(spacePos+1, space[space.find(',')+1:])
                                #     # pos += 1 
                                #     # s.insert(n+pos, space[space.find(',')+1:])


                            
                    else:
                        if space == 'n':
                            # print('non zero newline')
                            s.insert(n+pos, '"newline"\n')
                        elif space != '':
                            # print('inserting: ' + space)

                            
                            if len(space) > 1 and space[0] == '"':
                                s.insert(n+pos, '"')
                                pos+=1
                                s.insert(n+pos, '"'+space[1:]+'"\n')
                                # print('QUOTE: ', s[n+pos])
                            elif len(space) > 2 and (space[0:2] == '//' or space[0:2] == '/*'):
                                s.insert(n+pos, space[0:2])
                                pos +=1
                                s.insert(n+pos, '"'+space[2:]+'"\n')
                            else:
                                s.insert(n+pos, '"'+space+'"\n')
                            space = s[n]
                            if len(space) > 2 and space[len(space)-2:] == '*/':
                                s[n] = space[:len(space)-2]
                                pos+=1
                                s.insert(n+pos, '"'+space[len(space)-2:]+'"\n')

                            # print(s[n], s[n+1], s[n+pos])
                        
                    pos+=1

                if space == 't':
                    spaceCount+=4
                if len(spaces) > 1 and spacePos < len(spaces)-1:
                    spaceCount +=1

    if spaceCount > 0:
        # print('extra spaces, ')
        if pos == 0:
            s[n] = str(spaceCount)+'space'
        else:
            s.insert(n+pos, '"'+str(spaceCount)+'space'+'"\n')
        pos +=1
        spaceCount = 0

    current = n
    
    # if '\"' in i:
        # print('quotation')

    # print('new: ', s[n])


# print(sys.argv[1])
# print(s)
# print(sys.argv[1])
# print(s[42])
# s = removeComments(s)
# s = removeStrings(s)
# sample = open(sys.argv[1], "w")

for i in s:
    if i != 'r':
        print(i, file=sample)
    # print(i)
    # sample.writelines(i)
sample.close()

# print(re.split(' ', '\n\n'))