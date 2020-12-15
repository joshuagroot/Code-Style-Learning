
# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
import warnings
import os
import glob
from sklearn.manifold import TSNE
import numpy as np



# sentence = "Gday mate how are you?"

# tokens = nltk.word_tokenize(sentence)
# print(tokens)
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

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
            if (fullPath.endswith('cpp.txt') or fullPath.endswith('h.txt')):
                allFiles.append(fullPath)
                
    return allFiles

print('-----')
fileNames = getListOfFiles('allSubmissions2')

print(len(fileNames))

data = [[]]
wordSet = set()
count = 0
noCount = 0

comment = False
commentSymbol = ''
string = False

for i in fileNames:
    # print(i)
    sample = open(i, 'r')
    s = sample.read()
    sample.close()
    temp = s.split('\n')
    # Comment contents handling
    for i in list(temp):
        # if '//' in i:
        #     comment = True
        #     if commentSymbol != '/*':
        #         commentSymbol = '//'
        # elif '/*' in i:
        #     comment = True
        #     commentSymbol = '/*'
        # elif i == 'newline':
        #     if commentSymbol == '//':
        #         comment = False
        #         commentSymbol = ''
        # elif i == '*/':
        #     if commentSymbol == '/*':
        #         comment = False
        #         commentSymbol = ''
        # else:
        #     if comment == True:
        #         temp.remove(i)
        #         continue

        # if '"' in i:
        #     if string:
        #         string = False
        #     else:
        #         string = True
        # elif string:
        #     temp.remove(i)
        #     continue


        if '\\n' in i:
            temp.remove(i)

    for word in temp:
        if '\\n' not in word:
            wordSet.add(word)

    # print(temp)
    count += len(temp)
    data.append(temp)

print(len(data))
print('count: ' + str(count))
print(count / len(data))
# print(wordSet)

# Word2Vec model here, set workers to number of cpu cores
model = gensim.models.Word2Vec(data, size=100, min_count=20, iter=200, window=30, workers=8)
word_vectors = model.wv

# Save word2vec model here!
# fname = get_tmpfile("Vectors/allSubLessNoiseWindow30MinCount10.kv")
# kv = KeyedVectors(100)
# vectors = []
# for i in model.wv.vocab.keys():
#     vectors.append(model.wv[i])

# print(list(model.wv.vocab.keys()))
# kv.add(list(model.wv.vocab.keys()), vectors)
# kv.save('Vectors/allSubWindow30MinCount10.kv')
# model.save('W2VModels/allSubLessNoiseWindow30MinCount10.model')

# kv2 = KeyedVectors.load('Vectors/prac42019.kv')
# print(kv2['int'])
# word_vectors.save(fname)

# print(model.wv.vocab.keys())

# # # print('-----')
# # print(model.wv['}'])



print('------')
mostSimilarInt = model.most_similar(positive=['int'])

for i in mostSimilarInt:
    i = list(i)
    print(i[1])
    print(type(i[1]))
    i[1] = round(i[1], 2)
       
print(mostSimilarInt)

print(type(mostSimilarInt[1][1]))
print(model.most_similar(positive=['int']))
print('------')
mostSimilarMain = model.most_similar(positive=['main'])

print(model.most_similar(positive=['main']))
print('------')
mostSimilarBrace = model.most_similar(positive=['}'])

print(model.most_similar(positive=['}']))
print('-------')
mostSimilarCout = model.most_similar(positive=['cout'])

print(model.most_similar(positive=['cout']))
print('-------')
mostSimilarEndl = model.most_similar(positive=['endl'])

print(model.most_similar(positive=['endl']))

# # TSNE  - clustering visualisation
embedding_clusters =[]
word_clusters = []
keys = ['endl', 'int', 'main', '}']

for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=5):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)


embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=1000, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}

# matplotlib.rc('font', **font)

matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename='PosterPNG.png'):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=10, fontsize=15)
    plt.legend(loc=4, prop={'size': 12})
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()

tsne_plot_similar_words('Word clustering', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

