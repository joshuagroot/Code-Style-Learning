SemesterTokens.sh - Convert code files into token files

cleanTokens.py - Used by SemesterTokens to clean up token files

word2vec.py - Run word2vec model on student token files. The resulting representations can then be saved in a keyed vector file (kv).

SaveVectors.py - Create vector files based off of representations generated through word2vec. The keyed vector (kv) file needs to be loaded in

Word2VecCNN.py - Run the CNN model on the vectorised files.

CodeColour.py - Create color representations of files using np arrays.

ColourLearning.py - Run the CNN model on the code colour files.

CodeImageLearning.py - Run the CNN model on screenshots of code (Note: this is currently too large a model to run on a standard laptop)
