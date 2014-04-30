from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist
from queue import PriorityQueue
import numpy as np
import pandas as pd
from scipy.spatial.distance import *

learnRaw = np.loadtxt('learn.txt', delimiter='	')
testRaw = np.loadtxt('test.txt', delimiter='	')

spuriousColumns = learnRaw.std(axis=0) > 1e-4

learnAnswers = np.array(learnRaw[::, 1], np.int)
learn = learnRaw[::, spuriousColumns][::, 2:]

testAnswers = np.array(testRaw[::, 1], np.int)
test = testRaw[::, spuriousColumns][::, 2:]

scaler = preprocessing.StandardScaler().fit(learn)
learn = scaler.transform(learn)
test = scaler.transform(test)

np.savetxt("learn_transformed", learn, header = "{} {}".format(learn.shape[0],learn.shape[1]),comments='')
np.savetxt("learn_transformed_answers",learnAnswers, header = "{}".format(learnAnswers.shape[0]),comments='')

np.savetxt("test_transformed", test,header = "{} {}".format(test.shape[0],test.shape[1]),comments='')
np.savetxt("test_transformed_answers",testAnswers, header = "{}".format(testAnswers.shape[0]),comments='')
