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

learn_raw = np.loadtxt('learn.txt', delimiter='	')
test_raw = np.loadtxt('test.txt', delimiter='	')

spurious_columns = learn_raw.std(axis=0) > 1e-4

shuffle_ind = numpy.random.permutation(learn_raw.shape[0])
learn_raw = learn_raw[shuffle_ind, ]

split_point = (int)(0.8 * learn_raw.shape[0])

learn_answers = np.array(learn_raw[::split_point, 1], np.int)
learn = learn_raw[::, spurious_columns][::split_point, 2:]

val_answers = np.array(learn_raw[split_point::,1],np.int)
val = learn_raw[::,spurious_columns][split_point::,2:]





test_answers = np.array(test_raw[::, 1], np.int)
test = test_raw[::, spurious_columns][::, 2:]


scaler = preprocessing.StandardScaler().fit(learn)
learn = scaler.transform(learn)
val = scaler.transform(val)
test = scaler.transform(test)

np.savetxt("learn_transformed", learn, header = "{} {}".format(learn.shape[0],learn.shape[1]),comments='')
np.savetxt("learn_transformed_answers",learn_answers, header = "{}".format(learn_answers.shape[0]),comments='')

np.savetxt("val_transformed", val, header = "{} {}".format(val.shape[0],val.shape[1]),comments='')
np.savetxt("val_transformed_answers",val_answers, header = "{}".format(val_answers.shape[0]),comments='')

np.savetxt("test_transformed", test,header = "{} {}".format(test.shape[0],test.shape[1]),comments='')
np.savetxt("test_transformed_answers",test_answers, header = "{}".format(test_answers.shape[0]),comments='')
