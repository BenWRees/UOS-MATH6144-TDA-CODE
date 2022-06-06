import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import ttest_ind

import pandas as pd
form sklearn import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

def generate_data_set(num_pts,r) :
	data_sets = []
	X = np.empty([num_pts,2])
	x, y = np.random.uniform(), np.random.uniform()
	for i in range(num_pts):
	    X[i,:] = [x, y]
	    x = (X[i,0] + r * X[i,1] * (1-X[i,1])) % 1
	    y = (X[i,1] + r * x * (1-x)) % 1

def generate_data_sets(num_times, r) :
	data_sets = []
	for i in range(0,num_times) :
		generate_data_set(np.random.randint(50,1001),)



data_set_1 = generate_data_sets(30,3.5)
skel_1_test = Methods.Alpha_complexes(data_set_1[::2])
skel_1_train = Methods.Alpha_complexes(data_set_1[1::2])

data_set_2 = generate_data_sets(30,4)
skel_2 = Methods.Alpha_complexes(data_set_2)
skel_2_test = Methods.Alpha_complexes(data_set_2[::2])
skel_2_train = Methods.Alpha_complexes(data_set_2[1::2])

data_set_3 = generate_data_sets(30,4.5)
skel_3 = Methods.Alpha_complexes(data_set_3)
skel_3_test = Methods.Alpha_complexes(data_set_3[::2])
skel_3_train = Methods.Alpha_complexes(data_set_3[1::2])

data_set_4 = generate_data_sets(30,4.1)
skel_4 = Methods.Alpha_complexes(data_set_4)
skel_4_test = Methods.Alpha_complexes(data_set_4[::2])
skel_4_train = Methods.Alpha_complexes(data_set_4[1::2])

data_set_5 = generate_data_sets(30,4.5)
skel_5 = Methods.Alpha_complexes(data_set_5)
skel_5_test = Methods.Alpha_complexes(data_set_5[::2])
skel_5_train = Methods.Alpha_complexes(data_set_5[1::2])

data_set_6 = generate_data_sets(30,4.3)
skel_6 = Methods.Alpha_complexes(data_set_6)
skel_6_test = Methods.Alpha_complexes(data_set_6[::2])
skel_6_train = Methods.Alpha_complexes(data_set_6[1::2])

imgs_1_test = zip(Methods.persistence_images(skel_1_test,1,100,0.05),0)
imgs_2_test = zip(Methods.persistence_images(skel_2_test,1,100,0.05),1)
imgs_3_test = zip(Methods.persistence_images(skel_3_test,1,100,0.05),2)
imgs_4_test = zip(Methods.persistence_images(skel_4_test,1,100,0.05),3)
imgs_5_test = zip(Methods.persistence_images(skel_5_test,1,100,0.05),4)
imgs_6_test = zip(Methods.persistence_images(skel_6_test,1,100,0.05),5)

imgs_1_train = zip(Methods.persistence_images(skel_1_train,1,100,0.05),0)
imgs_2_train = zip(Methods.persistence_images(skel_2_train,1,100,0.05),1)
imgs_3_train = zip(Methods.persistence_images(skel_3_train,1,100,0.05),2)
imgs_4_train = zip(Methods.persistence_images(skel_4_train,1,100,0.05),3)
imgs_5_train = zip(Methods.persistence_images(skel_5_train,1,100,0.05),4)
imgs_6_train = zip(Methods.persistence_images(skel_6_train,1,100,0.05),5)

experiment_1_test = imgs_1_test + imgs_2_test + imgs_3_test + imgs_4_test + imgs_5_test + imgs_6_test 
experiment_1_train = imgs_1_train + imgs_2_train + imgs_3_train + imgs_4_train + imgs_5_train + imgs_6_train

clf = svm.SVC()
clf.fit(experiment_1_train[::,0],experiment_1_train[::,1])




