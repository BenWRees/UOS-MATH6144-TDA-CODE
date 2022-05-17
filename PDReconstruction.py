import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import ttest_ind

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns



#persistence images 
#compute n distributions and their persistence diagrams as well as random 0 or 1 or real value
#package into list of pairs
def build_input_data(list_of_values,n) :
	distributions = Methods.generate_norm_distributions(4,n,100,Methods.generate_rand_tori)
	return [(a,b) for a,b in zip(distributions,list_of_values)]

data = build_input_data([np.random.randint(0,2) for a in range(0,3)],3)
#compute persistence images for each diagram and package with diagrams corresponding
#0 or 1 or real value
def images_data(data,dim,var,f,res) :
	images_data = []
	for i in data :
		distribution = i[0]
		ACcomplex = gd.AlphaComplex(points = distribution).create_simplex_tree()
		pers = ACcomplex.persistence()
		PI = PersistenceImage(bandwidth=var, weight=f, im_range=[0,.004,0,.004], resolution=[res,res])
		pi = PI.fit_transform([ACcomplex.persistence_intervals_in_dimension(dim)])
		images_data.append((pi[0],i[1]))
	return images_data

img_data = images_data(data,1,0.002,lambda x : x[1], 100)
print(img_data)

#apply logistic or linear regression on set of pairs 
model = LogisticRegression()

model.fit([a[0] for a in img_data], [a[1] for a in img_data])
print(len(model.coef_[0]))
"""
#accuracy of training
train_acc = model.score([a[0] for a in img_data], [a[1] for a in img_data])
print("The Accuracy for Training Set is {}".format(train_acc*100))

test_acc = accuracy_score(y_test, y_pred)
print("The Accuracy for Test Set is {}".format(test_acc*100))
"""

#get weight - new persistence image
learned_weight = model.coef_[0]
