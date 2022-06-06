import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from sklearn import svm
from sklearn import metrics

#generate 10 data sets of between 1-4 Tori each of 1000 pts 
distributions_torus = Methods.generate_norm_distributions(1,10,1000,Methods.generate_rand_tori,2)
distributions_sphere = Methods.generate_norm_distributions(1,10,1000,Methods.generate_rand_spheres,2)

distributions_train_torus = distributions_torus[::2] 
distributions_train_sphere = distributions_sphere[::2]

distributions_test_torus = distributions_torus[1::2] 
distributions_test_sphere = distributions_sphere[1::2]

alpha_skeletons_train_torus = Methods.Alpha_complexes(distributions_train_torus)
alpha_skeletons_train_sphere = Methods.Alpha_complexes(distributions_train_sphere)

alpha_skeletons_test_torus = Methods.Alpha_complexes(distributions_test_torus)
alpha_skeletons_test_sphere = Methods.Alpha_complexes(distributions_test_sphere)

"""
for i in distributions :
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	#ax1 = fig.add_subplot(121, projection='3d')
	ax.scatter(i[:,0], i[:,1], i[:,2])
	ax.set_zlim(-3,3)
	ax.view_init(36, 116)
	plt.show()
"""
def plot_imgs(list_of_images) :
	for pi in list_of_images :
		size = int(np.sqrt(len(pi[0][0])))
		plt.imshow(np.flip(np.reshape(pi[0], [size,size]), 0))
		plt.title("Persistence Image")
		plt.show()

#Persistence Images - investigate effects of resolution, distribution and weighting function 
def images_parameters(distributions, label, dimension, resolution=20,variance=1.0, f=lambda x: 1) :
	imgs = []
	for distr in distributions :
		acx = gd.AlphaComplex(points=distr).create_simplex_tree()
		acx.persistence()
		PI = gd.representations.PersistenceImage(bandwidth=variance, weight=f, resolution=[resolution,resolution])
		pi = PI.fit_transform([acx.persistence_intervals_in_dimension(dimension)])
		imgs.append((pi[0],label))
	return imgs 


def run_model_get_acc(train_model,test_model) :
	model = LogisticRegression()
	model.fit([a[0] for a in train_model], [a[1] for a in train_model])
	pred = model.predict(test_data)
	print(metrics.accuracy_score(test_labels,pred))
	return metrics.accuracy_score(test_labels,pred)

#effect of weighting fuctions
#linear
lin = lambda x: x[1]
#const
const = lambda x : 1
#soft arctangent
stan = lambda x : np.arctan(0.5*np.sqrt(x[1]))
#hard arctangent
htan = lambda x : np.arctan(x[1])

accuracies = []

#effect of variance
var = np.linspace(0,10,40)
accuracies = []
variances_lin_train = []
variances_const_train = []
variances_stan_train = []
variances_htan_train = []

variances_lin_test = []
variances_const_test = []
variances_stan_test = []
variances_htan_test = []
#torus = 1 sphere = 0

coords_lin = []
coords_const = []
coords_stan = []
coords_htan = []

for i in var :

	variances_lin_train.append(images_parameters(distributions=alpha_skeletons_train_torus,label=1,dimension=1,variance=i,f=lin))
	variances_const_train.append(images_parameters(distributions=alpha_skeletons_train_torus,label=1,dimension=1,variance=i,f=const))
	variances_stan_train.append(images_parameters(distributions=alpha_skeletons_train_torus,label=1,dimension=1,variance=i,f=stan))
	variances_htan_train.append([images_parameters(distributions=alpha_skeletons_train_torus,label=1,dimension=1,variance=i,f=htan)])

	variances_lin_test.append(images_parameters(distributions=alpha_skeletons_test_torus,label=1,dimension=1,variance=i,f=lin))
	variances_const_test.append(images_parameters(distributions=alpha_skeletons_test_torus,label=1,dimension=1,variance=i,f=const))
	variances_stan_test.append(images_parameters(distributions=alpha_skeletons_test_torus,label=1,dimension=1,variance=i,f=stan))
	variances_htan_test.append(images_parameters(distributions=alpha_skeletons_test_torus,label=1,dimension=1,variance=i,f=htan))

	variances_lin_train.append(images_parameters(distributions=alpha_skeletons_train_sphere,label=0,dimension=1,variance=i,f=lin))
	variances_const_train.append(images_parameters(distributions=alpha_skeletons_train_sphere,label=0,dimension=1,variance=i,f=const))
	variances_stan_train.append(images_parameters(distributions=alpha_skeletons_train_sphere,label=0,dimension=1,variance=i,f=stan))
	variances_htan_train.append([images_parameters(distributions=alpha_skeletons_train_sphere,label=0,dimension=1,variance=i,f=htan)])

	variances_lin_test.append(images_parameters(distributions=alpha_skeletons_test_sphere,label=0,dimension=1,variance=i,f=lin))
	variances_const_test.append(images_parameters(distributions=alpha_skeletons_test_sphere,label=0,dimension=1,variance=i,f=const))
	variances_stan_test.append(images_parameters(distributions=alpha_skeletons_test_sphere,label=0,dimension=1,variance=i,f=stan))
	variances_htan_test.append(images_parameters(distributions=alpha_skeletons_test_sphere,label=0,dimension=1,variance=i,f=htan))

	run_model_get_acc(variances_lin_train,variances_lin_test)
	run_model_get_acc(variances_const_train,variances_const_test)
	run_model_get_acc(variances_stan_train,variances_stan_test)
	run_model_get_acc(variances_htan_train,variances_htan_test)


#effect of resolution
res =  range(1,41)
resolutions_lin_train = []
resolutions_const_train = []
resolutions_stan_train = []
resolutions_htan_train = []

resolutions_lin_test = []
resolutions_const_test = []
resolutions_stan_test = []
resolutions_htan_test = []

for i in res :
	resolutions_lin_train.append(images_parameters(skeletons=alpha_skeletons_train_torus,label=1,dimension=1,resolution=i,f=lin))
	resolutions_const_train.append(images_parameters(skeletons=alpha_skeletons_train_torus,label=1,dimension=1,resolution=i,f=const))
	resolutions_stan_train.append(images_parameters(skeletons=alpha_skeletons_train_torus,label=1,dimension=1,resolution=i,f=stan))
	resolutions_htan_train.append(images_parameters(skeletons=alpha_skeletons_train_torus,label=1,dimension=1,resolution=i,f=htan))

	resolutions_lin_test.append(images_parameters(skeletons=alpha_skeletons_test_torus,label=1,dimension=1,resolution=i,f=lin))
	resolutions_const_test.append(images_parameters(skeletons=alpha_skeletons_test_torus,label=1,dimension=1,resolution=i,f=const))
	resolutions_stan_test.append(images_parameters(skeletons=alpha_skeletons_test_torus,label=1,dimension=1,resolution=i,f=stan))
	resolutions_htan_tes.append(images_parameters(skeletons=alpha_skeletons_test_torus,label=1,dimension=1,resolution=i,f=htan))
	
	resolutions_lin_train.append(images_parameters(skeletons=alpha_skeletons_train_sphere,label=0,dimension=1,resolution=i,f=lin))
	resolutions_const_train.append(images_parameters(skeletons=alpha_skeletons_train_sphere,label=0,dimension=1,resolution=i,f=const))
	resolutions_stan_train.append(images_parameters(skeletons=alpha_skeletons_train_sphere,label=0,dimension=1,resolution=i,f=stan))
	resolutions_htan_train.append(images_parameters(skeletons=alpha_skeletons_train_sphere,label=0,dimension=1,resolution=i,f=htan))

	resolutions_lin_test.append(images_parameters(skeletons=alpha_skeletons_test_sphere,label=0,dimension=1,resolution=i,f=lin))
	resolutions_const_test.append(images_parameters(skeletons=alpha_skeletons_test_sphere,label=0,dimension=1,resolution=i,f=const))
	resolutions_stan_test.append(images_parameters(skeletons=alpha_skeletons_test_sphere,label=0,dimension=1,resolution=i,f=stan))
	resolutions_htan_tes.append(images_parameters(skeletons=alpha_skeletons_test_sphere,label=0,dimension=1,resolution=i,f=htan))

resolutions_train = resolutions_lin_train + resolutions_const_train + resolutions_stan_train+ resolutions_htan_train
resolutions_test = resolutions_lin_test + resolutions_const_test + resolutions_stan_test + resolutions_htan_test

train_lin = list(zip(resolutions_lin_train,variances_lin_train))
train_const = list(zip(resolutions_const_train,variances_const_train))
train_stan = list(zip(resolutions_stan_train,variances_stan_train))
train_htan = list(zip(resolutions_htan_train,variances_htan_train))

test_lin = list(zip(resolutions_lin_test,variances_lin_test))
test_const = list(zip(resolutions_const_test,variances_const_test))
test_stan = list(zip(resolutions_stan_test,variances_stan_test))
test_htan = list(zip(resolutions_htan_test,variances_htan_test))


run_model_get_acc(train_lin,test_lin)
run_model_get_acc(train_const,test_const)
run_model_get_acc(train_stan,test_stan)
run_model_get_acc(train_htan,test_htan)





#weighted silhouette - investigate effects of change in weighting