import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import DataSetGen
import StatAnal
from gudhi.representations import Landscape,Silhouette,PersistenceImage

#generate 10 data sets of between 1-4 Tori each of 1000 pts 
distributions = DataSetGen.generate_norm_distributions(4,2,1000,DataSetGen.generate_rand_tori)
"""
for i in distributions :
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	#ax1 = fig.add_subplot(121, projection='3d')
	ax.scatter(i[:,0], i[:,1], i[:,2])
	ax.set_zlim(-3,3)
	ax.view_init(36, 116)
	plt.show()
"""
alpha_skeletons = StatAnal.Alpha_complexes(distributions)

def plot_imgs(list_of_images) :
	for pi in list_of_images :
		plt.imshow(np.flip(np.reshape(pi[0], [len(pi[0]),len(pi[0])]), 0))
		plt.title("Persistence Image")
		plt.show()

#Persistence Images - investigate effects of resolution, distribution and weighting function 
def images_parameters(skeletons, dimension, resolution=20,variance=1.0, f=lambda x: 1) :
	imgs = []
	for acx in skeletons :
		PI = gd.representations.PersistenceImage(bandwidth=variance, weight=f, resolution=[resolution,resolution])
		pi = PI.fit_transform([acx.persistence_intervals_in_dimension(dimension)])
		imgs.append(pi)
	return imgs 

#effect of weighting fuctions
#linear
lin = lambda x: x[1]
#const
const = lambda x : 1
#soft arctangent
stan = lambda x : np.arctan(0.5*np.sqrt(x))
#hard arctangent
htan = lambda x : np.arctan(x[1])


#effect of variance
var = np.linspace(0,1,10)
variances_lin = []
variances_const = []
variances_stan = []
variances_htan = []
variances_lin.append(images_parameters(skeletons=alpha_skeletons,dimension=1,variance=0.1,f=lin))
for i in var :
	variances_lin.append(images_parameters(skeletons=alpha_skeletons,dimension=1,variance=i,f=lin))
	variances_const.append(images_parameters(skeletons=alpha_skeletons,dimension=1,variance=i,f=const))
	#variances_stan.append(images_parameters(skeletons=alpha_skeletons,dimension=1,variance=i,f=stan))
	#variances_htan.append(images_parameters(skeletons=alpha_skeletons,dimension=1,variance=i,f=htan))

#effect of resolution
res =  range(10,110,10)
resolutions_lin = []
resolutions_const = []
resolutions_stan = []
resolutions_htan = []
for i in res :
	resolutions_lin.append(images_parameters(skeletons=alpha_skeletons,dimension=1,resolution=i,f=lin))
	resolutions_const.append(images_parameters(skeletons=alpha_skeletons,dimension=1,resolution=i,f=const))
	#resolutions_stan.append(images_parameters(skeletons=alpha_skeletons,dimension=1,resolution=i,f=stan))
	#resolutions_htan.append(images_parameters(skeletons=alpha_skeletons,dimension=1,resolution=i,f=htan))

plot_imgs(variances_lin)
plot_imgs(variances_const)
#plot_imgs(variances_stan)
#plot_imgs(variances_htan)


plot_imgs(resolutions_lin)
plot_imgs(resolutions_const)
#plot_imgs(resolutions_stan)
#plot_imgs(resolutions_htan)




#weighted silhouette - investigate effects of change in weighting