import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import DataSetGen
import StatAnal
from gudhi.representations import Landscape,Silhouette,PersistenceImage

#generate 10 distributions of 2 tori
distributions = DataSetGen.generate_norm_distributions(2,2,1000,DataSetGen.generate_rand_circs)
"""
for i in distributions :
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	#ax1 = fig.add_subplot(121, projection='3d')
	ax.scatter(i[:,0], i[:,1], i[:,2])
	ax.set_zlim(-3,3)
	ax.view_init(36, 116)
	plt.show()
"""
for i in distributions :
	#fig, ax = plt.subplots(subplot_kw={"projection": "2d"})
	#ax1 = fig.add_subplot(121, projection='3d')
	plt.scatter(i[:,0], i[:,1])
	#ax.set_zlim(-3,3)
	#ax.view_init(36, 116)
	plt.show()

#acX = gd.AlphaComplex(points=distributions[0]).create_simplex_tree()
#dgmX = acX.persistence()
#gd.plot_persistence_diagram(dgmX)
#plt.show()

#create a simplex tree for each data set
def complexes(distributions) :
	complexes = []
	for i in distributions :
		cmplx = gd.AlphaComplex(points=i).create_simplex_tree()
		cmplx.persistence()
		complexes.append(cmplx)
	return complexes


def persistenc_imgs(complexes) :
	imgs = []
	for acx in complexes :
		PI = gd.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2,im_range=[0,.004,0,.004], resolution=[100,100])
		pi = PI.fit_transform([acx.persistence_intervals_in_dimension(1)])
		imgs.append(pi)
	return imgs 

list_of_complexes = complexes(distributions)
list_of_images = persistenc_imgs(list_of_complexes)

def show_imags(list_of_images) :
	for pi in list_of_images :
		plt.imshow(np.flip(np.reshape(pi[0], [100,100]), 0))
		plt.title("Persistence Image")
		plt.show()

show_imags(list_of_images)