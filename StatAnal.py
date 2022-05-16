import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import DataSetGen
from gudhi.representations import Landscape,Silhouette,PersistenceImage


"""
	Function generates VR complexes for a user-chosen list of plots
	parameters: 
		distributions - a list of lists that represent plots
		edge_length - the edge length defined in the VR filtration
	return - a list of VR filtrations for each plot in distributions
"""
def complexes(distributions, edge_length) :
	complexes = []
	for dist in distributions :
		complex = gd.RipsComplex(points = dist, max_edge_length = edge_length)
		complexes.append(complex)
	return complexes 

"""
	Function generates VR skeletons for a user-chosen list of plots
	parameters: 
		complexes - a list of VR filtrations
		dimension - the largest dimension of the complex
	return - a list of VR skeletons for each plot in distributions
"""
def skeletons(complexes, dimension) : 
	skeletons = []
	for complx in complexes : 
		skeleton = complx.create_simplex_tree(max_dimension=dimension)
		skeletons.append(skeleton)
	return skeletons

def persistences(skeletons) :
	persistences = []
	for i in skeletons :
		persistences.append(i.persistence())
	return persistences

#Compute persistence Landscape mean
def persistence_landscapes(skeletons,dim,k) :
	landscapes = []
	for i in skeletons :
		l = Landscape(num_landscapes=k,resolution=10)
		L = l.fit_transform([i.persistence_intervals_in_dimension(dim)])
		#print(L)
		landscapes.append(L)
	return landscapes


def persistence_silhouettes(skeletons,dim,pow) :
	silhouettes = []
	for i in skeletons :
		#denom = []
		#for pt in dgm :
 		#	denom.append(np.power(np.absolute(dgm[1]-dgm[0]),pow))

		l = Silhouette(resolution=10,weight= lambda x : np.power(np.absolute(x[0]-x[1]),pow))
		L = l.fit_transform([i.persistence_intervals_in_dimension(dim)])
		#print(L)
		silhouettes.append(L)
	return silhouettes

def persistence_images(skeletons,dim) :
	images = []
	for i in skeletons :
		PI = PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1]**2, im_range=[0,.004,0,.004], resolution=[100,100])
		pi = PI.fit_transform(i.persistence_intervals_in_dimension(dim))
		images.append(pi)
	return images


def mean_landscape(landscapes, t) :
	mean = 0
	for j in range(0,len(landscapes)) :
		#print(j)
		for i in landscapes[j] :
			#print(i)
			mean += i[t]

	return mean/len(landscapes)

def mean_image(images,t) :
	
	

def accuracy(true_val,obs_val) :
	return 100*(1-np.absolute(true_val-obs_val)/true_val)


distributions = DataSetGen.generate_norm_distributions(2,10,100,DataSetGen.generate_rand_circs)
print(distributions)

#Compute statistical measures
true_mean = [a for a in map(np.mean, distributions)]
true_variance = [a for a in map(np.var,distributions)]


complexes = complexes(distributions,12)
#compute 0-dim PD using VR
zero_skeletons = skeletons(complexes,1)
zero_persistences = persistences(zero_skeletons)

#compute 1-dim PD using VR
one_skeletons = skeletons(complexes,2)
one_persistences = persistences(one_skeletons)


#Statistical means of each vectorisation technique

#calculate landscapes over t of lambda_k
one_landscapes = persistence_landscapes(one_skeletons,1,3)

print("true mean",np.mean(true_mean))

#caluclate mean at lambda(3,4)
landscape_mean = mean_landscape(one_landscapes,4)
print("landscape mean",landscape_mean)


#compute Silhouette mean
one_silhouettes_low = persistence_silhouettes(one_skeletons,1,1)
one_silhouettes_high = persistence_silhouettes(one_skeletons,1,200)

silhouette_mean_low = mean_landscape(one_silhouettes_low,4)
silhouette_mean_high = mean_landscape(one_silhouettes_high,4)
print("silhouette mean with low weighting",silhouette_mean_low)
print("silhouette mean with high weighting",silhouette_mean_high)


#compute persistence image mean
#persistence_images = persistence_images(one_skeletons,1)
"""
for i in persistence_images :
	plt.imshow(np.flip(np.reshape(i[0], [100,100]), 0))
	plt.title("Persistence Image")
"""
#Accuracy of measures
print("accuracy of landscape",accuracy(np.mean(true_mean),landscape_mean))
print("accuracy of silhouette (low)",accuracy(np.mean(true_mean),silhouette_mean_low))
print("accuracy of silhouette (high)",accuracy(np.mean(true_mean),silhouette_mean_high))



