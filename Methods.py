import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import DataSetGen
from gudhi.representations import Landscape,Silhouette,PersistenceImage


def generate_uniform_dist(amount,num_pts,r) :
	distributions = []
	for i in range(amount) :
		X = np.empty([num_pts,2])
		x, y = np.random.uniform(), np.random.uniform()
		for i in range(num_pts):
			X[i,:] = [x, y]
			x = (X[i,0] + r * X[i,1] * (1-X[i,1])) % 1.
			y = (X[i,1] + r * x * (1-x)) % 1.
		distributions.append(X)
	return distributions



#need to generate around common point
def generate_rand_circs(amount,size) :
	circs = []
	for i in range(amount) :
		factor = np.random.randint(1,5)
		radius = factor*np.random.random()+1
		y_center = factor*np.random.random()
		x_center = factor*np.random.random()
		t = np.linspace(0,1,size)
		x,y = radius*np.cos(2*np.pi*t) + x_center, radius*np.sin(2*np.pi*t) + y_center
		circ = np.array([[h,v] for h,v in zip(x,y)])
		X = np.random.normal(loc=0,scale=np.random.random(),size=(size,2))
		n_circ = circ + X
		circs.append(n_circ)
	return circs
	#return [item for sublist in circs for item in sublist]

def generate_rand_tori(amount,size) :
	tori = []
	for i in range(amount) :
		factor = np.random.randint(1,5)
		radius = factor*np.random.random()+1
		tube_radius = factor*np.random.random()+1

		theta = np.linspace(0, 2.*np.pi, size)
		phi = np.linspace(0, 2.*np.pi, size)

		x,y,z = (radius+tube_radius*np.cos(theta))*np.cos(phi), (radius+tube_radius*np.cos(theta))*np.sin(phi), tube_radius*np.sin(theta) 
		tor = np.array([[h,v,w] for h,v,w in zip(x,y,z)])
		X = np.random.normal(loc=0,scale=np.random.random(),size=(size,3))
		n_tor = tor + X
		tori.append(n_tor)
	return tori



def generate_norm_distributions(num_shapes,amount,size,shape) :
	distributions = []
	for i in range(amount) :
		#random circles
		shapes = shape(np.random.randint(1,num_shapes+1),size)
		#add circles to one data set
		distributions.append(np.concatenate(shapes))
	return distributions

"""
	Function generates VR complexes for a user-chosen list of plots
	parameters: 
		distributions - a list of lists that represent plots
		edge_length - the edge length defined in the VR filtration
	return - a list of VR filtrations for each plot in distributions
"""
def Rips_complexes(distributions, edge_length) :
	complexes = []
	for dist in distributions :
		complx = gd.RipsComplex(points = dist, max_edge_length = edge_length)
		complexes.append(complx)
	return complexes 

def Alpha_complexes(distributions) :
	complexes = []
	for dist in distributions :
		complx = gd.AlphaComplex(points = dist).create_simplex_tree()
		complx.persistence()
		complexes.append(complx)
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
		skeleton.persistence()
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


def persistence_silhouettes(skeletons,dim,k,pow) :
	silhouettes = []
	for i in skeletons :
		#denom = []
		#for pt in dgm :
 		#	denom.append(np.power(np.absolute(dgm[1]-dgm[0]),pow))

		l = Silhouette(resolution=(k*10),weight= lambda x : np.power(np.absolute(x[0]-x[1]),pow))
		L = l.fit_transform([i.persistence_intervals_in_dimension(dim)])
		#print(L)
		silhouettes.append(L)
	return silhouettes

def persistence_images(skeletons,dim,res,var=1) :
	images = []
	for i in skeletons :
		PI = PersistenceImage(bandwidth=var, weight=lambda x: x[1]**2, im_range=[0,.004,0,.004], resolution=[res,res])
		pi = PI.fit_transform([i.persistence_intervals_in_dimension(dim)])
		images.append(pi)
	return images


def mean_persistence(representations) :
	return np.mean(representations,axis=0)[0]



def dgm_pts(skeletons,dim) :
	points = []
	for skelly in skeletons :
		points.append(skelly.persistence_intervals_in_dimension(dim))
	return points



#calculate distance matrix for collection of representations for the wasserstein metric of a certain distance
def distance_matrix_wasserstein(dgms,ord) :
	#loop over each diagram
	dist_matrix = np.empty((len(dgms),len(dgms)), dtype=float )
	for i in range(0,len(dgms)) :
		for j in range(0,len(dgms)) :
			print("i: ", i," " ,dgms[i])
			print("j: ", j," ",dgms[j])
			if i==j : 
				dist_matrix[i,j] = 0
			else :
				#continue
				dist_matrix[i,j] = wasserstein_distance(dgms[i],dgms[j],order=ord)[0]
			print("dist: ", dist_matrix[i][j])

	return dist_matrix

#calculate distance matrix for collection of representations for the bottleneck distance
def distance_matrix_bottleneck(dgm) :
	#loop over each diagram
	dist_matrix = np.zeros((len(dgm),len(dgm)))
	for i in range(0,len(dgm)) :
		for j in range(0,len(dgm)) :
			if i==j : 
				dist_matrix[i,j] = 0
			else :
				dist_matrix[i][j] = gd.bottleneck_distance(dgm[i],dgm[j])
	return dist_matrix


