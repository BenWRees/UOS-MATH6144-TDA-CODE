import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import DataSetGen
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from gudhi.wasserstein.barycenter import lagrangian_barycenter
import csv 


def populate(file) :
	data = []
	with open(file) as f :
		csv_reader = csv.reader(f, delimiter=',')
		for row in csv_reader :
			if len(row) != 0 :
				value_1, value_2 = float(row[0]),float(row[1])
				data.append([value_1,value_2])
	#add in diagonal points 
	#create_diagonal(data)
	return np.array(data)


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
def generate_circs(size) :
	t = np.linspace(0,1,size)
	x,y = np.cos(2*np.pi*t), np.sin(2*np.pi*t)
	circ = np.array([[h,v] for h,v in zip(x,y)])
	X = np.random.uniform()
	n_circ = circ + X
	return n_circ

#need to generate around common point
def generate_rand_circs(amount,size,r) :
	circs = []
	for i in range(amount) :
		factor = np.random.randint(1,5)
		radius = factor*np.random.random()+1
		y_center = factor*np.random.random()
		x_center = factor*np.random.random()
		t = np.linspace(0,1,size)
		x,y = radius*np.cos(2*np.pi*t) + x_center, radius*np.sin(2*np.pi*t) + y_center
		circ = np.array([[h,v] for h,v in zip(x,y)])
		vals = np.linspace(0,r,1000)
		X = np.random.normal(loc=np.random.randint(1,r+1),scale=vals[np.random.randint(0,len(vals))],size=(size,2))
		n_circ = circ + X
		circs.append(n_circ)
	return circs

def generate_rand_spheres(amount,size,r) :
	spheres = []
	for i in range(0,amount) :
		factor = np.random.randint(1,5)
		radius = factor*np.random.random()+1
		theta = np.linspace(0, 2.*np.pi, size)
		phi = np.linspace(0, 2.*np.pi, size)
		x,y,z = radius*np.cos(theta)*np.sin(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(theta) 
		sphr = np.array([[h,v,w] for h,v,w in zip(x,y,z)])
		X = np.random.normal(loc=np.random.randint(1,r+1),scale=vals[np.random.randint(0,len(vals))],size=(size,2))
		n_sphr = sphr + X
		spheres.append(n_sphr)
	return spheres 

def generate_rand_tori(amount,size,r) :
	tori = []
	for i in range(amount) :
		factor = np.random.randint(1,5)
		radius = factor*np.random.random()+1
		tube_radius = factor*np.random.random()+1

		theta = np.linspace(0, 2.*np.pi, size)
		phi = np.linspace(0, 2.*np.pi, size)

		x,y,z = (radius+tube_radius*np.cos(theta))*np.cos(phi), (radius+tube_radius*np.cos(theta))*np.sin(phi), tube_radius*np.sin(theta) 
		tor = np.array([[h,v,w] for h,v,w in zip(x,y,z)])
		vals = np.linspace(0,r,1000)
		X = np.random.normal(loc=np.random.randint(1,r+1),scale=vals[np.random.randint(0,len(vals))],size=(size,3))
		n_tor = tor + X
		tori.append(n_tor)
	return tori



def generate_norm_distributions(num_shapes,amount,size,shape,r) :
	distributions = []
	for i in range(amount) :
		#random circles
		shapes = shape(np.random.randint(1,num_shapes+1),size,r)
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

#Compute persistence Landscape in 3d 
#returns list of tuples (k,t,lambda(k,t))
def persistence_landscapes_3d(skeletons,dim,K) :
	landscapes = []
	for i in skeletons :
		k_landscape = []
		for k in range(1,K) :
			#print("k value: ",k)
			l = Landscape(num_landscapes=k,resolution=10)
			L = l.fit_transform([i.persistence_intervals_in_dimension(dim)])[0]
			ts = [a for a in range(len(L))]
			values = [[k,t,l] for t,l in zip(ts,L)]
			k_landscape = k_landscape + values
			#print('landscape coords:', values )
		landscapes.append(k_landscape)
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

def mean_persistence_curve(skeletons,dim) :
	curves = []
	for i in skeletons :
		dgm_pts = i.persistence_intervals_in_dimension(dim)
		print("diagram points: ",dgm_pts)
		filtrations = [j[1] for j in tuple(i.get_filtration()) ]
		print("filtrations: ", filtrations)
		pts = []
		for t in filtrations :
			fund_box = [a for a in dgm_pts if (a[0] <= t) and (a[1] > t)]
			fund_box.remove([ 0., np.inf])
			if len(fund_box) == 0 :
				fund_box.append(0)
			print("fundamental box at ",t, ": ",fund_box)
			pts.append(np.mean(fund_box))
		curves.append(pts)
	return curves

"""circle = generate_circs(100)
print(circle)
circ_complex = gd.RipsComplex(points=circle,max_edge_length=2).create_simplex_tree()
circ_complex.persistence()
curve = mean_persistence_curve([circ_complex],0)
print(curve)
plt.plot(curve[0])
plt.show()"""






"""def frechet_mean(skeletons,dim) :
	persistences(skeletons)
	diagrams_inf = [i.persistence_intervals_in_dimension(dim) for i in skeletons]
	ess = []
	diagram_inf.remove(a)
	return lagrangian_barycenter(pdiagset=diagrams,init=None,verbose=False)"""


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


