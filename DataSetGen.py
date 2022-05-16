import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import datasets

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
		shapes = shape(np.random.randint(1,num_shapes),size)
		#add circles to one data set
		distributions.append(np.concatenate(shapes))
	return distributions

"""
distr = generate_norm_distributions(5,10,1000,generate_rand_tori)
for i in distr :
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	#ax1 = fig.add_subplot(121, projection='3d')
	ax.scatter(i[:,0], i[:,1], i[:,2])
	ax.set_zlim(-3,3)
	ax.view_init(36, 116)
	plt.show()
"""