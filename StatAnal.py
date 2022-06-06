import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from gudhi.wasserstein.barycenter import lagrangian_barycenter

"""
	Function generates VR complexes for a user-chosen list of plots
	parameters: 
		distributions - a list of lists that represent plots
		edge_length - the edge length defined in the VR filtration
	return - a list of VR filtrations for each plot in distributions
"""
def Rips_complexes(distributions, edge_length=10) :
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

#Compute persistence Landscape 2d
def persistence_landscapes_2d(skeletons,dim,k) :
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
	return (np.mean(representations,axis=0)[0])

def frechet_mean(skeletons,dim) :
	Methods.persistences(skeletons)
	diagrams = [i.persistence_intervals_in_dimension(dim) for i in skeletons]
	return lagrangian_barycenter(pdiagset=diagrams,init=None)

	
		

distributions = Methods.generate_norm_distributions(2,4,100,Methods.generate_rand_circs,2)

"""for i in distributions :
	plt.scatter(i[:,0],i[:,1])
	plt.show()
"""#print(distributions)


rips_complexes = Rips_complexes(distributions,12)
alpha_skeletons = Alpha_complexes(distributions)
#compute 0-dim PD using VR
zero_skeletons = skeletons(rips_complexes,1)
zero_persistences = persistences(zero_skeletons)

#compute 1-dim PD using VR
one_skeletons = skeletons(rips_complexes,2)
one_persistences = persistences(one_skeletons)
gd.plot_persistence_diagram(one_persistences[1])
plt.show()

"""for i in zero_persistences :
	gd.plot_persistence_diagram(i)
	plt.title("zero persistence")
	plt.show()

for i in one_persistences :
	gd.plot_persistence_diagram(i)
	plt.title("one persistence")
	plt.show()"""

#print(true_mean(zero_skeletons,0))

#Statistical means of each vectorisation technique

#calculate landscapes over t of lambda_k
zero_landscapes = persistence_landscapes_2d(zero_skeletons,0,3)
one_landscapes = persistence_landscapes_2d(one_skeletons,1,3)

one_landscapes_3d = persistence_landscapes_3d(one_skeletons,1,6)[0]



#plt.plot(x_cords,y_cords,z_cords)
#plt.show()
#print("true mean",true_mean)


#caluclate mean at lambda(3,4)
landscape_mean_one = mean_persistence(one_landscapes)
landscape_mean_zero = mean_persistence(zero_landscapes)

#add a legend

"""plt.plot(landscape_mean_zero, '--', color="red")
for i in zero_landscapes[:5] :
	plt.plot(i[0])
plt.show()"""

print('landscape 2d vals: ',one_landscapes)
plt.plot(landscape_mean_one, '--', color="red",label="mean Landscape")
count = 0 
for i in one_landscapes :
	title = "Landscape " + str(count)
	plt.plot(i[0],label=title)
plt.legend()
plt.show()



"""
#compute Silhouette mean
one_silhouettes_low = persistence_silhouettes(one_skeletons,1,3,1)
one_silhouettes_high = persistence_silhouettes(one_skeletons,1,3,200)




silhouette_mean_low = mean_persistence(one_silhouettes_low)
silhouette_mean_high = mean_persistence(one_silhouettes_high)
print("silhouette mean with low weighting",silhouette_mean_low)
print("silhouette mean with high weighting",silhouette_mean_high)


plt.plot(silhouette_mean_high,'--', color="red")
for i in one_silhouettes_high[:5] :
	plt.plot(i[0])
plt.show()



#compute persistence image mean
persistence_images = persistence_images(alpha_skeletons,1,100,0.23)
images_mean = mean_persistence(persistence_images)


plt.imshow(np.flip(np.reshape(images_mean, [100,100]), 0))
plt.title("Mean Persistence Image")
plt.show()

for i in persistence_images :
	plt.imshow(np.flip(np.reshape(i[0], [100,100]), 0))
	plt.title("Persistence Image")
	plt.show()


def accuracy(observed) :
	true = np.mean(true_mean)
	return 100*(1-(np.absolute(true-observed)/true))
"""
#andscape_mean_acc = list(map(accuracy, landscape_mean))
#silhouette_mean_high_acc = list(map(accuracy, silhouette_mean_high))
#silhouette_mean_low_acc = list(map(accuracy, silhouette_mean_low))

#Accuracy of measures
#print("accuracy of landscape",landscape_mean_acc)
#print("accuracy of silhouette (low)",silhouette_mean_low_acc)
#print("accuracy of silhouette (high)",silhouette_mean_high_acc)

#plt.plot(range(0,len(landscape_mean_acc)),landscape_mean_acc, color = "red")
#plt.plot(range(0,len(silhouette_mean_low)),silhouette_mean_low, color='blue')
#plt.plot(range(0,len(silhouette_mean_high)),silhouette_mean_high, color = 'yellow')
#plt.show()

#calculate distance matrices
