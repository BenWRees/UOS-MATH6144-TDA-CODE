import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage

#
def dgm_pts(dgm) :
	points = [] 
	for i in dgm :
		i_pts = []
		for pts in i :
			(p,q) = pts[1]
			i_pts.append([p,q])
		points.append(i_pts)
	return points


#calculate distance matrix for collection of representations for the wasserstein metric of a certain distance
def distance_matrix_wasserstein(persistences,ord) :
	dgms = dgm_pts(persistences)
	#loop over each diagram
	dist_matrix = np.empty((len(dgms),len(dgms)), dtype=float )
	for i in range(0,len(dgms)) :
		for j in range(0,len(dgms)) :
			if i==j : 
				dist_matrix[i,j] = 0
			else :
				dist_matrix[i,j] = gd.wasserstein.wasserstein_distance(dgms[i],dgms[j],order=ord)
	return dist_matrix

#calculate distance matrix for collection of representations for the bottleneck distance
def distance_matrix_bottleneck(persistences) :
	dgm = dgm_pts(persistences)
	#loop over each diagram
	dist_matrix = np.zeros((len(dgm),len(dgm)))
	for i in range(0,len(dgm)) :
		for j in range(0,len(dgm)) :
			if i==j : 
				dist_matrix[i,j] = 0
			else :
				dist_matrix[i][j] = gd.bottleneck_distance(dgm[i],dgm[j])
	return dist_matrix

#generate 4 distributions

distribution = Methods.generate_norm_distributions(2,2,10,Methods.generate_rand_circs)


rips_complexes = Methods.Rips_complexes(distribution,10)
zero_skeletons = Methods.skeletons(rips_complexes,1)
zero_persistences = Methods.persistences(zero_skeletons)
one_skeletons = Methods.skeletons(rips_complexes,2)
one_persistences = Methods.persistences(one_skeletons)

print("persistence diagram: ",one_persistences)

print("distance matrix: ", distance_matrix_bottleneck(one_persistences))

