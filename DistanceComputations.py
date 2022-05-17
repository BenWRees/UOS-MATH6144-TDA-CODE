import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
from gudhi.wasserstein import wasserstein_distance

#
"""def dgm_pts(dgm) :
	points = [] 
	for i in dgm :
		i_pts = []
		for pts in i :
			(p,q) = pts[1]
			if (p == 'inf') : 
				i_pts.append([np.inf,q])
			elif (q=='inf') :
				i_pts.append([p,np.inf])
			else :
				i_pts.append([p,q])
	return points"""
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
			if i==j : 
				dist_matrix[i,j] = 0
			else :
				#continue
				dist_matrix[i,j] = wasserstein_distance(dgms[i],dgms[j],order=ord)

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


#wraps curves into appropriate form
def wrap_curves(curves) :
	pass


distribution = Methods.generate_norm_distributions(2,4,10,Methods.generate_rand_circs)

rips_complexes = Methods.Rips_complexes(distribution,10)
zero_skeletons = Methods.skeletons(rips_complexes,1)
zero_persistences = Methods.persistences(zero_skeletons)
one_skeletons = Methods.skeletons(rips_complexes,2)
one_persistences = Methods.persistences(one_skeletons)


#print(dgm_pts(one_skeletons,0))

print(distance_matrix_wasserstein(dgm_pts(one_skeletons,0),2))
