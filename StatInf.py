import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage, WassersteinDistance
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import ttest_ind


distribution = Methods.generate_norm_distributions(4,4,100,Methods.generate_rand_tori)

rips_complexes = Methods.Rips_complexes(distribution,10)
alpha_complex = Methods.Alpha_complexes(distribution)
zero_skeletons = Methods.skeletons(rips_complexes,1)
zero_persistences = Methods.persistences(zero_skeletons)
one_skeletons = Methods.skeletons(rips_complexes,2)
one_persistences = Methods.persistences(one_skeletons)

#takes n persistence diagrams and splits them into two sets n1 and n2
def rand_labelling(diagrams) :
	#shuffle list 
	diagram_pos = [i for i in range(0,len(diagrams))]
	split_val = np.random.randint(1,len(diagrams))
	np.random.shuffle(diagram_pos)

	set_1 =  diagram_pos[:split_val:]
	#print(set_1)
	set_2 = diagram_pos[split_val::]
	#print([set_1,set_2])
	return [set_1,set_2]

#return false if the labels are not equal and true if all elements are equal
def check_valid(labelling1, labelling2) :
	#check first labelling 
	label1_1 = labelling1[0]
	label2_1 = labelling2[0]
	for i,j in zip(label1_1,label2_1) :
		if i != j :
			return False
		else :
			continue

	#check first labelling 
	label1_2 = labelling1[1]
	label2_2 = labelling2[1]
	for i,j in zip(label1_2,label2_2) :
		if i != j :
			return False
		else :
			continue

	return True 

def construct_labels(labelling,dgms) :
	#print("dgms: ",dgms)
	labelling1 = labelling[0]
	labelling2 = labelling[1]
	#print("label 1: ",labelling1)
	#print("label 2: ",labelling2)

	label1 = []
	label2 = []
	for i in labelling1 :
		#print("i dgm: ",dgms[i])
		label1.append(dgms[i])
	for i in labelling2 :
		label2.append(dgms[i])
	#print([label1,label2])
	return [label1,label2]

#Joint Loss function method of Diagrams
def joint_loss(label1,label2,dim,p=1,q=1) :
	sum1 = 0
	sum2 = 0
	denom1 = 2*len(label1)*(len(label1)-1)
	denom2 = 2*len(label2)*(len(label2)-1)

	#Wassersteins for all Label 1s
	for i in label1 :
		for j in label1:
			k = WassersteinDistance(order=q)
			sum1 += np.power(k(i.persistence_intervals_in_dimension(dim),j.persistence_intervals_in_dimension(dim)),p)

	#Wassersteins for all Label 2s
	for i in label2 :
		for j in label2:
			k = WassersteinDistance(order=q)
			sum2 += np.power(k(i.persistence_intervals_in_dimension(dim),j.persistence_intervals_in_dimension(dim)),p)
	return denom1*sum1 + denom2*sum2

def unbiased_estimator(skeletons,dim,observed_labelling,repetitions,loss_func,p,q) :
	Z = 0
	constructed_dgm1,construct_dgm2 = construct_labels(observed_labelling,skeletons)
	#print(diagrams)

	observed_loss = loss_func(constructed_dgm1,construct_dgm2,p,q)

	for i in range(0,repetitions) :
		shuffled_labels = rand_labelling(skeletons)
		shuffled_dgms = construct_labels(rand_labelling(skeletons),skeletons)
		#print(shuffled_labels)
		#ensure shuffle is a new labelling to observed labelling
		while check_valid(observed_labelling,shuffled_labels) :
			#print("same")
			shuffled_labels = rand_labelling(skeletons)
			shuffled_dgms = construct_labels(shuffled_labels,skeletons)
			#print(shuffled_labels)

		loss = loss_func(shuffled_dgms[0], shuffled_dgms[1],p,q)
		if loss <= observed_loss :
			Z += 1
	return Z/repetitions

def p_value(skeletons,dim,observed_labelling,repetitions,loss_func,p,q) :
	Z = unbiased_estimator(skeletons,dim,observed_labelling,repetitions,loss_func,p,q)
	return (repetitions*Z+1)/(repetitions+1)



#Persistence Landscape Hypothesis Testing

#functional 
def function(k,b,landscapes,K) :
	B = np.max([np.max(a) for a in landscapes])
	if k <= K and (b >= -B and b <= B) :
		return 1
	return 0


def landscape_random_variable(landscapes,f,K) :
	#take the wasserstein distance between f*landscapes_pts and [[0,0],...,[0,0]]
	zero_dgm = [[0,0] for a in range(0,len(landscapes[0][0]))]
	ks = [a for a in range(0,len(landscapes[0][0]),1)]

	B = np.max([np.max(a) for a in landscapes])
	bs = np.linspace(-B,B,len(landscapes[0][0]))

	#calculate values of f over the domain [0,K]x[-B,B]
	#f_vals = [f(k,b,landscapes,K) for k,b in zip(ks,bs)]
	f_vals = [f(k,b,landscapes,K) for b,k in zip(bs,ks)]

	#apply f_vals to every landscape 
	modified_land = [np.multiply(f_vals,i[0]) for i in landscapes]
	#print(modified_land)
	#rewrite f\Gamma as [[x1,y1],[x2,y2],...,[xn,yn]] 
	#where x are values in range [0,....,len(landscape[i]))]
	#and y are our modified_land values
	modified_pts = []
	#wrap to be suitable for distance
	for i in modified_land :
		vals = [c for c in range(0,len(i))]
		#print(vals)
		modified_pts.append([[a,b] for a,b in zip(vals,i)])
	#print(modified_pts)
	
	#return list of distances that is the norm of all possible values for the random variable
	#return [wasserstein_distance(np.array(zero_dgm),np.array(j),order=1) for j in modified_pts]
	return [np.linalg.norm(np.array(j),ord=1) for j in modified_land]


def two_test(landscapes,labelling,crit_val) :
	landscape_labelling = construct_labels(labelling,landscapes)

	data1 = landscape_random_variable(landscape_labelling[0],function,3)
	data2 = landscape_random_variable(landscape_labelling[1],function,3)
	stat, p = ttest_ind(data1, data2)
	#print('stat=%.3f, p=%.3f' % (stat, p))
	if p > crit_val or p < (1-crit_val) :
		print('Probably the same distribution')
	else:
		print('Probably different distributions')
	return p

def two_test_trad(distributions,labelling,crit_val) :
	data1,data2 = construct_labels(labelling,distributions)

	stat, p = ttest_ind(data1, data2)
	#print('stat=%.3f, p=%.3f' % (stat, p))
	if p > crit_val :
		print('Probably the same distribution')
	else:
		print('Probably different distributions')
	return p

for i in range(1,11) :
	print("TEST ", i)
	distribution = Methods.generate_norm_distributions(4,4,100,Methods.generate_rand_tori)

	rips_complexes = Methods.Rips_complexes(distribution,10)
	alpha_complex = Methods.Alpha_complexes(distribution)
	zero_skeletons = Methods.skeletons(rips_complexes,1)
	zero_persistences = Methods.persistences(zero_skeletons)
	one_skeletons = Methods.skeletons(rips_complexes,2)
	one_persistences = Methods.persistences(one_skeletons)
	landscapes = Methods.persistence_landscapes(one_skeletons,1,3)

	p_val_land = two_test(landscapes,[[0,2],[1,3]],0.05)

	p_val_dgm = unbiased_estimator(one_skeletons,1,[[0,2],[1,3]],100,joint_loss,1,2)
	print('landscape: ',p_val_land)
	print('diagrams: ',p_val_dgm)

	print("land + dgm: ", p_val_dgm+p_val_land)
	print("difference: ", np.absolute(p_val_dgm - p_val_land))

"""
each persistence image has the form 
np.flip(np.reshape(i[0], [res,res]), 0) - is a resolution x resolution matrix
"""

#zero_images = Methods.persistence_images(alpha_complex,2,10,0.3)


def transform_dgm(diagram_pts) :
	pass

#Persistence Images Hypothesis Testing
# n - number of distributions
# m - resolution
def images_hypothesis_test(images, label, threshold) :
	n = len(images)
	m = int(np.sqrt(len(images[0][0])))
	print(n)
	print(m)
	V = np.zeros((np.power(m,2)))
	#populate V to 2d vectors of points on vectorised dgm
	#V =
	#filt
	V = filter(lambda x : x[0] >= x[1],V)
	#populate loc_v with all 

	m_num = (m*(m+1))/2
	T = np.zeros((m_num))
	#filtering stage
	for i in range (0,m_num) :
		T[i] = np.mean(V(i))

	c_percentile = np.percentile(T,threshold)
	V = filter(lambda x: x>c_percentile ,loc_V)

	Z = np.zeros((n))

	#testing
	for j in range(1,n) :
		v_1 = V[0][j]
		v_2 = V[1][j]
		#conduct hypothesis test
		Z[j]

#images_hypothesis_test(alpha_complex,labelling,10)





