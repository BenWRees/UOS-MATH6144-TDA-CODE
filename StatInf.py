import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage, WassersteinDistance
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import ttest_ind #,permutation_test
import csv
from statsmodels.stats.multitest import multipletests


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
	L = []
	for i in range(0,repetitions) :
		shuffled_labels = rand_labelling(skeletons)
		shuffled_dgms = construct_labels(rand_labelling(skeletons),skeletons)
		#ensure shuffle is a new labelling to observed labelling
		while check_valid(observed_labelling,shuffled_labels) :
			#print("same")
			shuffled_labels = rand_labelling(skeletons)
			shuffled_dgms = construct_labels(shuffled_labels,skeletons)

		loss = loss_func(shuffled_dgms[0], shuffled_dgms[1],p,q)
		L.append(loss)
		if loss <= observed_loss :
			Z += 1
	Z_2 = np.sum(list(filter(lambda p : p < observed_loss,L)))/repetitions
	return (Z/repetitions)

def p_value(skeletons,dim,observed_labelling,repetitions,loss_func,p,q) :
	Z = unbiased_estimator(skeletons,dim,observed_labelling,repetitions,loss_func,p,q)
	return (repetitions*Z+1)/(repetitions+1)



#Persistence Landscape Hypothesis Testing

#functional 
def function(k,b,B,K) :
	if k <= K and (b >= -B and b <= B) :
		return 1
	return 0

#ladscapes is a list of values
def landscape_random_variable(landscapes,f,K) :
	functional_val = []
	for i in landscapes :
		B = np.max([j[1] for j in i])
		f_vals = [f(j[0],j[1],B,K) for j in i]
		mod_landscape_norm = np.linalg.norm(np.multiply(f_vals,[a[2] for a in i]),1)
		functional_val.append(mod_landscape_norm)
	return functional_val


"""def landscape_random_variable(landscapes,f,K) :
	#take the wasserstein distance between f*landscapes_pts and [[0,0],...,[0,0]]
	mean_land = Methods.mean_persistence(landscapes)
	zero_dgm = [[0,0] for a in range(0,len(landscapes[0][0]))]
	ks = [a for a in range(0,len(landscapes[0][0]),1)]

	B = np.max([np.max(a) for a in landscapes])
	bs = np.linspace(-B,B,len(landscapes[0][0]))

	#calculate values of f over the domain [0,K]x[-B,B]
	#f_vals = [f(k,b,landscapes,K) for k,b in zip(ks,bs)]
	f_vals = [f(k,b,landscapes,K) for b,k in zip(bs,ks)]

	#apply f_vals to every landscape 
	modified_land = [np.multiply(f_vals,i) for i in mean_land]
	#print(modified_land)
	#rewrite fGamma as [[x1,y1],[x2,y2],...,[xn,yn]] 
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
	return [np.linalg.norm(np.array(j),ord=1) for j in modified_land]"""


def two_test(landscapes,labelling,crit_val) :
	landscape_labelling = construct_labels(labelling,landscapes)

	data1 = landscape_random_variable(landscape_labelling[0],function,3)
	data2 = landscape_random_variable(landscape_labelling[1],function,3)

	#as random variables satisy central limit theorem then variables satisfy clt

	stat, p = ttest_ind(data1, data2)
	#print('stat=%.3f, p=%.3f' % (stat, p))
	if p > crit_val  :
		#print(p)
		print('Probably the same distribution')
	else:
		#print(p)
		print('Probably different distributions')
	return p

def two_test_trad(distributions,labelling,crit_val) :
	print(distributions)
	data1 = [distributions[i] for i in labelling[0]]
	data2 = [distributions[i] for i in labelling[1]]
	print("data1: ",len(data1))
	print("data2: ",len(data2))

	stat, p = ttest_ind(data1, data2)
	print(p)
	if p > crit_val :
		print('Probably the same distribution')
	else:
		print('Probably different distributions')
	return p

#V must be of shape (m^2)
def create_vectors(pers_images) :
	coords = [list(zip(x,y)) for x,y in zip(pers_images[0],pers_images[1])]

	#coords has shape of ((m^2/2,m^2/2),n)
	return coords[0]

def percentile(T,threshold) :
	vals_below = int((threshold*len(T))/100)
	print(vals_below)
	print(T[vals_below])
	return T[vals_below]

#Persistence Images Hypothesis Testing
# n - number of distributions
# m - resolution
def images_hypothesis_test(images, label, threshold) :
	n = len(images)
	m = int(np.sqrt(len(images[0][0])))

	V = np.zeros((np.power(m,2)))
	#populate V to 2d vectors of points on vectorised dgm
	V = create_vectors(images)
	print("length of V", len(V))
	#filt
	V = list(filter(lambda x : x[0] >= x[1],V))
	#populate loc_v with all 

	T = np.zeros((len(V)))

	#filtering stage
	print('length of V: ',len(V))
	for i in range (0,len(V)) :
		T[i] = (V[i][0] + V[i][1])/2
	c_percentile = percentile(T,threshold)
	V = list(filter(lambda x: x>c_percentile ,V))

	Z = np.zeros(n)

	#testing
	for j in range(1,len(V)) :
		j_label1 = label[0][j%len(label[0])]
		j_label2 = label[1][j%len(label[1])] 
		v_1 = V[j_label1]
		v_2 = V[j_label2]
		#conduct hypothesis test
		stat,Z[j] = ttest_ind(v_1, v_2)
	#need to do multiple testing adj.
	return multipletests(Z)[1]


#---TESTING---
def construct_tori() :
	tori = []
	for i in range(0,20) :
		theta = np.linspace(0, 2.*np.pi, 100)
		phi = np.linspace(0, 2.*np.pi, 100)

		x,y,z = (np.cos(theta))*np.cos(phi), (np.cos(theta))*np.sin(phi), np.sin(theta) 
		tor = np.array([[h,v,w] for h,v,w in zip(x,y,z)])
		#X = np.random.normal(loc=1,scale=0.5,size=(100,3))
		n_tor = tor #+ X
		tori.append(n_tor)
	return tori

def construct_spheres() :
	spheres = []
	for i in range(0,20) :
		theta = np.linspace(0, 2.*np.pi, 100)
		phi = np.linspace(0, 2.*np.pi, 100)

		x,y,z = (np.cos(theta))*np.sin(phi), (np.sin(theta))*np.sin(phi), np.cos(theta) 
		sphr = np.array([[h,v,w] for h,v,w in zip(x,y,z)])
		X = np.random.normal(loc=1,scale=0.5,size=(100,3))
		n_sphr = sphr + X
		spheres.append(n_sphr)
	return spheres 


def test(i) :
	#construct sample spaces
	distributions_tori = construct_tori()
	distribution_sphere = construct_spheres()
	distributions = distribution_sphere + distributions_tori

	#build diagrams on 0th and 1st
	cmplex = Methods.Rips_complexes(distributions,10)
	cmplex_alpha = Methods.Alpha_complexes(distributions)

	zero_skeletons = Methods.skeletons(cmplex,0)
	one_skeletons = Methods.skeletons(cmplex,1)
	two_skeletons = Methods.skeletons(cmplex,2)

	zero_persistences = Methods.persistences(zero_skeletons)
	one_persistences = Methods.persistences(one_skeletons)
	two_persistences = Methods.persistences(two_skeletons)

	#build landscapes and images 
	landscapes_zero = Methods.persistence_landscapes(zero_skeletons,0,3)
	landscapes_one = Methods.persistence_landscapes(one_skeletons,1,3)
	landscapes_two = Methods.persistence_landscapes(two_skeletons,2,3)

	images_zero = Methods.persistence_images(cmplex_alpha,0,10,0.3)
	images_one = Methods.persistence_images(cmplex_alpha,1,10,0.3)
	images_two = Methods.persistence_images(cmplex_alpha,2,10,0.3)


	#const vars - crit value and initial labelling 
	labelling = [[0,2,3,5,7,9,10,11,12,15],[1,4,6,8,13,14,16,17,18,19]]
	crit_val = 0.05
	threshold = 10


	
	p_val_dg_0 = unbiased_estimator(zero_skeletons,0,labelling,100,joint_loss,1,2)
	p_val_dg_1 = unbiased_estimator(one_skeletons,1,labelling,100,joint_loss,1,2)
	p_val_dg_2 = unbiased_estimator(two_skeletons,2,labelling,100,joint_loss,1,2)

	p_val_land_0 = two_test(landscapes_zero,labelling, crit_val)
	p_val_land_1 = two_test(landscapes_one,labelling, crit_val)
	p_val_land_2 = two_test(landscapes_two,labelling, crit_val)
	

	p_val_img_0 = images_hypothesis_test(images_zero,labelling,threshold)
	p_val_img_1 = images_hypothesis_test(images_one,labelling,threshold)
	p_val_img_2 = images_hypothesis_test(images_two,labelling,threshold)

	print("TEST ", i)

		
	print("P value 0 degree diagam: ", p_val_dg_0)
	print("P value 1 degree diagam: ", p_val_dg_1)
	print("P value 2 degree diagam: ", p_val_dg_2)

	print("P value 0 degree landscape: ", p_val_land_0)
	print("P value 1 degree landscape: ", p_val_land_1)
	print("P value 2 degree landscape: ", p_val_land_2)
	

	print("P value 0 degree images: ", p_val_img_0)
	print("P value 1 degree images: ", p_val_img_1)
	print("P value 2 degree images: ", p_val_img_2)


for i in range(1,6) :
	test(i)



