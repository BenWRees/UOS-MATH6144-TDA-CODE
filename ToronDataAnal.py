import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import csv
from gudhi.representations import Landscape,Silhouette,PersistenceImage

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

toron_pts = populate('frame553_thr120.csv')
toron_dgm = populate('barcode_frame553.csv')
test_dgm = populate('dgm.csv')
test = populate('test.csv')


toron_compl = gd.AlphaComplex(points = toron_pts)
toron_skel = toron_compl.create_simplex_tree()
toron_pers = toron_skel.persistence()

#plot two filtration levels 
toron_compl_res_a = gd.RipsComplex(points=toron_pts,max_length=6250).create_simplex_tree()
toron_compl_res_b = gd.RipsComplex(points=toron_pts,max_length=6251).create_simplex_tree()
toron_pers_a = toron_compl_res_a.persistence()
toron_pers_b = toron_compl_res_b.persistence()

a_x_coords = [a[1][0] for a in tuple(toron_pers_a)]
a_y_coords = [a[1][1] for a in tuple(toron_pers_a)]

b_x_coords = [a[1][0] for a in tuple(toron_pers_b)]
b_y_coords = [a[1][1] for a in tuple(toron_pers_b)]

plt.plot(a_x_coords,a_y_coords)
plt.title("filtration of toron data set with filtration level 6250")
plt.show()
plt.plot(b_x_coords,b_y_coords)
plt.title("filtration of toron data set with filtration level 6251")
plt.show()

gd.plot_persistence_diagram(toron_pers, legend=True)
plt.show()

x_coords = [a[0] for a in toron_skel.persistence_intervals_in_dimension(1)]
y_coords = [a[1] for a in toron_skel.persistence_intervals_in_dimension(1)]
persist = [[d-b,b,d] for b,d in zip(x_coords,y_coords)]
max_persist = max([a[0] for a in persist])

def find_coords(max,vals) :
	for v in vals :
		if v[0] == max :
			return v
		continue
	return None

coords = find_coords(max_persist,persist)
print(coords)
#get k-th landscapes over values of t in resolution
landscape_two = Landscape(num_landscapes=2,resolution=100).fit_transform([toron_skel.persistence_intervals_in_dimension(1)])

plt.plot(landscape_two[0])
plt.title("Landscape")
plt.show()

silh = Silhouette(resolution=10,weight= lambda x : np.power(np.absolute(x[0]-x[1]),20)).fit_transform([toron_skel.persistence_intervals_in_dimension(1)])
plt.plot(silh[0])
plt.title("Silhouette")
plt.show()

#look at persistence lifespan
def lifespan_curve_val(dgm,t) :
	sum = 0 
	for pts in dgm :
		if (t < pts[1] and t >= pts[0]) :
			sum += pts[1]-pts[0]
	return sum

def lifespan_curve(dgm,resolution) :
	t = np.linspace(0,max(dgm[:,1]),resolution)
	lifespan_cv = []
	for time in t :

		lifespan_cv.append(lifespan_curve_val(dgm,time))
	return np.array(lifespan_cv),t

persistence_lifespan,t = lifespan_curve(toron_dgm,10)
plt.plot(t,persistence_lifespan)
plt.title('Persistence Lifespan Curve')
plt.show()

#Need to fix
PI = gd.representations.PersistenceImage(bandwidth=2, weight=lambda x: x[1]**2, im_range=[0,.004,0,.004], resolution=[100,100])
pi = PI.fit_transform([toron_skel.persistence_intervals_in_dimension(1)])
plt.imshow(np.reshape(pi[0], [100,100]))
plt.title("Persistence Image")
plt.show()

"""gd.plot_persistence_diagram(toron_pers, legend=True)
plt.show()"""

#print(persistence_lifespan)



