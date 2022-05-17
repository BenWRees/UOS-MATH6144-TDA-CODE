import matplotlib.pyplot as plt
import csv
import gudhi as gd
import numpy as np




#Populate a list to plot the data of parameter 'file'. Returns a list of pairs
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



skeleton = gd.RipsComplex(points = populate('frame553_thr120.csv'), max_edge_length = 0.2).create_simplex_tree(max_dimension=0)
dgmX = skeleton.persistence()


gd.plot_persistence_diagram(dgmX)
plt.show()

