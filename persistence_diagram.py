#mport gudhi
import matplotlib.pyplot as plt
import csv
import numpy as np

#Populate a list to plot the data of parameter 'file'. Returns a list of pairs
def populate(file) :
	data = []
	with open(file) as f :
		csv_reader = csv.reader(f, delimiter=',')
		for row in csv_reader :
			if len(row) != 0 :
				data.append(row)
	return data

data_bar = populate('barcode_frame553.csv')
print(data_bar)
persist_pts = zip([item[0] for item in data_bar], [item[1] for item in data_bar])
persist_diagram = np.zeros((0,2))
print(persist_diagram)


for pts in persist_pts :
	persist_diagram = np.vstack([persist_diagram,pts])

print(persist_diagram)

#gudhi.plot_persistence_diagram(persistence=persist_pts)