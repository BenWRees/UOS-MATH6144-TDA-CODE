import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd


X = np.random.uniform(size=[200,2])

#plot of the random dist
plt.scatter(X[:,0],X[:,1], color = 'red') 
plt.show()



complex = gd.RipsComplex(points = X, max_edge_length = 12)
skeleton = complex.create_simplex_tree(max_dimension=4)

"""
result_str = 'Rips complex is of dimension ' + repr(skeleton.dimension()) + ' - ' + \
    repr(skeleton.num_simplices()) + ' simplices - ' + \
    repr(skeleton.num_vertices()) + ' vertices.'
print(result_str)
fmt = '%s -> %.2f'
for filtered_value in skeleton.get_filtration():
    print(fmt % tuple(filtered_value))
"""

dgmX = skeleton.persistence()
gd.plot_persistence_barcode(dgmX,legend=True)
plt.show()

gd.plot_persistence_diagram(dgmX,legend=True)
plt.show()