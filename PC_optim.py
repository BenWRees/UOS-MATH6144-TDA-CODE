import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import csv
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score

def construct_data_set(file) :
    x_coords = []
    y_coords = []
    with open(file) as f :
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader :
            if len(row) != 0 :
                value_1, value_2 = float(row[0]),float(row[1])

                x_coords.append(value_1)
                y_coords.append(value_2)
    #add in diagonal points 
    #create_diagonal(data)
    print(len(x_coords))
    print(len(y_coords))
    return np.array([x_coords,y_coords])


def Rips(DX, mel, dim, card):
    # Parameters: DX (distance matrix), 
    #             mel (maximum edge length for Rips filtration), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)

    # Compute the persistence pairs with Gudhi
    rc = gd.RipsComplex(distance_matrix=DX, max_edge_length=mel)
    st = rc.create_simplex_tree(max_dimension=dim+1)
    dgm = st.persistence()
    pairs = st.persistence_pairs()

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim+1:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1,:][:,l1]),[len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2,:][:,l2]),[len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))
    
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
    # Output indices
    indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
    return list(np.array(indices, dtype=np.int32))

class RipsModel(tf.keras.Model):
    def __init__(self, X, mel=12, dim=1, card=50):
        super(RipsModel, self).__init__()
        self.X = X
        self.mel = mel
        self.dim = dim
        self.card = card
        
    def call(self):
        m, d, c = self.mel, self.dim, self.card
        
        # Compute distance matrix
        DX = tfa.losses.metric_learning.pairwise_distance(self.X)
        DXX = tf.reshape(DX, [1, DX.shape[0], DX.shape[1]])
        
        # Turn numpy function into tensorflow function
        RipsTF = lambda DX: tf.numpy_function(Rips, [DX, m, d, c], [tf.int32 for _ in range(4*c)])
        
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        ids = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(RipsTF,DXX,dtype=[tf.int32 for _ in range(4*c)]))
        
        # Get persistence diagram by simply picking the corresponding entries in the distance matrix
        dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(ids, [2*c,2])), [c,2])
        return dgm

def remove_pts_from_set(data_set,n) :
    new_data_set = data_set
    for i in range(0,n) :  
        #new_data_set.pop(np.random.randint(0,len(new_data_set)))
        np.delete(new_data_set,np.random.randint(0,len(new_data_set)))
        #print(new_data_set)
    return np.array([[a[0] for a in new_data_set],[a[1] for a in new_data_set]],dtype=np.float32)

expected_model_rewrite = construct_data_set('frame553_thr120.csv')
expected_model = np.array([[a[0] for a in expected_model_rewrite],[a[1] for a in expected_model_rewrite]],dtype=np.float32)


n_pts    = 300   # number of points in the point clouds
card     = 50    # max number of points in the diagrams
hom      = 1     # homological dimension
ml       = 12.   # max distance in Rips
n_epochs = 100    # number of optimization steps

points = [170,431,517,861,1033] 
for i in points : 
    print("Data set with points removed: ", i)
    #Xinit = np.array(np.random.uniform(size=(n_pts,2)), dtype=np.float32)
    Xinit = remove_pts_from_set(expected_model,points)
    X = tf.Variable(initial_value=Xinit, trainable=True)

    model = RipsModel(X=X, mel=ml, dim=hom, card=card)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    for epoch in range(n_epochs+1):
        
        with tf.GradientTape() as tape:
            
            # Compute persistence diagram
            dgm = model.call()
            
            # Loss is sum of squares of distances to the diagonal
            loss = -tf.math.reduce_sum(tf.square(.5*(dgm[:,1]-dgm[:,0])))
            
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch % 20 == 0:
            #plt.figure()
            #plt.scatter(model.X.numpy()[:,0], model.X.numpy()[:,1])
            #plt.title("Point cloud at epoch " + str(epoch))
            #plt.show()
            print('epoch: ', epoch)
            dh, i, j = jaccard_score(Xinit,model.X.numpy())
            print('similarity', dh)


