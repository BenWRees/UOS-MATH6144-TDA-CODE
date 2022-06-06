import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import Methods
from gudhi.representations import Landscape,Silhouette,PersistenceImage
import plotly.graph_objects as go
from mpl_toolkits import mplot3d


distributions = Methods.generate_norm_distributions(4,4,50,Methods.generate_rand_circs,2)
complexes = Methods.Rips_complexes(distributions,4)
zero_skeletons = Methods.skeletons(complexes,0)
one_skeletons = Methods.skeletons(complexes,2)
print(zero_skeletons[0].persistence_intervals_in_dimension(0))
zero_pers = Methods.persistences(zero_skeletons)
one_pers = Methods.persistences(one_skeletons)

zero_land = Methods.persistence_landscapes(zero_skeletons,0,3)
one_land = Methods.persistence_landscapes(one_skeletons,1,3)

zero_sil = Methods.persistence_silhouettes(zero_skeletons,0,3,20)
one_sil = Methods.persistence_silhouettes(one_skeletons,1,3,20)

mean_zero_land = Methods.mean_persistence(zero_land)
mean_one_land = Methods.mean_persistence(one_land)
#print(mean_zero_land)


mean_zero_sil = Methods.mean_persistence(zero_sil)
mean_one_sil = Methods.mean_persistence(one_sil)

"""frech_mean_zero = Methods.frechet_mean(zero_skeletons,0)
frech_mean_one = Methods.frechet_mean(one_skeletons,1)"""

plt.plot(mean_zero_land, label="mean landscape in degree 0")
plt.plot(mean_one_land, label="mean landscape in degree 1")
plt.legend(loc=1)
plt.show()

plt.plot(mean_zero_sil, label="mean silhouette in degree 0")
plt.plot(mean_one_sil, label="mean silhouette in degree 1")
plt.legend(loc=1)
plt.show()
"""
plt.plot([a[0] for a in frech_mean_zero],[a[1] for a in frech_mean_zero], label="$Fr\'{e}chet Mean of degree 0$")
plt.plot([a[0] for a in frech_mean_one],[a[1] for a in frech_mean_one], label="$Fr\'{e}chet Mean of degree 0$")
plt.legend()
plt.show()"""



