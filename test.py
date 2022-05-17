import numpy as np
from gudhi.representations import Landscape
import matplotlib.pyplot as plt

D = np.array([[0.,4.],[1.,2.],[3.,8.],[6.,8.]])
diags = [D]
L=Landscape(num_landscapes=2,resolution=1000).fit_transform(diags)
print(L)

plt.plot(L[0][:1000])
#plt.plot(L[0][1000:2000])
#plt.plot(L[0][2000:3000])
plt.title("Landscape")
plt.show()
