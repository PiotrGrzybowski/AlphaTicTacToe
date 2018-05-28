import numpy as np

g = np.zeros(shape=(3, 3))
g[0, 0] = 1
g[1, 0] = -1
g[0, 1] = 1
print(g)
print(np.argwhere(g == 0))
