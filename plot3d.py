
#avery tan altan 1392212 CMPUT366A7

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')




x = np.load('x.npy')
y = np.load('y.npy')
z = np.load('z.npy')


ax.plot_wireframe(x,y,z)
ax.set_zlabel('state-action value')
plt.title('Mountain Car Cost-To-Go Function')
plt.xlabel('position')
plt.ylabel('velocity')
# plt.zlabel('state-action value')
plt.legend()
plt.show()
