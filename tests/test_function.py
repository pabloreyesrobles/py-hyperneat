import math
import sys
import numpy as np

z = [-1.5, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 6.0]
z_exp = [math.exp(i) for i in z]
sum_z_exp = sum(z_exp)

softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(softmax)

a = np.array(z).reshape((2, 4))
print(a[0][2])