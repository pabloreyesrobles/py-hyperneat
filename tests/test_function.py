import math
import sys

z = [-1.5, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
z_exp = [math.exp(i) for i in z]
sum_z_exp = sum(z_exp)

softmax = [round(i / sum_z_exp, 3) for i in z_exp]
print(softmax)


a = [[] for i in range(6)]
print(sys.getsizeof(a))

a[0].append(2)
a[3].append(3)

for arr in a:
    if len(arr) == 0:
        arr = None
print(a)
print(sys.getsizeof(a[0]))