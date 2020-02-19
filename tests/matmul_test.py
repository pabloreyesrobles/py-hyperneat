import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 1], [1, 1]])
i = np.array([[1, 0], [0, 1]])

print(b)
print(a)
print(np.matmul(b, a))

a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
print(a)