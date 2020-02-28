import random
import numpy as np
import copy

class TestObj:
    
    def __init__(self):
        self.a = random.randint(0, 5)
        self.b = random.uniform(0, 1)

obj_arr = [TestObj(), TestObj()]
obj_arr_2 = []

obj_arr_2.append(obj_arr[0])

print(obj_arr[0].a)
print(obj_arr_2[0].a)

obj_arr[0].a = random.randint(5, 10)

a = {3: [TestObj(), TestObj()], 5: [TestObj(), TestObj()]}
b = copy.deepcopy(a)

print(a)
print(b)

a.pop(3)

print(a)
print(b)

print(np.linspace(0, 99, 100))