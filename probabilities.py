import random
import numpy as np

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

print(obj_arr[0].a)
print(obj_arr_2[0].a)

def funcA():
    return 10

def funcB():
    return 5

print(funcA == funcA)