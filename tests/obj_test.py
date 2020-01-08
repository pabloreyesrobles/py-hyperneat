import random, sys

class Guges:

    def __init__(self):
        self.a = 0
        self.b = []

    def mutate(self):
        num = random.randint(0, 10)
        self.a = num
        self.b.append(num)
    
def append_obj(arr):
    gugu = Guges()
    gugu.mutate()
    arr.append(gugu)

arr = []
tup = (0, 1)
print(tup[1])
sys.exit()
print(arr)
for i in range(10):
    append_obj(arr)

for x, y in enumerate(arr):
    print(x)