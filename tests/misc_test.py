from hyperneat import Substrate

arr = [Substrate() for i in range(10)]

for key, value in enumerate(arr):
	print('{:d}, {}'.format(key, value))