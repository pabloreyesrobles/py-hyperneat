import time
from pair import pair
from statistics import median

x = [2, 4, 6, 8, 10]
y = [3, 5, 7, 9, 11]

tups = []

x_check = 6
y_check = 7

for i in range(len(x)):
    tups.append((x[i], y[i]))

tuple_time_arr = []
pair_time_arr = []

for i in range(10000):
    dict_1 = {}
    dict_2 = {}

    t = time.process_time()
    for tup in tups:
        dict_1[tup] = 1
    elapsed_time = time.process_time() - t
    tuple_time_arr.append(elapsed_time)

    t = time.process_time()
    for i in range(len(x)):
        dict_2[pair(x[i], y[i])] = 1
    elapsed_time = time.process_time() - t
    pair_time_arr.append(elapsed_time)

print('tuple time: {:f}, pair time: {:f}'.format(median(tuple_time_arr), median(pair_time_arr)))

tuple_time_arr = []
pair_time_arr = []

for i in range(10000):
    t = time.process_time()
    dict_1[(x_check, y_check)]
    tuple_time_arr.append(elapsed_time)

    t = time.process_time()
    dict_2[pair(x_check, y_check)]
    pair_time_arr.append(elapsed_time)

print('tuple time: {:f}, pair time: {:f}'.format(median(tuple_time_arr), median(pair_time_arr)))
