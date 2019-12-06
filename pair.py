import math

def pair(k1, k2):
    return 0.5 * (k1 + k2) * (k1 + k2 + 1) + k2

def depair(z):
    w = math.floor(0.5 * (math.sqrt(8 * z + 1) - 1))
    t = 0.5 * (w * (w + 1))
    y = z - t
    x = w - y

    return (x, y)