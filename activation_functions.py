import math
import random

SIGMOID_CONST = 4.9
GAUSS_DESV = 0.1
FREQ = 2 * math.pi
PHASE = 0.0

def UnsignedSigmoid(x):
    return 1.0 / (1.0 + math.exp(SIGMOID_CONST * x))

def SignedSigmoid(x):
    return (UnsignedSigmoid(x) - 0.5) * 2.0

def Tanh(x):
    return math.tanh(x)

def SignedSine(x):
    return math.sin(x * FREQ + PHASE)

def UnsignedSine(x):
    return (math.sin(x * FREQ + PHASE) + 1.0) / 2.0

def SignedCosine(x):
    return math.cos(x * FREQ + PHASE)

def UnsignedCosine(x):
    return (math.cos(x * FREQ + PHASE) + 1.0) / 2.0

def UnsignedGauss(x):
    value = -0.5 * (x ** 2) / (GAUSS_DESV ** 2.0)
    return 0.25 * (1 / (math.sqrt(2 * math.pi) * GAUSS_DESV)) * math.exp(value)

def SignedGauss(x):
    value = UnsignedGauss(x)
    return (value - 0.5) * 2.0

def UnsignedStep(x):
    if x > 0.0:
        return 1.0
    else:
        return 0.0

def SignedStep(x):
    if x > 0.0:
        return 1.0
    else:
        return -1.0

def Relu(x):
    if x < 0.0:
        return 0.0
    return x

def Abs(x):
    # Between 0 and 1.0. Check
    if math.fabs(x) > 1.0:
        return 1.0
    else:
        return math.fabs(x)

def Linear(x):
    if math.fabs(x) > 1.0:
        return (x) / math.fabs(x)
    else:
        return x

class ActivationFunction:

    def __init__(self):
        self.functions = {}
        self.functions['SIGNED_SIGMOID'] = SignedSigmoid
        self.functions['UNSIGNED_SIGMOID'] = UnsignedSigmoid
        self.functions['UNSIGNED_GAUSS'] = UnsignedGauss
        self.functions['SIGNED_GAUSS'] = SignedGauss
        self.functions['TANH'] = Tanh
        self.functions['UNSIGNED_SINE'] = UnsignedSine
        self.functions['SIGNED_SINE'] = SignedSine
        self.functions['UNSIGNED_COSINE'] = UnsignedCosine
        self.functions['SIGNED_COSINE'] = SignedCosine
        self.functions['UNSIGNED_STEP'] = UnsignedStep
        self.functions['SIGNED_STEP'] = SignedStep
        self.functions['RELU'] = Relu
        self.functions['ABS'] = Abs
        self.functions['LINEAR'] = Linear

    def get(self, function_name):
        return self.functions[function_name]

    def get_function_name(self, function):
        for function_name in self.functions:
            if function == self.functions[function_name]:
                return function_name

    def get_random_function(self):
        function_list = list(self.functions.values())
        return random.choice(function_list)

    def set_atemporal_set(self):
        self.functions.pop('SIGNED_SINE')
        self.functions.pop('UNSIGNED_SINE')
        self.functions.pop('SIGNED_COSINE')
        self.functions.pop('UNSIGNED_COSINE')

    def unset_lin_group(self):
        self.functions.pop('LINEAR')
        self.functions.pop('ABS')
        self.functions.pop('RELU')

    def use_only_sigmoid(self):
        self.functions = {}
        self.functions['SIGNED_SIGMOID'] = SignedSigmoid