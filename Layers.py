import numpy


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output):
    #Initilize the FullyConnectedLayer
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))

    def init_param(self, std=0.01):
    #Initilize parameters
        self.weight = numpy.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
    #Forward propagation of FullyConnectedLayer
        start_time = time.time()

        self.input = input
        output = numpy.dot(self.input, self.weight) + self.bias
        return output
        
    def backward(self, top_diff):
    #Back propagation of FullyConnectedLayer
    #Calculate parameter gradients and losses in this layer
        self.d_weight = numpy.dot(self.input.T, top_diff)
        self.d_bias = top_diff
        bottom_diff = numpy.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
    #Update the parameters of the FullyConnectedLayer
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ActivationLayer(object):
    def forward(self, input):
        start_time = time.time()
        self.input = input
        output = numpy.maxium(self.input, 0)
        return output

    def backward(self, top_diff):
        Act = self.input
        Act[Act > 0] = 1
        Act[Act < 0] = 0
        bottom_diff = numpy.multiply(b, top_diff)
        return bottom_diff

class LossLayer(object):
    def forward
