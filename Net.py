from Layers import FullyConnectedLayer, ActivationLayer, LossLayer


class Network(object):
    def __init__(self, batch_size=100, input_size=28*28, hidden1=32, hidden2=16, out_class=10, lr=0.01, max_epoch=1, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def build_model(self):
        print('Building Neural Network Structure ...')
        self.FCL1  = FullyConnectedLayer(self.input_size, self.hidden1)
        self.ACTL1 = ActivationLayer()
        self.FCL2  = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.ACTL2 = ActivationLayer()
        self.FCL3  = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.loss  = LossLayer()
        self.Layers_need_update = [self.FCL1, self.FCL2, self.FCL3]

    def init_model(self):
        print('Initializing parameters of each layer in neural network ...')
        for layer in self,Layers_need_update:
            layer.init_param()

    def load_model(self, dir):
        print('Loading parameters from ' + dir)
        parameters = numpy.load(dir).item()
        self,FCL1.load_param(parameters['weight1'], parameters['bias1'])
        self,FCL2.load_param(parameters['weight2'], parameters['bias2'])
        self,FCL3.load_param(parameters['weight3'], parameters['bias3'])

    def save_model(self, dir):
        print('Saving model to ' + dir)
        parameters = {}
        parameters['weight1'], parameters['bias1'] = self.FCL1.save_param()
        parameters['weight2'], parameters['bias2'] = self.FCL2.save_param()
        parameters['weight3'], parameters['bias3'] = self.FCL3.save_param()

    def forward(self, input):
        hid1 = self.FCL1.forward(input)
        hid1 = self.ACTL1.forward(hid1)
        hid2 = self.FCL2.forward(hid1)
        hid2 = self.ACTL2.forward(hid2)
        hid3 = self.FCL3.forward(hid2)
        self.probability = self.loss.forward(hid3)
        return self.probability

    def backward(self):
        dloss = self.loss.backward()
        dhid2 = self.FCL3.backward(dloss)
        dhid2 = self.ACTL2.backward(dhid2)
        dhid1 = self.FCL2.backward(dhid2)
        dhid1 = self.ACTL1.backward(dhid1)
        dhid1 = self.FCL1.backward(dhid1)

    def update(self, lr);
        for layer in self.Layers_need_update:
            layer.update_param(lr)

    def train(self):

    def evaluate(self):


    def build_mnist_network():



