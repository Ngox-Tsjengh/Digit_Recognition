import os
import struct
import numpy
from Layers import FullyConnectedLayer, ActivationLayer, LossLayer

MNIST_DIR = "mnist"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

class Network(object):
    def __init__(self, batch_size=100, input_size=28*28, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=1):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch

    def load_mnist(self, file_dir, is_image = "True"):
#Load MNIST data
        # 1 Read binary data
        file = open(file_dir, 'rb');
        data = file.read()
        file.close()
        # 2 Decide file type due to header
        if is_image: 
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, data, 0)
        else:       #read labels
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols;
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', data, struct.calcsize(fmt_header))
        mat_data = numpy.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))

        return mat_data


    def load_data(self):
#Load all data(images and labels) from MNIST files 
# Read files and Append lables to images
        print('Loading MNIST data')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        self.train_data = numpy.append(train_images, train_labels, axis=1)

        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
        self.test_data = numpy.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
#shuffle data for cross validation
        print('Randomly shuffling data for cross validation ')
        numpy.random.shuffle(self.train_data)

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
        for layer in self.Layers_need_update:
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

    def update(self, lr):
        for layer in self.Layers_need_update:
            layer.update_param(lr)

    def train(self):
        print('Starting training ... ')
        for index_epoch in range(self.max_epoch):
            self.shuffle_data()
            for index_batch in range(int(self.train_data.shape[0] / self.batch_size)):
                # Deal with data
                image_pixels = self.train_data[index_batch*self.batch_size:(index_batch+1)*self.batch_size, :-1]
                image_labels = self.train_data[index_batch*self.batch_size:(index_batch+1)*self.batch_size, -1]
                # Train network
                probability = self.forward(image_pixels)
                loss = self.loss.get_loss(image_labels)
                self.backward()
                # Update parameter
                self.update(self.lr)
                if index_batch % self.batch_size == 0:
                    print('Epoch %d, Iteration %d, Loss: %.6f' % (index_epoch, index_batch, loss))

    def inference(self):
        pred_results = numpy.zeros([net.test_data.shape[0]])
        for index in range(int(self.test_data.shape[0] / self.batch_size)):
            image_pixels = self.test_data[index*self.batch_size:(index+1)*self.batch_size, :-1]
            results = self.test_data[index*self.batch_size:(index+1)*self.batch_size, -1]

            probability = self.forward(image_pixels)
            pred_labels = numpy.argmax(probability, axis=1)
            pred_results[idx*net.batch_size:(idx+1)*net.batch_size] = pred_labels
        accuracy = numpy.mean(results = self.test_data[:, -1])
        print('Test Set Accuracy: %f' % accuracy)

def build_mnist_network(epochs):
    hid1, hid2, epoch = 32, 16, epochs
    net = Network(hidden1 = hid1, hidden2 = hid2, max_epoch = epoch)
    net.load_data()
    net.build_model()
    net.init_model()
    net.train()

    return net
