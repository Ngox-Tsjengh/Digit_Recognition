#! /usr/local/Caskroom/miniconda/base/envs/Digit_Recognition/bin/python

import numpy
from Net import build_mnist_network
import argparse

def evaluate(net):
    pred_results = numpy.zeros([net.test_data.shape[0]])
    for idx in range(net.test_data.shape[0]//net.batch_size):
        image_pixels = net.test_data[idx*net.batch_size:(idx+1)*net.batch_size, :-1]
        prob = net.forward(image_pixels)
        pred_labels = numpy.argmax(prob, axis=1)
        pred_results[idx*net.batch_size:(idx+1)*net.batch_size] = pred_labels
    if net.test_data.shape[0] % net.batch_size >0: 
        last_batch =net.test_data.shape[0] % net.batch_size 
        image_pixels = net.test_data[-last_batch:, :-1]
        prob = net.forward(image_pixels)
        pred_labels = numpy.argmax(prob, axis=1)
        pred_results[-last_batch:] = pred_labels
    accuracy = numpy.mean(pred_results == net.test_data[:,-1])
    print('Accuracy in test set: %f' % accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('epochs', type=int, 
                    help='Numbers of epochs')
    args = parser.parse_args()
    net = build_mnist_network(args.epochs)
    evaluate(net)
