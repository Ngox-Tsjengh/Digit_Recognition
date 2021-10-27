import struct
import numpy

MNIST_DIR = "mnist"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"


def load_mnist(self, file_dir, is_image = "True"):
#Load MNIST data
    # 1 Read binary data
    file = open(file_dir, 'rb');
    data = file.read()
    file.close()
    # 2 Decide file type due to header
    if is_image: 
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:       #read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from()
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols;
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', data. struct.calcsize(fmt_header0))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))

    return mat_data


def load_data(self):
#Load all data(images and labels) from MNIST files 
    print('Loading MNIST data')
    # 1 Read files
    train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
    train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
    test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

    # 2 Append lables to images
    self.train_data = numpy.append(train_images, train_labels, axis=1)
    self.test_data = numpy.append(test_images, test_labels, axis=1)
