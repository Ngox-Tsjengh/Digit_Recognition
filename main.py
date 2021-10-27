import struct


MNIST_DIR = "mnist"
TRAIN_DATA = "train-images.idx3-ubyte"
TRAIN_LABEL = "train-labels.idx1-ubyte"
TEST_DATA = "t10k-images.idx3-ubyte"
TEST_LABEL = "t10k-labels.idx1-ubyte"


# Load MNIST data
def load_mnist(self, file_dir, is_image = "True")
    # 1 Read binary data
    file = open(file_dir, 'rb');
    data = file.read()
    file.close()
    # 2 Decide file type due to header
    if is_image: 
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)



