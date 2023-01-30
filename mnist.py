# mnist.py
# source code file is an object too
# let current file name be a context over the code below
import numpy as np
import gzip

CITATION = """\
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}
"""


URL_BASE = 'http://yann.lecun.com/exdb/mnist'

DATASETS_CORSPD = {
    'train_images' : 'train-images-idx3-ubyte.gz',
    'train_labels' : 'train-labels-idx1-ubyte.gz',
    'test_images' : 't10k-images-idx3-ubyte.gz',
    'test_labels' :'t10k-labels-idx1-ubyte.gz'
}

FILENAME = list(DATASETS_CORSPD.values())
TMP_FILE_DIR = 'mnist_data'
concat_dir = lambda dir, f: f'{dir}/{f}'
DATASETS_CORSPD = {attr : concat_dir(TMP_FILE_DIR,name) for attr, name in DATASETS_CORSPD.items()}

IMAGE_SIZE = 28
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
TRAIN_SIZE = 6*10**4
TEST_SIZE = 1*10**4

def download(force=False):
    import urllib.request, os
    if not os.path.exists(TMP_FILE_DIR):
        os.makedirs(TMP_FILE_DIR)
    if force == False:
        filenames = [f for f in FILENAME if f not in os.listdir(TMP_FILE_DIR)]
    else:
        filenames = FILENAME
    for filename in filenames:
        urllib.request.urlretrieve(concat_dir(URL_BASE,filename), concat_dir(TMP_FILE_DIR,filename))

def extract_images(image_filepath, num_images):
    with gzip.open(image_filepath,'r') as f:
        f.read(16) # header
        buf = f.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return np.array(data)

def extract_labels(labels_filepath, num_labels):
    with gzip.open(labels_filepath,'r') as f:
        f.read(8)  # header
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return np.array(labels)

def extract_test(n = TEST_SIZE):
    return extract_images(DATASETS_CORSPD['test_images'], n), extract_labels(DATASETS_CORSPD['test_labels'], n)

def extract_train(n = TRAIN_SIZE):
    return extract_images(DATASETS_CORSPD['train_images'], n), extract_labels(DATASETS_CORSPD['train_labels'], n)
