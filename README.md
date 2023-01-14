# handwritten-digits-recognition
Machine Learning Group Project

# Main References
1. [TensorFlow, Keras and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist#0)
2. [Kmeans](https://github.com/sharmaroshan/MNIST-Using-K-means/blob/master/KMeans%20Clustering%20for%20Imagery%20Analysis%20(Jupyter%20Notebook).ipynb)
3. [Sklearn: Recognizing hand-written digits](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)
4. [Recognizing digits from images](https://yash-kukreja-98.medium.com/recognizing-handwritten-digits-in-real-life-images-using-cnn-3b48a9ae5e3)
5. [countour detection](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)

# Dataset source
1. [yann.lecun (raw compressed)](https://yann.lecun.com/exdb/mnist/)
2. [tensorflow mnist](https://www.tensorflow.org/datasets/catalog/mnist)
3. Collect data from coursemates

In this project, we develop a script `mnist.py` to extract the data; you must call `mnist.download` once imported the `mnist`.

# Dataset Description
MNIST dataset contains train digit images, train labels, test digits and test labels. There are 10 digits. The digit images are grayscaled and 28*28 pixels. Its problem is to classify the digit from provided images.

# Data Collection
We have developed the code to batch label  and transform digits from given image file. The resultant file is a 28*28 pixels of siginle digit with last of its filename excluding extension is the label. 

The code refers to [Recognizing digits from images](https://yash-kukreja-98.medium.com/recognizing-handwritten-digits-in-real-life-images-using-cnn-3b48a9ae5e3). The code will auto crop the digits from pciture via [countour detection](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html).

Scanned picture sized A6 (i.e.: A4 fold 2 times), then set to high constrast works best.

First milestone would be 1000 data labelled. a significant figures less than MNIST train dataset size.

Second milestone is 10000 data labelled; same significant figures as MNIST train dataset size.

Third milesteon is 30000 data labelled; half of MNIST train dataset size.

# Data Preprocessing

Note that transformation and labelling are done just after cropping out the digits from images.

The transformation is as follow:

1. convert image to grayscale
2. padding
3. resize to 28*28 pixels

Due to how contour detection work, the padding to digit 1 requires special treatment such that we pad more to horizontal than vertical whereas other digits are padded equally.

# Data Analysis
Digits from MNIST are padded hence the raw data are padded similiarily then add to data.

# General Remark
28*28 is 784 which are  too large for kmeans model to fit, hence the data are unpadded and resized accordingly before fed into the kmeans model.

Since CNN training takes time, we save the CNN model, and then load the trained model in prediction. 

There are 2 models for comparison. Googler model is taken from the main reference notebook, local model is trained using collected data.

# Method

Kmeans and CNN are fitted according to the train dataset for comparison. In validation dataset, Kmeans does have 90% accuracy whereas CNN have 99% accuracy.

# Results


