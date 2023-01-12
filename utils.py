from IPython import display
def display_digits(digits, predictions, labels, title, n):
    import matplotlib.pyplot as plt
    import numpy as np
    '''
    Adapated from: https://colab.research.google.com/github/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-mnist-tutorial/keras_01_mnist.ipynb
    
    Author: Martin Gorner
    '''
    plt.rc('xtick', top=False, bottom=False, labelsize='large')
    plt.rc('axes', edgecolor='white')
    plt.rc('figure', facecolor='F0F0F0', figsize=(16,9))
    
    fig = plt.figure(figsize=(13,3))
    digits = np.reshape(digits, [n, 28, 28])
    digits = np.swapaxes(digits, 0, 1)
    digits = np.reshape(digits, [28, 28*n])
    plt.yticks([])
    plt.xticks([28*x+14 for x in range(n)], predictions)
    plt.xticks
    plt.grid(visible=None)
    for i,t in enumerate(plt.gca().xaxis.get_ticklabels()):
        if predictions[i] != labels[i]: t.set_color('red') # bad predictions in red
    plt.imshow(digits, cmap = 'gray_r')
    plt.grid(None)
    plt.title(title)
    display.display(fig)
    plt.rcParams.update(plt.rcParamsDefault) # reset to default
    
def display_digits_alt(digits, labels, title, n):
    display_digits(digits, labels, labels, title, n)

def display_digits_line(digits, labels, title, n):
    k = 16
    n += n%k
    for x in range(k, n, k):
        i = x - k
        display_digits_alt(digits[i:x], labels[i:x], title, k)