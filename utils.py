import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython import display

def display_digits(digits, predictions, labels, title, n):
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
    plt.rcParams.update(plt.rcParamsDefault) # reset to default
    
def display_digits_line(digits, labels, pred, title, n):
    k = 16
    for x in range(k, n, k):
        i = x - k
        display_digits(digits[i:x], labels[i:x],pred[i:x], title, k)
    if n%k != 0:
        r = n%k
        display_digits(digits[n-r:n], labels[n-r:n],pred[n-r:n], title,r)
        
def display_digits_alt(digits, labels, title, n):
    display_digits(digits, labels, labels, title, n)

def display_digits_alt_line(digits, labels, title, n):
    k = 16
    for x in range(k, n, k):
        i = x - k
        display_digits_alt(digits[i:x], labels[i:x], title, k)
    if n%k != 0:
        r = n%k
        display_digits_alt(digits[n-r:n], labels[n-r:n], title,r)

def manual_label(digits):
    labels = []
    del_idx = []
    for i, digit in enumerate(digits):
        plt.axis('off')
        plt.imshow(digit, cmap='gray')
        plt.show()
        is_valid_inp = lambda x : x.isdigit() and int(x) >= 0 and int(x) <= 9
        
        # walrus operator returns value
        while not is_valid_inp(x := input()) and x != 'x':
            print("please enter single digit or 'x'")
            continue
        
        if is_valid_inp(x):
            labels.append(int(x))
        elif x == 'x':
            del_idx.append(i)
        
        display.clear_output(wait=True)

    digits = np.delete(digits, del_idx, 0)
    
    display.clear_output()
    # display_digits(digits, labels, labels, 'labelled', len(labels))
    # display_digits_line(digits, labels,'', len(labels))
    return digits, labels

def crop_number(img, verbose=False):
    grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey.copy(), (5,5), 0)
    _, thresh = cv2.threshold(blur.copy(),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_rect = [cv2.boundingRect(c) for c in contours]

    # small_param = 0.001
    digits = []

    for x,y,w,h in sorted(contours_rect):
        
        # filter out small rect
        # if w*h < small_param * img.shape[0] * img.shape[1]:
            # continue
        
        cv2.rectangle(img, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)
        
        digit = thresh[y:y+h, x:x+w]
        # digit = cv2.resize(digit, (18,18))
        # digit = np.pad(digit, 5, "constant", constant_values=0)
        digits.append(digit)

    digits = np.array(digits)
    if verbose:
        plt.title('coutoured image')
        plt.axis('off')
        plt.imshow(img, cmap="gray")
        plt.show()
    return digits

def transform_cropped_digit(digits, labels = None):
    from itertools import repeat
    res = []
    if labels is None: labels = repeat(None)
    for digit, label in zip(digits, labels):
        if label == 1: # requiring special treatment
            digit = cv2.resize(digit, (100,100))
            # (up,down), (left,right)
            hpad_width = 150
            vpad_width = 25
            pad_width=((vpad_width,vpad_width),
                       (hpad_width,hpad_width))
            digit = np.pad(digit, pad_width=pad_width)
            digit = cv2.resize(digit, (28,28))
        else:
            digit = cv2.resize(digit, (18,18))
            digit = np.pad(digit, 5, "constant", constant_values=0)
        res.append(digit)
    return np.array(res)

def str_today():
    import datetime
    tmp = str(datetime.date.today())
    for s in "-:. ": tmp = tmp.replace(s, '')
    return tmp

def save_label_digit(digits, labels):
    import glob
    from itertools import count
    tdy_processed_img = glob.glob(f'data/{str_today()}_*.png')
    counter = count(len(tdy_processed_img) + 1)

    for img,lbl in zip(digits, labels):
        filename = f'{str_today()}_{next(counter)}_{lbl}.png' # png is lossless format
        cv2.imwrite('data/'+filename, img)

def unpad(arr, width):
    return arr[width:arr.shape[0]-width, width:arr.shape[1]-width]

def get_data(n = None):
    from os import listdir, path
    digits, labels = [], []
    for filename in listdir('data'):
        digit = cv2.imread(path.join('data', filename))
        digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
        label = path.splitext(filename)[0][-1]
        digits.append(digit)
        labels.append(int(label))
    return np.array(digits), np.array(labels)
