
###########################start of deep feature generation######################

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().magic(u'matplotlib inline')

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


# In[2]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
# caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


# In[3]:

import os
if os.path.isfile('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    get_ipython().system(u'../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


# In[4]:

caffe.set_mode_cpu()

model_def = 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# In[5]:

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('python1/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# In[6]:

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


# In[7]:

from glob import glob
from os import path

def get_files_in(folder, pattern='*.txt'):
    return glob(path.join(folder, pattern))

def filenames(folder):
    filename = get_files_in(folder, '*.jpg')
    #filename1 = get_files_in(folder, '*.mat')
    #for i in range(len(filename)):
        #filename.append(filename1[i])
    return filename


# In[8]:

dic_dir = filenames('images')


# In[9]:

def image_lable(file_dir):
    img_dir = filenames(file_dir)
    lable = []
    img_name = []
    for i in range(len(img_dir)):
        image = caffe.io.load_image(img_dir[i])
        if image == None:
            continue
        img_name.append(img_dir[i])
        if img_dir[i].split("/")[1][0].isupper() == True:
            lable.append(0) # 0 means cat
        else:
            lable.append(1) # dog
            
    return np.array(img_name), np.array(lable)


# In[10]:

df_x, df_y = image_lable('images')




i = 0
image = caffe.io.load_image(X_train_name[0])
net.blobs['data'].data[...] = transformer.preprocess('data', image)
#feature = np.array(net.blobs['fc6'].data[0])
#feature = np.reshape(net.blobs['norm1'].data[0], 69984, order='C')
feature0 = np.vstack([net.blobs['norm1'].data[0][4,:,:],net.blobs['norm1'].data[0][9,:,:]])
feature = feature0.reshape(1458, order='C')
for name in X_train_name[1:]:
    image = caffe.io.load_image(name)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()
    #feature = np.vstack([feature,(np.array(net.blobs['fc6'].data[0]))])
    feature0 = np.vstack([net.blobs['norm1'].data[0][4,:,:],net.blobs['norm1'].data[0][9,:,:]])
    feature = np.vstack([feature, feature0.reshape(1458, order='C')])
    i += 1
    print i

feature.to_csv('data_norm2_last.csv')

###############################end of feature generation##################################

########################testing and modelling##################################

#perform classification
    #net.forward()

# obtain the output probabilities
    #output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
#top_inds = output_prob.argsort()[::-1][:5]

#plt.imshow(image)

#print 'probabilities and labels:'
#zip(output_prob[top_inds], labels[top_inds])


# In[34]:

print feature.shape, Y_train.shape


# In[35]:

print len(feature), len(Y_train)


# In[36]:

#from sklearn import linear_model, svm, metrics, datasets


# In[37]:

#svm = svm.LinearSVC()
#svm.fit(feature, Y_train)


# In[38]:

#log = linear_model.LogisticRegressionCV(cv = 5, Cs = 3)
#log.fit(feature, Y_train)


# In[39]:

#pac = linear_model.PassiveAggressiveClassifier(C = 0.1)
#pac.fit(feature, Y_train)


# In[40]:

#sgd = linear_model.SGDClassifier(alpha = 0.01)
#sgd.fit(feature, Y_train)


# In[41]:

#print metrics.classification_report(Y_train, sgd.predict(feature)), metrics.classification_report(Y_train, svm.predict(feature))


# In[159]:

i = 0
image = caffe.io.load_image(X_train_name1[0])
net.blobs['data'].data[...] = transformer.preprocess('data', image)
#feature_test1 = np.array(net.blobs['pool5'].data[0])
feature_test1 = np.reshape(net.blobs['norm1'].data[0], 69984, order='C')
for name in X_train_name1[1:]:
    image = caffe.io.load_image(name)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward()
    #feature_test1 = np.vstack([feature_test1,(np.array(net.blobs['pool5'].data[0]))])
    feature_test1 = np.vstack([feature_test1, np.reshape(net.blobs['norm1'].data[0], 69984, order='C')])
    i += 1
    print i


# In[160]:

#print metrics.classification_report(Y_train1, sgd.predict(feature_test1)), metrics.classification_report(Y_train1, svm.predict(feature_test1))


# In[20]:

image1 = caffe.io.load_image("images/Abyssinian_11.jpg")
net.blobs['data'].data[...] = transformer.preprocess('data', image1)
net.forward()
feature = np.array(net.blobs['conv1'].data[0])


#############################visulaisation of the network####################################
# In[21]:

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')


# In[22]:

net.blobs['fc6'].data[0].shape


# In[23]:

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


# In[24]:

a = np.reshape(net.blobs['conv1'].data[0], 290400, order='C')
#a = np.reshape(net.params['conv1'].data[0], 290400, order='C')
a.max()
#np.amax(a)
#a.shape


# In[26]:

feat = net.blobs['conv1'].data[0]
vis_square(feat)


# In[35]:

get_ipython().magic(u'pinfo np.pad')


# In[190]:

new = np.vstack([net.blobs['norm1'].data[0][4,:,:],net.blobs['norm1'].data[0][9,:,:]])


# In[191]:

new.shape


# In[192]:

new1 = new.reshape(1458, order='C')


# In[193]:

new2 = np.vstack([new.reshape(1458, order='C'),new1])


# In[194]:

new2.shape


# In[181]:

Y_train.shape

