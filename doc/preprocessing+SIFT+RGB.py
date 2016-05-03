import numpy as np
import cv2
#get_ipython().magic(u'matplotlib inline')
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:

from sklearn import cross_validation


# # Preparing data

# In[3]:

# read all the files names 
from glob import glob
from os import path

def get_files_in(folder, pattern='*.txt'):
    return glob(path.join(folder, pattern))

def filenames(folder):
    filename = get_files_in(folder, '*.jpg')
    filename1 = get_files_in(folder, '*.mat')
    #for i in range(len(filename1)):
        #filename.append(filename1[i])
    return filename


# In[297]:

# for each image lable it as cat or dog
def image_lable(file_dir):
    image_dir = filenames(file_dir)
    lable = []
    img_name = []
    for i in range(len(image_dir)):
        image = cv2.imread(image_dir[i]) #read in BGR form
        if image == None: # ignore image that can not be read
            print image_dir[i]
            continue
            
        img_name.append(image_dir[i])
        #if image_dir[i].split("/")[1][0].isupper() == True:
            #lable.append(0) # 0 means cat
        #else:
            #lable.append(1) # 1 means dog
    return np.array(img_name)#, np.array(lable)

df_x = image_lable("images")


def SIFT_feature(img_list):
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = []
    for i in range(len(img_list)):
        image = cv2.imread(img_list[i]) #read in BGR form
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # transfer to gray color
        res = cv2.resize(gray,(256,256),interpolation = cv2.INTER_LINEAR) #resize to 256*256
        (kp_sift, descs_sift) = sift.detectAndCompute(res,None)
        des_list.append(descs_sift)
    
    return des_list


# In[10]:

def BoW(des_list, word_number, voc):   
    im_features = np.zeros((len(des_list), word_number), "float32")
    for i in range(len(des_list)):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1
            
    # Perform Tf-Idf vectorization        
    #nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    #idf = np.array(np.log((1.0*len(des_list)+1) / (1.0*nbr_occurences + 1)), 'float32')
    
    # Perform L2 normalization
    #im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l1')

    return im_features


# In[11]:

def color_histogram(image, Red_Int, Green_Int, Blue_Int):
    Rrange = range(0,255,256/Red_Int)
    Grange = range(0,255,256/Green_Int)
    Brange = range(0,255,256/Blue_Int)
    r1 = image[:,:,0]
    r2 = image[:,:,1]
    r3 = image[:,:,2]
    
    H1, edge1 = np.histogram(r1, bins = Rrange)
    H1 = np.hstack([H1,r1.size-sum(H1)])
    H2, edge2 = np.histogram(r2, bins = Grange)
    H2 = np.hstack([H2,r2.size-sum(H2)])
    H3, edge3 = np.histogram(r3, bins = Brange)
    H3 = np.hstack([H3,r3.size-sum(H3)])
    
    freq = []
    for i in range(len(H1)):
        for j in range(len(H2)):
            for k in range(len(H3)):
                freq.append(float(H1[i]*H2[j]*H3[k])/float(r1.size * r2.size *r3.size))
    return np.array(freq)    


# In[12]:

# for each image take the feature to form matrix
def model_feature_RGB(img_list, Red, Green, Blue):
    feature = []
    for i in range(len(img_list)):
        image = cv2.imread(img_list[i]) #read in BGR form
        image2 = image[:,:,::-1] # convert to RGB
        res = cv2.resize(image2,(256,256),interpolation = cv2.INTER_LINEAR) #resize to 256*256
        feature.append(color_histogram(res,Red,Green,Blue))
    return np.array(feature)




df_x = DF_last['0']


# In[422]:

SIFT_X = SIFT_feature(DF_last['0'])


# In[336]:

RGB_X = model_feature_RGB(df_x, 8,8,8)
train_feature_SIFT = BoW(SIFT_X, 2000, voc_sift)

# In[353]:


# In[338]:

DF_last = pd.read_csv('data_norm2_last.csv',header=0)


# In[344]:

train_feature_SIFT.shape


# In[339]:

feature_total_last = np.hstack([DF_last,train_feature_SIFT,RGB_X])


# In[317]:

feature_total_last = pd.DataFrame(feature_total_last)


# In[319]:

feature_1sel_last = feature_total.loc[:,imp_feature_lable + 2]


# In[320]:

feature_1sel.to_csv('feature_we_need.csv')

###############################################################################################
#                                                                                              
#                end of feature generation
#
###############################################################################################



########## modelling and testing###########################

# In[427]:

DF_last


# In[420]:

svm2.predict(feature_1sel)


# In[355]:

#feature_we_need = pd.read_csv('feature_we_need.csv')


# In[419]:

svm2.fit(DF_last.ix[:,'2':],DF_last['1'])



SIFT_ALL = SIFT_feature(X_train_name)


# In[37]:

sum(TEST_Y == TEST_Y_SIFT)


# In[34]:

list1 = []
for name in DF['0']:
    if name in df_x:
        list1.append(name)


# In[35]:

DF = DF[DF['0'].isin(list1)]


# In[22]:

descriptors = SIFT_ALL[0]
m = 1
for descriptor in SIFT_ALL[1:]:
    descriptors = np.vstack([descriptors, descriptor])
    m+=1
    print m


# In[170]:

descriptors.shape


# In[23]:

voc_sift, lable_sift = kmeans2(descriptors, 2000, iter = 35) 


# In[166]:




# In[44]:

TRAIN_X_RGB, TEST_X_RGB, TRAIN_Y_RGB, TEST_Y_RGB = cross_validation.train_test_split(RGB_X, DF['1'], test_size = 0.1, random_state = 10)


# In[28]:

from sklearn import linear_model, metrics, datasets, svm


# In[29]:

def my_kernel(X,Y):
    return metrics.pairwise.chi2_kernel(X,Y)


# In[45]:

svm_lin = svm.LinearSVC()
svm_lin.fit(TRAIN_X_RGB, TRAIN_Y)


# In[46]:

print metrics.classification_report(TRAIN_Y, svm_lin.predict(TRAIN_X_RGB)), metrics.classification_report(TEST_Y, svm_lin.predict(TEST_X_RGB))


# In[31]:

svm_chi = svm.SVC(kernel = my_kernel, C=1)
svm_chi.fit(train_feature_SIFT, TRAIN_Y)


# In[32]:

print metrics.classification_report(TRAIN_Y, svm_chi.predict(train_feature_SIFT)), metrics.classification_report(TEST_Y, svm_chi.predict(test_feature_SIFT))


# In[38]:

svm1 = svm.LinearSVC()
svm1.fit(TRAIN_X, TRAIN_Y)


# In[40]:

sgd = linear_model.SGDClassifier(alpha = 0.01)
sgd.fit(TRAIN_X, TRAIN_Y)


# In[96]:

type(TRAIN_Y)


# In[41]:

print metrics.classification_report(TRAIN_Y, svm1.predict(TRAIN_X)), metrics.classification_report(TRAIN_Y, sgd.predict(TRAIN_X))


# In[42]:

print metrics.classification_report(TEST_Y, svm1.predict(TEST_X)),metrics.classification_report(TEST_Y, sgd.predict(TEST_X))


# In[57]:

my_predict = sgd.predict(TEST_X).astype(float)*0.85+svm_chi.predict(test_feature_SIFT).astype(float)*0.75+svm_lin.predict(TEST_X_RGB).astype(float)*0.63


# In[58]:

my_predict


# In[59]:

for i in range(len(my_predict)):
    if my_predict[i] > 1.38:
        my_predict[i] = 1
    else:
        my_predict[i] = 0


# In[60]:

print metrics.classification_report(TEST_Y, my_predict)


# In[83]:

feature_total = pd.DataFrame(feature_total)
RF = ensemble.RandomForestClassifier(n_estimators=100, oob_score= True)
RF.fit(feature_total.loc[:,3:],feature_total.loc[:,2].astype('category'))




feature_importance = pd.DataFrame(RF.feature_importances_)


# In[110]:

feature_importance['imp'] = range(45776)


# In[124]:

feature_importance[0].mean()


# In[128]:

imp_feature_lable = feature_importance[feature_importance[0] > feature_importance[0].mean()]['imp']


# In[134]:




# In[137]:

feature_1sel.shape


# In[135]:

svm2 = svm.LinearSVC()
#svm2.fit(feature_1sel,feature_total.loc[:,2].astype('category'))


# In[136]:

scores = cross_validation.cross_val_score(svm2, feature_1sel,feature_total.loc[:,2].astype('category'), cv = 5)


# In[291]:

RF = ensemble.RandomForestClassifier(n_estimators=100, oob_score= True)


# In[295]:

RF.fit(feature_total.loc[:,2:],pd.DataFrame(lable)[0])


# In[294]:

pd.DataFrame(lable)[0]


# In[253]:

RF.oob_score_


# In[286]:




# In[141]:

scores


# In[250]:

svm2 = svm.LinearSVC(C = 1000)


# In[259]:

log = linear_model.LogisticRegression()


# In[260]:

cross_validation.cross_val_score(log, feature_1sel,lable, cv = 5)


# In[223]:

svm2.fit(feature_1sel,feature_total.loc[:,2].astype('category'))


# In[276]:

lable


# In[275]:

feature_total.loc[:,2:]


# In[273]:




# In[268]:

feature_total


# In[249]:

img_name[2214]


# In[238]:

img_name = []
lable = []
for i in range(len(name['0'])):

    img_name.append(name['0'][i])
    if name['0'][i].split("/")[1][0].isupper() == True:
        lable.append(0) # 0 means cat
    else:
        lable.append(1) # 1 means dog


# In[241]:

len(lable)


# In[227]:

feature_total

