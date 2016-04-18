
# coding: utf-8

# In[59]:

from data_utils import load_CIFAR10
from neural_net import *
import matplotlib.pyplot as plt

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = './datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    # easyier for py
    X_train=X_train.swapaxes(1,3)
    X_val=X_val.swapaxes(1,3)
    X_test=X_test.swapaxes(1,3)
    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

rfSize = 6
numCentroids=1600
whitening=True
numPatches = 400000
CIFAR_DIM=[32,32,3]

#create unsurpervised data
patches=[]
for i in range(numPatches):
    if(np.mod(i,10000) == 0):
        print "sampling for Kmeans",i,"/",numPatches
    start_r=np.random.randint(CIFAR_DIM[0]-rfSize)
    start_c=np.random.randint(CIFAR_DIM[1]-rfSize)
    patch=np.array([])
    img=X_train[np.mod(i,X_train.shape[0])]
    for layer in img:
        patch=np.append(patch,layer[start_r:start_r+rfSize].T[start_c:start_c+rfSize].T.ravel())
    patches.append(patch)
patches=np.array(patches)
#normalize patches
patches=(patches-patches.mean(1)[:,None])/np.sqrt(patches.var(1)+10)[:,None]


# In[66]:

#whitening

[D,V]=np.linalg.eig(np.cov(patches,rowvar=0))

P = V.dot(np.diag(np.sqrt(1/(D + 0.1)))).dot(V.T)
patches = patches.dot(P)


# In[ ]:

centroids=np.random.randn(numCentroids,patches.shape[1])*.1
num_iters=50
batch_size=1000#CSIL do not have enough memory, dam
for ite in range(num_iters):
    print "kmeans iters",ite+1,"/",num_iters
#     c2=.5*np.power(centroids,2).sum(1)
#     idx=np.argmax(patches.dot(centroids.T)-c2,axis=1) # x2 the same omit
    hf_c2_sum=.5*np.power(centroids,2).sum(1)
    counts=np.zeros(numCentroids)
    summation=np.zeros_like(centroids)
    for i in range(0,len(patches),batch_size):
        last_i=min(i+batch_size,len(patches))
        idx=np.argmax(patches[i:last_i].dot(centroids.T)                  -hf_c2_sum.T,                  axis=1)        
        S=np.zeros([last_i-i,numCentroids])
        S[range(last_i-i),
          np.argmax(patches[i:last_i].dot(centroids.T)-hf_c2_sum.T
                    ,axis=1)]=1
        summation+=S.T.dot(patches[i:last_i])
        counts+=S.sum(0)
    centroids=summation/counts[:,None]
    centroids[counts==0]=0 # some centroids didn't get members, divide by zero
    #the thing is, they will stay zero forever
    


# In[82]:

def sliding(img,window=[6,6]):
    out=np.array([])
    for i in range(3):
        s=img.shape
        row=s[1]
        col=s[2]
        col_extent = col - window[1] + 1
        row_extent = row - window[0] + 1
        start_idx = np.arange(window[0])[:,None]*col + np.arange(window[1])
        offset_idx = np.arange(row_extent)[:,None]*col + np.arange(col_extent)
        if len(out)==0:
            out=np.take (img[i],start_idx.ravel()[:,None] + offset_idx.ravel())
        else:
            out=np.append(out,np.take (img[i],start_idx.ravel()[:,None] + offset_idx.ravel()),axis=0)
    return out


# In[111]:

def extract_features(X_train):
    trainXC=[]
    idx=0
    for img in X_train:
        idx+=1
        if not np.mod(idx,1000):
            print "extract features",idx,'/',len(X_train)
        patches=sliding(img,[rfSize,rfSize]).T
        #normalize
        patches=(patches-patches.mean(1)[:,None])/(np.sqrt(patches.var(1)+10)[:,None])
        #map to feature space
        patches=patches.dot(P)
        #calculate distance using x2-2xc+c2
        x2=np.power(patches,2).sum(1)
        c2=np.power(centroids,2).sum(1)
        xc=patches.dot(centroids.T)

        dist=np.sqrt(-2*xc+x2[:,None]+c2)
        u=dist.mean(1)
        patches=np.maximum(-dist+u[:,None],0)
        rs=CIFAR_DIM[0]-rfSize+1
        cs=CIFAR_DIM[1]-rfSize+1
        patches=np.reshape(patches,[rs,cs,-1])
        q=[]
        q.append(patches[0:rs/2,0:cs/2].sum(0).sum(0))
        q.append(patches[0:rs/2,cs/2:cs-1].sum(0).sum(0))
        q.append(patches[rs/2:rs-1,0:cs/2].sum(0).sum(0))
        q.append(patches[rs/2:rs-1,cs/2:cs-1].sum(0).sum(0))
        q=np.array(q).ravel()
        trainXC.append(q)
    trainXC=np.array(trainXC)
    trainXC=(trainXC-trainXC.mean(1)[:,None])/(np.sqrt(trainXC.var(1)+.01)[:,None])
    return trainXC


# In[112]:

valXC=extract_features(X_val)

testXC=extract_features(X_test)


# # save features

# In[131]:

import cPickle as pickle
with open("features.pickle","w") as f:
    pickle.dump([trainXC,valXC,testXC,y_train,y_val,y_test],f)


# In[125]:

from neural_net import *

input_size = trainXC.shape[1]
hidden_size = 150
num_classes = 10

net = TwoLayerNet(input_size, hidden_size, num_classes,1e-4)
stats = net.train(trainXC, y_train, valXC, y_val,
                            num_iters=20000, batch_size=100,
                            learning_rate=1e-3, learning_rate_decay=0.95,
                            reg=0, verbose=True,update="momentum",arg=0.9,dropout=0.5)


# In[126]:

val_acc = (net.predict(trainXC) == y_train).mean()
print 'Train accuracy: ', val_acc
val_acc = (net.predict(valXC) == y_val).mean()
print 'Validation accuracy: ', val_acc

val_acc = (net.predict(testXC) == y_test).mean()
print 'Test accuracy: ', val_acc


# In[121]:

#Plot the loss function and train / validation accuracies
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
#plt.savefig("dropout loss_history.eps")

plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.show()
plt.ylabel('Clasification accuracy')
#plt.savefig('dropout accuracy.eps')


# In[ ]:



