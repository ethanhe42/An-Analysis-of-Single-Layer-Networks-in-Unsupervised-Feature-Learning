import cPickle as pickle
with open("features.pickle") as f:
    [trainXC,valXC,testXC,y_train,y_val,y_test]=pickle.load(f)

print "train",trainXC.shape
print "val",valXC.shape
print "test",testXC.shape
# In[125]:

from neural_net import *
import matplotlib.pyplot as plt
input_size = trainXC.shape[1]
num_classes = 10


# In[126]:
import os.path
if not os.path.isfile("feats.csv"):
    with open("feats.csv","w") as f:
        f.write("hidden_size,momentum,dropout,learning_rate,learning_rate_decay"+'\n')

def tryArgs(hidden_size,momentum,dropout,learning_rate,learning_rate_decay):

    net = TwoLayerNet(input_size, hidden_size, num_classes,1e-4)
    # Train the network
    stats = net.train(trainXC, y_train, valXC, y_val,
                            num_iters=20000, batch_size=100,
                            learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
                            reg=0, verbose=False,update="momentum",arg=momentum,dropout=dropout)

       # Predict on the validation set
    val_acc = (net.predict(valXC) == y_val).mean()
    train_acc = (net.predict(trainXC) == y_train).mean()
    f=open("feats.csv","a")
    tune=[hidden_size,momentum,dropout,learning_rate,learning_rate_decay]
    f.write(str(tune).strip("[]")+'\n')
    f.close()
    print hidden_size,learning_rate_dacay,train_acc, val_acc


hidden_size = range(150,600,50)
momentum=[.5,.9,.95,.99]
dropout=[.3,.5,.7]
learning_rate=[5e-4*i for i in range(1,4,20)]
learning_rate_decay=[.9,.95,.99]

for i in hidden_size:
    for j in momentum:
        for k in dropout:
            for m in learning_rate:
                for n in learning_rate_decay:
                    tryArgs(i,j,k,m,n)

# In[121]:
if False:
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


