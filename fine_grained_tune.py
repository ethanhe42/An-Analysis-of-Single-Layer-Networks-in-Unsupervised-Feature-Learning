from neural_net import *
from threading import *
from data_utils import *


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = "C:\Users\Pomodori\workspace\cifar-10-batches-py"
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

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape



def tryArgs(hidden_size,learning_rate,reg):
    net = TwoLayerNet(3072, i, 10)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
        num_iters=1000, batch_size=200,
        learning_rate=learning_rate, learning_rate_decay=0.95,
        reg=reg, verbose=False)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    f=open("fine_grained_nn.csv","a")
    f.write(str(hidden_size)+','+str(learning_rate)+','+str(reg)+','+str(val_acc)+'\n')
    f.close()
    print hidden_size,learning_rate,reg, val_acc


hidden_size = range(300,450,10)
reg=[0.05*2**i for i in range(-2,8)]
f=open("naive_nn.csv","w")
for i in hidden_size:
    for k in reg:
        # t=Thread(target=tryArgs,args=(i,j,k))
        # t.daemon=True
        # t.start()
        tryArgs(i,0.001,k)
