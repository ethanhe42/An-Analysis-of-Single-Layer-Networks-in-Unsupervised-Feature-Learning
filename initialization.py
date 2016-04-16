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

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10

# for method in ["Nesterov momentum","rmsprop"]:
#     net = TwoLayerNet(input_size, 500, num_classes,1e-5)
#     stats = net.train(X_train, y_train, X_val, y_val,
#                             num_iters=1000, batch_size=100,
#                             learning_rate=1e-4, learning_rate_decay=0.95,
#                             reg=0.8, verbose=True,update="momentum",arg=0.9)
#     val_acc = (net.predict(X_train) == y_train).mean()
#     print 'Train accuracy: ', val_acc
#     val_acc = (net.predict(X_val) == y_val).mean()
#     print 'Validation accuracy: ', val_acc
#     val_acc = (net.predict(X_test) == y_test).mean()
#     print 'Test accuracy: ', val_acc
methods=['normal','o']
for i in methods:
    net = TwoLayerNet(input_size, 500, num_classes,1e-5,init_method=i)
    stats = net.train(X_train, y_train, X_val, y_val,
                                num_iters=1000, batch_size=100,
                                learning_rate=1e-4, learning_rate_decay=0.95,
                                reg=0, verbose=True,update="momentum",arg=0.9,dropout=0.5)
    val_acc = (net.predict(X_train) == y_train).mean()
    print 'Train accuracy: ', val_acc
    val_acc = (net.predict(X_val) == y_val).mean()
    print 'Validation accuracy: ', val_acc
    val_acc = (net.predict(X_test) == y_test).mean()
    print 'Test accuracy: ', val_acc

    #Plot the loss function and train / validation accuracies
    plt.plot(stats['loss_history'])
plt.legend(methods)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig("dropout loss_history.eps")

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.savefig('dropout accuracy.eps')
