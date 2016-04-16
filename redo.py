import data_utils
from neural_net import *
filename="./datasets/cifar-10-batches-py"
Xtr, Ytr, Xte, Yte=data_utils.load_CIFAR10(filename)
Xtr=np.array([i.flatten()-i.mean() for i in Xtr])
Xte=np.array([i.flatten()-i.mean() for i in Xte])

train_len=2000
test_len=train_len/5
nn=TwoLayerNet(np.shape(Xtr)[1],10000,10)

if False:
    nn.gradient_check(Xte[:train_len],Yte[:train_len])
    exit()

print nn.train(Xtr[:train_len],
                Ytr[:train_len],
                Xtr[train_len:train_len+test_len],
                Ytr[train_len:train_len+test_len],
               learning_rate=1e-3,
               learning_rate_decay=0.95,
               reg=1e-5,
               num_iters=100,
               batch_size=test_len)
print nn.accuracy(Xte[:test_len],Yte[:test_len])


