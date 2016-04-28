### Single Layer neural network
Single Layer neural network's performence is not so good.  
Which have accuracy of 55% on CIFAR-10  
However, with two images preprocessing techniques(PCA whintening and Kmeans), it can reach 75% on CIFAR-10  
[Detailed report in here](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans/blob/master/report/mp1_Yihui%20He.pdf)  
reference:  
["Analysis Single Layer Unsupervised Feature Learning"](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans/blob/master/paper/AnalysisSingleLayerUnsupervisedFeatureLearning.pdf)  
#### How to run  
1. Put cifar-10 dataset in ./dataset
2. run python redo.py dataset

#### files description
neural_net.py implement the neural net  
redo.py implement preprocessing  
The accuracy of different combinations of hyperparameters without preprocessing are shown in two *.csv* files 
[here](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans/blob/master/fine_grained_nn.csv) and [here](https://github.com/yihui-he/Single-Layer-neural-network-with-PCAwhitening-Kmeans/blob/master/naive_nn.csv)  
Other files are not important, written for comparing different techniques and searching for parameters  
  
  
######If you have any questions, I'm glad to discuss with you.
