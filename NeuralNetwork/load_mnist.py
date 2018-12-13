import numpy as np
import os
import pdb
import matplotlib as plt

datasets_dir = ''


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


# def mnist(noTrSamples=1000, noTsSamples=100, digit_range=[0, 10], noTrPerClass=100, noTsPerClass=10):
def mnist():
    data_dir = os.path.join(datasets_dir, 'MNIST/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)
    

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(int)
 

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)
  

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(int)
    

    trData = np.transpose(trData)
    tsData = np.transpose(tsData)

    tvData = np.zeros((784,10000))
    tvLabels=np.zeros(10000,dtype=int)
    incr = 1000
    i=0
    digits = np.arange(10)
    pos_for_delete=np.array([])
    for d in digits:
        pos_tuple = np.where(trLabels==d)
        pos = pos_tuple[0]
        pos_first = pos[pos<30000]
        pos_last = pos[pos>=30000]
        np.random.shuffle(pos_first)
        np.random.shuffle(pos_last)
        pos_first_500 = pos_first[:500]
        pos_last_500 = pos_last[:500]
        pos_final = np.concatenate((pos_first_500,pos_last_500))
        pos_for_delete=np.concatenate((pos_for_delete,pos_final))
        tvData[:,i:i+incr] = trData[:,pos_final]
        tvLabels[i:i+incr] = trLabels[pos_final]
        i=i+incr
    trData2= np.delete(trData,pos_for_delete,1)
    trLabels2=np.delete(trLabels,pos_for_delete)
        

    tr_X = (trData2 - np.mean(trData2))/np.std(trData2)
    ts_X = (tsData- np.mean(tsData))/np.std(tsData)
    tv_X = (tvData- np.mean(tvData))/np.std(tvData)

    tr_Y = np.zeros((10,50000))
    ts_Y = np.zeros((10,10000))
    tv_Y = np.zeros((10,10000))
    for i in range(0,len(trLabels2)):
        tr_Y[trLabels2[i],i]= 1
    for k in range(0,len(tvLabels)):
        tv_Y[tvLabels[k],k]=1
    for j in range(0,len(tsLabels)):
        ts_Y[tsLabels[j],j] = 1


    return tr_X, tr_Y, tv_X, tv_Y,ts_X, ts_Y


def main():

    train_X,train_Y, Valid_X, Valid_Y, test_X, test_Y = mnist()
    train_X.dump('MNIST_data/train-images.dat')
    train_Y.dump('MNIST_data/train-labels.dat')
    Valid_X.dump('MNIST_data/validation-images.dat')
    Valid_Y.dump('MNIST_data/validation-labels.dat')
    test_X.dump('MNIST_data/test-images.dat')
    test_Y.dump('MNIST_data/test-labels.dat')
    # print(np.shape(train_X))
    # print(np.shape(train_Y))
    # print(np.shape(Valid_X))
    # print(np.shape(Valid_Y))
    # print(np.shape(test_X))
    # print(np.shape(test_Y))


    

    # print(train_X[1,:].reshape(28,28))
    # print(train_Y[1])
    # figure()
    # gray()
    # print(test_Y[:,9001])
    # imshow(test_X[:,9001].reshape(28,28))
    # show()


    # trX, trY, tsX, tsY = mnist(noTrSamples=60000,
    #                            noTsSamples=10000, digit_range=[0, 10],
    #                            noTrPerClass=100, noTsPerClass=10)
  

if __name__ == "__main__":
    main()

