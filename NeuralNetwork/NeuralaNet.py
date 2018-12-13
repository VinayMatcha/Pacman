import numpy as np
import matplotlib.pyplot as plt
import copy

from ActivationFuncs import sigmoid
from FuncPrimes import sigmoid_prime
from ExtractMNIST import *
from CostFuncs import *

class NeuralNetwork:
    def __init__(self, neuralCountsPerlayer, lr, epochs, batchSize, optimizer):
        self.no_of_layers = len(neuralCountsPerlayer)
        self.neurons_per_layer = neuralCountsPerlayer
        self.weights=[]
        self.bias = []
        self.vdw = []
        self.vdb = []
        self.sdw = []
        self.sdb = []
        self.optimizer = optimizer
        self.learningRate = lr
        self.epochs = epochs
        self.batchSize = batchSize
        self.beta = 0.9
        self.beta1 = 0.99
        for n in range(0,self.no_of_layers-1):
            self.weights.append(np.random.randn(self.neurons_per_layer[n],self.neurons_per_layer[n+1]))
            self.bias.append(np.random.randn(self.neurons_per_layer[n+1],1))
            self.vdw.append(np.zeros((self.neurons_per_layer[n], self.neurons_per_layer[n+1]),dtype=np.float64))
            self.vdb.append(np.zeros((self.neurons_per_layer[n+1], 1), dtype=np.float64))
            self.sdw.append(np.zeros((self.neurons_per_layer[n], self.neurons_per_layer[n + 1]), dtype=np.float64))
            self.sdb.append(np.zeros((self.neurons_per_layer[n + 1], 1), dtype=np.float64))


    def feed_forward(self,x_in):
        activations = [x_in]
        net_inputs=[x_in]
        count =1
        for w,b in zip(self.weights,self.bias):
            count=count+1
            z = np.dot(w.T,activations[-1]) + b
            net_inputs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return (net_inputs,activations)

    def feed_forward_classify(self,x_in):
        activation =x_in
        for w,b in zip(self.weights,self.bias):
            z = np.dot(w.T,activation) + b
            a = sigmoid(z)
            activation = a
        return activation


    def back_prop(self,net_inputs,observed_out,target_out):
        sensitivities = []
        s_last = (observed_out - target_out) * (sigmoid_prime(net_inputs[-1]))
        sensitivities.append(s_last)
        for i in range(self.no_of_layers-2, 0, -1):
            s = np.dot(self.weights[i],sensitivities[0]) * sigmoid_prime(net_inputs[i])
            sensitivities.insert(0,s)
        return sensitivities



    def Momentumupdate(self, sensitivities, activations):
        for l in range(0, self.no_of_layers-1):
            self.vdw[l] = self.vdw[l] * self.beta  +   (np.dot(activations[l], sensitivities[l].T)) * (1-self.beta)
            self.weights[l] = self.weights[l] - (self.learningRate) * self.vdw[l]
            self.vdb[l] = self.vdb[l] * self.beta +   (np.sum(sensitivities[l],axis = 1)[:,np.newaxis]) * (1-self.beta)
            self.bias[l] = self.bias[l] - ((self.learningRate) * self.vdb[l])


    def normalUpdate(self, sensitivities, activations):
        for l in range(0, self.no_of_layers-1):
            self.weights[l] = self.weights[l] - ((self.learningRate) * (np.dot(activations[l], sensitivities[l].T)))
            self.bias[l] = self.bias[l] - ((self.learningRate) * (np.sum(sensitivities[l],axis = 1)[:,np.newaxis]))

    def rmsProp(self, sensitivities, activations):
        eps = 1e-8
        for l in range(0, self.no_of_layers-1):
            dw = np.dot(activations[l], sensitivities[l].T)
            db = np.sum(sensitivities[l],axis = 1)[:,np.newaxis]
            self.sdw[l] = self.beta1 * self.sdw[l] + (1-self.beta1) * (np.square((dw)))
            self.weights[l] = self.weights[l] - self.learningRate * (np.divide( dw, np.sqrt(self.sdw[l])+eps))
            self.sdb[l] = self.beta1 * self.sdb[l] + (1 - self.beta1) * (np.square((db)))
            self.bias[l] = self.bias[l] - ((self.learningRate) * (np.divide( db, np.sqrt(self.sdb[l])+eps)))

    def Adam(self, sensitivities, activations):
        eps = 1e-8
        for l in range(0, self.no_of_layers - 1):
            dw = np.dot(activations[l], sensitivities[l].T)
            db = np.sum(sensitivities[l], axis=1)[:, np.newaxis]
            self.vdw[l] = self.vdw[l] * self.beta + (dw) * (1 - self.beta)
            self.vdb[l] = self.vdb[l] * self.beta + (db) * (1-self.beta)

            self.sdw[l] = self.beta1 * self.sdw[l] + (1 - self.beta1) * (np.square((dw)))
            self.sdb[l] = self.beta1 * self.sdb[l] + (1 - self.beta1) * (np.square((db)))

            self.weights[l] = self.weights[l] - self.learningRate * (np.divide(self.vdw[l], np.sqrt(self.sdw[l]) + eps))
            self.bias[l] = self.bias[l] - ((self.learningRate) * (np.divide(self.vdb[l], np.sqrt(self.sdb[l]) + eps)))

    def Nestrov(self, sensitivities, activations):
        eps = 1e-8
        for l in range(0, self.no_of_layers - 1):
            dw = np.dot(activations[l], sensitivities[l].T)
            db = np.sum(sensitivities[l], axis=1)[:, np.newaxis]
            oldVdw = copy.copy(self.vdw[l])
            oldVdb = copy.copy(self.vdb[l])
            newVdw = self.vdw[l] * self.beta - self.learningRate * (dw)
            newVdb = self.vdb[l] * self.beta - self.learningRate * (db)
            self.weights[l] = self.weights[l] - self.beta * oldVdw + (1+self.beta) * newVdw
            self.bias[l] = self.bias[l] - self.beta * oldVdb + (1+self.beta) * newVdb

    def train_NN(self, x_in, target_out, lr):
        batch_size = x_in.shape[1]
        (net_inputs,activations) = self.feed_forward(x_in)
        sensitivities = self.back_prop(net_inputs,activations[-1],target_out)

        if(self.optimizer == "SGD"):
            self.normalUpdate(sensitivities, activations)
        elif(self.optimizer == "RmsProp"):
            self.rmsProp(sensitivities, activations)
        elif (self.optimizer == "Momentum"):
            self.Momentumupdate(sensitivities, activations)
        elif (self.optimizer == "Adam"):
            self.Adam(sensitivities, activations)
        elif (self.optimizer == "Nestrov"):
            self.Nestrov(sensitivities, activations)


    def preOptimizer(self,input_patterns,target_out,tv_x=None,tv_y=None):
        no_of_patterns = input_patterns.shape[1]
        positions = np.arange(no_of_patterns)
        training_cost = []
        tv_cost=[]
        train_accuracy = []
        tv_accuracy = []
        for e in range(1,self.epochs+1):
            print("epoch no:", e)
            np.random.shuffle(positions)
            for i in range(0,no_of_patterns,self.batchSize):
                self.train_NN(input_patterns[:,positions[i:i+self.batchSize]],target_out[:,positions[i:i+self.batchSize]], self.learningRate)
            a_tr = self.feed_forward_classify(input_patterns)
            cost_tr = mean_square_error(a_tr,target_out)
            training_cost.append(cost_tr)
            accuracy_train=self.accuracy(input_patterns,target_out)
            train_accuracy.append(accuracy_train)
            if(tv_x is not None):
                a_tv = self.feed_forward_classify(tv_x)
                cost_tv = mean_square_error(a_tv,tv_y)
                tv_cost.append(cost_tv)
                accuracy_tv=self.accuracy(tv_x,tv_y)
                tv_accuracy.append(accuracy_tv)

        return (training_cost,tv_cost,train_accuracy,tv_accuracy)


    def accuracy(self,ts_x,ts_y):
        y = self.feed_forward_classify(ts_x)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        total_patterns = ts_x.shape[1]
        count=0
        for i in range(0,total_patterns):
            if(np.array_equal(y[:,i],ts_y[:,i])):
                count+=1
        accuracy = (count/total_patterns) * 100
        print("correct classification count:", count)
        return accuracy


def main():
    optimizers = ['SGD'] #'SGD', 'RmsProp', 'Adam', 'Nestrov', 'Momentum'
    learning_rate = 0.03
    epochs = 35
    batchSize = 100
    neuralCountsPerlayer = [784,30,10]

    tr_x, tr_y, tv_x, tv_y, ts_x, ts_y = mnist()

    for optimizer in optimizers:
        print("---------------", optimizer, "--------------")
        NN = NeuralNetwork(neuralCountsPerlayer, learning_rate, epochs, batchSize, optimizer)
        training_cost,tv_cost = NN.preOptimizer(tr_x, tr_y)
        plt.plot(training_cost)
        plt.plot(tv_cost)
        plt.show()
        accuracy = NN.accuracy(ts_x, ts_y)
        print("accuracy of ",optimizer, " :", accuracy)


    # with open('NN_Handwritten_weights/weights.pickle','wb') as handle:
    # 	pickle.dump(NN.weights,handle,protocol=pickle.HIGHEST_PROTOCOL)
    # with open('NN_Handwritten_weights/bias.pickle','wb') as handle:
    # 	pickle.dump(NN.bias,handle,protocol=pickle.HIGHEST_PROTOCOL)
    # accuracy=NN.accuracy(tv_x,tv_y)
    # print("accuracy:",accuracy)

    # with open('NN_Handwritten_weights/weights.pickle','rb') as handle:
    # 	NN.weights = pickle.load(handle)
    # with open('NN_Handwritten_weights/bias.pickle','rb') as handle:
    # 	NN.bias = pickle.load(handle)


if __name__ == "__main__":
        main()
