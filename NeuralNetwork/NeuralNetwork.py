import numpy as np
from activationFuncs import sigmoid
from funcPrimes import sigmoid_prime
from ExtractMNIST import *
import pickle
from costFuncs import *





class NeuralNetwork:
    def __init__(self,counts):
        self.no_of_layers = len(counts)
        self.neurons_per_layer = counts
        self.weights=[]
        self.bias = []
        self.beta = 0.9
        self.vdw = []
        self.vbw = []
        for n in range(0,self.no_of_layers-1):
            self.weights.append(np.random.randn(self.neurons_per_layer[n],self.neurons_per_layer[n+1]))
            self.bias.append(np.random.randn(self.neurons_per_layer[n+1],1))
            self.vdw.append(np.zeros((self.neurons_per_layer[n], self.neurons_per_layer[n+1])))
            self.vbw.append(np.zeros((self.neurons_per_layer[n+1], 1)))

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



    def Momentumupdate(self, sensitivities, activations, lr, batch_size):
        for l in range(0, self.no_of_layers-1):
            self.weights[l] = self.weights[l] - ((lr/batch_size) * (np.dot(activations[l], sensitivities[l].T)))
            self.bias[l] = self.bias[l] - ((lr/batch_size) * (np.sum(sensitivities[l],axis = 1)[:,np.newaxis]))


    def normalUpdate(self, sensitivities, activations, lr, batch_size):
        for l in range(0, self.no_of_layers-1):
            self.weights[l] = self.weights[l] - ((lr/batch_size) * (np.dot(activations[l], sensitivities[l].T)))
            self.bias[l] = self.bias[l] - ((lr/batch_size) * (np.sum(sensitivities[l],axis = 1)[:,np.newaxis]))


    def train_NN(self, x_in, target_out, lr):
        batch_size = x_in.shape[1]
        (net_inputs,activations) = self.feed_forward(x_in)
        sensitivities = self.back_prop(net_inputs,activations[-1],target_out)
        self.Momentumupdate(sensitivities, activations, lr, batch_size)


    def SGD(self,input_patterns,target_out,lr,epochs,batch_size,tv_x=None,tv_y=None):
        no_of_patterns = input_patterns.shape[1]
        positions = np.arange(no_of_patterns)
        training_cost = []
        tv_cost=[]
        train_accuracy = []
        tv_accuracy = []
        for e in range(1,epochs+1):
            print("epoch no:", e)
            np.random.shuffle(positions)
            for i in range(0,no_of_patterns,batch_size):
                self.train_NN(input_patterns[:,positions[i:i+batch_size]],target_out[:,positions[i:i+batch_size]],lr)
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
        y[y>=0.5]=1
        y[y<0.5]=0
        total_patterns = ts_x.shape[1]
        count=0
        for i in range(0,total_patterns):
            if(np.array_equal(y[:,i],ts_y[:,i])):
                count+=1
        accuracy = (count/total_patterns) * 100
        return accuracy


def main():
    NN = NeuralNetwork([784,30,10])
    tr_x,tr_y,tv_x,tv_y,ts_x,ts_y = mnist()
    NN.SGD(tr_x,tr_y,3,30,100)
    # with open('NN_Handwritten_weights/weights.pickle','wb') as handle:
    # 	pickle.dump(NN.weights,handle,protocol=pickle.HIGHEST_PROTOCOL)
    # with open('NN_Handwritten_weights/bias.pickle','wb') as handle:
    # 	pickle.dump(NN.bias,handle,protocol=pickle.HIGHEST_PROTOCOL)
    accuracy=NN.accuracy(tv_x,tv_y)
    print("accuracy:",accuracy)

    # with open('NN_Handwritten_weights/weights.pickle','rb') as handle:
    # 	NN.weights = pickle.load(handle)
    # with open('NN_Handwritten_weights/bias.pickle','rb') as handle:
    # 	NN.bias = pickle.load(handle)


if __name__ == "__main__":
        main()
