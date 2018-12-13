import numpy as np


class NeuralNetwork(object):
    def __init__(self, layer_list, alpha, isClassification):
        self.layer_list = layer_list
        self.num_layers = len(layer_list)
        self.isClassification = isClassification
        self.alpha = alpha
        self.weights = {}
        self.biases = {}
        self.delta_weights = {}
        self.delta_biases = {}
        for i in range(self.num_layers - 1):
            self.weights[i] = np.random.randn(self.layer_list[i+1], self.layer_list[i]) / np.sqrt(layer_list[i])
            self.biases[i] = np.random.randn(self.layer_list[i+1],1) / np.sqrt(self.layer_list[i+1])
            self.delta_weights[i] = np.zeros((self.layer_list[i+1], self.layer_list[i]),float)
            self.delta_biases[i] = np.zeros((self.layer_list[i+1],1), float)

    # self.d_softmax(z_out) *
    def out_delta(self, y, h_out, z_out, difference, discount):
        if self.isClassification:
            return -1 * discount * difference * (y - h_out)
        return np.array([-1 * difference]).T

    def hidden_delta(self, next_delta, w_l, z_l):
        return np.dot(np.transpose(w_l), next_delta) * self.d_relu(z_l)

    def relu(self, z):
        z[z < 0] = 0
        return z

    def d_relu(self, z):
        z[z != 0] = 1
        return z

    def softmax(self, z):
        # temp_z = min(z)
        # for i in range(len(z)):
        #     z[i] = z[i] - temp_z
        denominator = np.sum(np.exp(z))
        return np.exp(z)/denominator

    def d_softmax(self, z):
        return self.softmax(z)*(1 - self.softmax(z))

    def forward_propagate(self, x):
        hidden = {}
        z_values = {}
        hidden[0] = x
        for layer in range(self.num_layers-1):
            print np.dot(self.weights[layer], hidden[layer]).shape
            z_values[layer + 1] = np.dot(self.weights[layer], hidden[layer])
            # + self.biases[layer]
            if layer+1 == self.num_layers-1:
                if self.isClassification:
                    hidden[layer + 1] = self.softmax(z_values[layer + 1])
                else:
                    hidden[layer + 1] = z_values[layer + 1]
            else:
                hidden[layer + 1] = self.relu(z_values[layer + 1])
        return hidden, z_values

    def back_propagate(self, y, h, z, difference, discount):
        layer_delta = {}
        for l in range(self.num_layers, 1, -1):
            if l == self.num_layers:
                layer_delta[l-1] = self.out_delta(y, h[l-1], z[l-1], difference, discount)
                self.delta_weights[l-2] += np.dot(layer_delta[l-1][:,np.newaxis], np.transpose(h[l-2][:,np.newaxis]))
                self.delta_biases[l-2] += layer_delta[l-1].T
            else:
                layer_delta[l-1] = self.hidden_delta(layer_delta[l], self.weights[l-1], z[l-1])
                self.delta_weights[l-2] += np.dot(layer_delta[l-1][:,np.newaxis], np.transpose(h[l-2][:,np.newaxis]))
                self.delta_biases[l-2] += layer_delta[l-1].T
        if (episode + 1) % 1 == 0:
            for l in range(self.num_layers-1, 0, -1):
                self.weights[l-1] = self.weights[l-1] - self.alpha * self.delta_weights[l-1]/10
                self.biases[l-1] = self.biases[l-1] - self.alpha * self.delta_biases[l-1]/10
            for i in range(self.num_layers - 1):
                self.delta_weights[i] = np.zeros((self.layer_list[i + 1], self.layer_list[i]), float)
                self.delta_biases[i] = np.zeros(self.layer_list[i + 1], float)
