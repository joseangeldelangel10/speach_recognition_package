import numpy
from numpy_matrix_list import numpy_matrix_list

class neuralNetwork():
    """ neuralNetwork class is a class designed to easily create
    a multilayer perceptron that is able to classify any input 
    based on its training.
    This class was designed by Jose Angel Del Angel Dominguez 2021
    and its inspired by Ahmed Gad's article named 'Artificial 
    Neural Network Implementation using NumPy and Classification 
    of the Fruits360 Image Dataset' [2019] found in:
    https://towardsdatascience.com/artificial-neural-network-
    implementation-using-numpy-and-classification-of-the-fruits360
    -image-3c56affa4491
    """
    def __init__(self, inital_weights, trainX, trainY, learning_rate):
        self.weights = inital_weights
        self.trainX = trainX
        self.trainY = trainY
        self.learning_rate = learning_rate
        self.output = None
        self.layers = len(self.weights.list)
        self.len_training_samples = len(self.trainX)
        self.deltas = self.build_deltas()
        self.a = []


    def build_deltas(self):
        res = []
        res.insert(0,  numpy.zeros( self.trainY[0].shape )  ) #d4
        for i in range(self.layers-1):
            res.insert(0,  numpy.zeros( ( 1 , self.weights[self.layers-i-1].shape[0] ) )  ) #d3,d2
        return res

    def sigmoid(self,inpt):
        return 1.0 / (1 + numpy.exp(-1 * inpt))

    def round(self,inpt):
        result = inpt == inpt.max()
        result = result*1.0
        return result

    def update_weights(self,i):
        cost_derivative = []
        for l in range(self.layers):
            aT = self.a[l].transpose()
            d = self.deltas[l]
            derivative = numpy.matmul(aT, d)
            cost_derivative.append(derivative)
        cost_derivative = numpy_matrix_list(cost_derivative)
        
        # we perform gradient descent: 
        new_weights = self.weights - cost_derivative.scalar_mult(self.learning_rate).scalar_mult(1/self.len_training_samples)
        self.weights = new_weights

    def train_network(self, num_iterations):
        #self.trainX = numpy.insert(self.trainX,0,1,1) # we insert bias term
        for iteration in range(num_iterations):
            print("Itreation ", iteration)
            for sample_idx in range(self.len_training_samples):
                self.forward_propagation(sample_idx)
            print("Error: {e}".format(e = self.total_cost_function()))


    def forward_propagation(self, i):
        r1 = numpy.array([self.trainX[i, :]])
        self.a = []
        self.a.append(r1)
        for l in range(self.layers-1):
            curr_weights = self.weights[l]
            r1 = numpy.matmul(r1, curr_weights)
            r1 = self.sigmoid(r1)
            self.a.append(r1)            
        curr_weights = self.weights[-1]            
        r1 = numpy.matmul(r1, curr_weights)
        r1 = self.sigmoid(r1)
        self.a.append(r1)
        self.output = r1
        predicted_label = self.round(r1)
        desired_label = self.trainY[i]
        if not numpy.all(predicted_label == desired_label):
            self.backprop(i)

    def backprop(self, i):
        layers = self.layers
        for l in range(layers):
            if l == 0:
                self.deltas[layers-1-l] = self.output-self.trainY[i]
            else:
                thetaN = self.weights[layers-l]
                deltaN = self.deltas[layers-l]
                deltaNT = deltaN.transpose()    
                mul = numpy.matmul(thetaN, deltaNT)
                g_prime = numpy.array(self.a[layers-l]*(numpy.ones(self.a[layers-l].shape)-self.a[layers-l]))
                self.deltas[layers-1-l] = mul.transpose()*g_prime
        
        self.update_weights(i)

    def predict_outputs(self, trainX):
        predictions = []
        for sample_idx in range(trainX.shape[0]):
            r1 = numpy.array([trainX[sample_idx, :]])
            for curr_weights in self.weights.list:
                r1 = numpy.matmul(r1,curr_weights)
                r1 = self.sigmoid(r1)
            predicted_label = self.round(r1)
            predictions.append(predicted_label.tolist()[0])
        return numpy.array(predictions)

    def total_cost_function(self):
        res = 0
        m = self.len_training_samples
        for i in range(self.len_training_samples):
            res += self.cost_function(i)            
        return -1/m * res  

    def cost_function(self, test_num):
        i = test_num
        res = 0
        r1 = numpy.array([self.trainX[i, :]])
        for l in range(self.layers-1):
            curr_weights = self.weights[l]
            r1 = numpy.matmul(r1, curr_weights)
            r1 = self.sigmoid(r1)            
        curr_weights = self.weights[-1]            
        r1 = numpy.matmul(r1, curr_weights)
        h = self.sigmoid(r1)
        y = self.trainY
        
        for k in range(len(self.trainY[0])):
            term = y[i,k]*numpy.log10(h)+(1-y[i,k])*numpy.log10(1-h)
            res += term.sum()

        return res