import numpy
from numpy_matrix_list import numpy_matrix_list

class neuralNetwork():
    """docstring for NeuralNetwork"""
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
        #print("before update_weights a and delta shapes are:")
        #print(self.a)
        #print("=========================================================")
        #print(self.deltas)
        cost_derivative = []
        for l in range(self.layers):
            aT = self.a[l].transpose()
            d = self.deltas[l]
            derivative = numpy.matmul(aT, d)
            cost_derivative.append(derivative)
        cost_derivative = numpy_matrix_list(cost_derivative)
        #print("cost_derivative len is {n}".format(n=len(cost_derivative.list)))
        #print("cost_derivative len is {n}".format(n=cost_derivative[0].shape))
        
        # theta = theta - a*deriv_cost(theta)
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
        r1 = self.trainX[i, :] # nth row of the xtrain data
        #print("shape of r1 starting fp is {s}".format(s = r1.shape))
        #r1 = numpy.insert(r1,0,1,1) # we add bias term
        self.a = []
        self.a.append(numpy.array([r1]))
        for l in range(self.layers-1):
            curr_weights = self.weights[l]
            #print("shape of r1 after bias is {s}".format(s = r1.shape))
            r1 = numpy.matmul(r1, curr_weights)
            r1 = self.sigmoid(r1)
            #r1 = numpy.insert(r1,0,1,1) # we add bias term
            self.a.append(numpy.array([r1]))            
        curr_weights = self.weights[-1]            
        r1 = numpy.matmul(r1, curr_weights)
        r1 = self.sigmoid(r1)
        #r1 = numpy.insert(r1,0,1,1) # we add bias term
        self.a.append(numpy.array([r1]))
        self.output = r1
        predicted_label = self.round(r1)
        desired_label = self.trainY[i]
        #print("predicted_label is: {f}".format(f = predicted_label))
        #print("desired_label is: {f}".format(f = desired_label))
        if not numpy.all(predicted_label == desired_label):
            #print("before entering backprop a is: {n}".format(n=self.a))
            self.backprop(i)
            #weights = update_weights(weights, learning_rate)

    def backprop(self, i):
        layers = self.layers
        for l in range(layers):
            if l == 0:
                self.deltas[layers-1-l] = self.output-self.trainY[i]
                self.deltas[layers-1-l] = numpy.array([self.deltas[layers-1-l]])
            else:
                thetaN = self.weights[layers-l]
                #print("\n theta shape is {n}".format(n=thetaN.shape))
                deltaN = self.deltas[layers-l]
                #print("\n delta shape is {n}".format(n=deltaN.shape))
                deltaNT = deltaN.transpose()    
                mul = numpy.matmul(thetaN, deltaNT)
                #print("\n mul shape is {n}".format(n=mul.shape))
                g_prime = numpy.array(self.a[layers-l]*(numpy.ones(self.a[layers-l].shape)-self.a[layers-l]))
                self.deltas[layers-1-l] = mul.transpose()*g_prime
        
        #self.deltas_to_np_array()
        #self.deltas = self.transpose_matrix(deltas) #?????
        self.update_weights(i)

    def deltas_to_np_array(self):
        temp = []
        for i in range(len(self.deltas)):
            temp.append(self.deltas[i].tolist())
        self.deltas = numpy.array(temp)



    def transpose_matrix(self, matrix):
        mat = matrix.tolist()
        res = self.empty_matrix(len(mat[0]), len(mat))
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                res[j][i] = mat[i][j]
        return numpy.array(res)

    def empty_matrix(self,n,m):
        res = []
        for i in range(n):
            row = []
            for j in range(m):
                row.append(0)
            res.append(row)
        return res

    def predict_outputs(self, trainX):
        predictions = []
        for sample_idx in range(trainX.shape[0]):
            r1 = trainX[sample_idx, :]
            for curr_weights in self.weights.list:
                r1 = numpy.matmul(r1,curr_weights)
                r1 = self.sigmoid(r1)
            predicted_label = self.round(r1)
            #predicted_label = r1
            predictions.append(predicted_label.tolist())
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
        r1 = self.trainX[i,:]
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

    @staticmethod
    def add_bias_terms(inp, weights):
        res = []
        res.append(numpy.insert(inp,0,1,1))
        biased_weights = []
        #for i in range(len(weights.list)):
        for i in range(1):
            biased_weights.append( numpy.insert(weights[i],0,1,0) )
        res.append(numpy_matrix_list(biased_weights))
        return res


