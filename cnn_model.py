import numpy
from numpy_matrix_list import numpy_matrix_list

def sigmoid(inpt):
    return 1.0 / (1 + numpy.exp(-1 * inpt))

def relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    return result

def round(inpt):
    result = inpt
    result[inpt < 0.5] = 0
    result[inpt >= 0.5] = 1
    return result

def update_weights(weights, learning_rate):
    new_weights = weights - weights.scalar_mult(learning_rate)
    return new_weights

def train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation="relu"):
    for iteration in range(num_iterations):
        print("Itreation ", iteration)
        for sample_idx in range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :] # nth row of the xtrain data
            #print("\n WEIGHTS ARE: {f} \n".format(f = str(weights)))
            for idx in range(len(weights.list)-1):
                curr_weights = weights[idx]
                #print("\n curr_weights are {f} \n".format(f = curr_weights))
                r1 = numpy.matmul(r1, curr_weights)
                if activation == "relu":
                    r1 = relu(r1)
                    #print("\n r1 is: {f} \n".format(f = r1))
                elif activation == "sigmoid":
                    r1 = sigmoid(r1)            
            curr_weights = weights[-1]
            r1 = numpy.matmul(r1, curr_weights)
            predicted_label = round(r1)
            desired_label = data_outputs[sample_idx]
            if not numpy.all(predicted_label == desired_label):
                weights = update_weights(weights, learning_rate=0.001)
    print("predicted_label is: {f}".format(f = predicted_label))
    print("desired_label is: {f}".format(f = desired_label))
    return weights

def predict_outputs(weights, data_inputs, activation="relu"):
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        for curr_weights in weights:
            r1 = numpy.matmul(a=r1, b=curr_weights)
            if activation == "relu":
                r1 = relu(r1)
            elif activation == "sigmoid":
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    return predictions

