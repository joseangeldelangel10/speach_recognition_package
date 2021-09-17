import numpy as np

def generate_simple_test_data():
    '''function that genrates simple 3x3 bit images that will be used
    to debug the neural network, neural network will check 
    if the image has a dot in the center or outside'''
    x_tests = []
    test1 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test1 = test1.reshape(-1)
    test1 = test1.tolist()
    x_tests.append(test1)
    test2 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test2 = test2.reshape(-1)
    test2 = test2.tolist()
    x_tests.append(test2)
    test3 = np.array([ [0,0,0],[0,0,0],[1,0,0] ])
    test3 = test3.reshape(-1)
    test3 = test3.tolist()
    x_tests.append(test3)
    test4 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test4 = test4.reshape(-1)
    test4 = test4.tolist()
    x_tests.append(test4)
    test5 = np.array([ [1,0,0],[0,0,0],[0,0,0] ])
    test5 = test5.reshape(-1)
    test5 = test5.tolist()
    x_tests.append(test5)
    test6 = np.array([ [0,1,0],[0,0,0],[0,0,0] ])
    test6 = test6.reshape(-1)
    test6 = test6.tolist()
    x_tests.append(test6)
    test7 = np.array([ [0,0,0],[0,1,0],[0,0,0] ])
    test7 = test7.reshape(-1)
    test7 = test7.tolist()
    x_tests.append(test7)
    test8 = np.array([ [0,0,0],[0,0,0],[0,1,0] ])
    test8 = test8.reshape(-1)
    test8 = test8.tolist()
    x_tests.append(test8)
    y_tests = [[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[1,0],[0,1]]

    return [x_tests, y_tests]