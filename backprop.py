import numpy as np
from numpy import array

W=[]
b=[]
# Initial parameters
input_dim = 8   
output_dim = 4  
epsilon = 0.00002  
reg_lambda = 0.3   
hidden_dim = 14
hidden_layers = 2
num_passes = 25000

# Helper function to evaluate the total loss on the dataset
def total_error(X, y):
    # Forward propagation
    num_layers = len(W)-1
    num_samples = len(X)

    act_input={}
    act_val={}
    act_input[0] = X.dot(W[0]) + b[0]
    act_val[0] = np.tanh(act_input[0])

    for i in range(num_layers-1) :
        act_input[i+1] = act_val[i].dot(W[i+1]) + b[i+1]
        act_val[i+1] = np.tanh(act_input[i+1])

    act_input[num_layers] = act_val[num_layers-1].dot(W[num_layers]) + b[num_layers]

    # Calculating the probability using softmax function
    softmax_val = np.exp(act_input[num_layers])
    probs = softmax_val / np.sum(softmax_val, axis=1, keepdims=True)
    # Calculating the loss using cross entropy loss function
    corect_logprobs = -np.log(probs[range(num_samples), y])
    # print corect_logprobs
    # print len(corect_logprobs)
    data_loss = np.sum(corect_logprobs)
    # print data_loss
    # Adding regularization to loss using L2-norm loss function
    # print W[0]
    reg_sum = np.sum(np.square(W[0]))
    # print reg_sum
    for i in range(num_layers) :
        reg_sum += np.sum(np.square(W[i+1]))
    # print reg_sum
    data_loss += reg_lambda / 2 * ( reg_sum )
    # print num_samples
    # print data_loss
    # print data_loss / float(num_samples) 
    return data_loss / float(num_samples) 

# X- features and Y- labels, and the number of passes.
def train_network(X, y, num_passes=20000):
    # Initialize the to random values. We need to learn these.
    num_samples = len(X)
    # set the bias to zero and the weights to random.
    np.random.seed(0)

    # Initializing weights and biases

    W.append(np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
    b.append(np.zeros((1, hidden_dim)))
    for i in range(hidden_layers-1) :
        W.append(np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        b.append(np.zeros((1, hidden_dim)))
    W.append(np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim))
    b.append(np.zeros((1, output_dim)))



    # updating the network num_pass times
    for j in xrange(num_passes):
        act_input={}
        act_val={}

        # Forward propagation
        act_input[0] = X.dot(W[0]) + b[0]
        act_val[0] = np.tanh(act_input[0])
        
        #hidden activations
        for i in xrange(hidden_layers-1) :
            act_input[i+1] = act_val[i].dot(W[i+1]) + b[i+1]
            act_val[i+1] = np.tanh(act_input[i+1])
        #output nodes.
        act_input[hidden_layers] = act_val[hidden_layers-1].dot(W[hidden_layers]) + b[hidden_layers]

        #calculating the probability of each class.
        #using soft-max function-
        # Since the classes are dependent
        softmax_val = np.exp(act_input[hidden_layers])
        probs = softmax_val / np.sum(softmax_val, axis=1, keepdims=True)


        # Backpropagation
        delta = {}
        del_Weight = {}
        del_bias = {}
        delta[hidden_layers] = probs
        #calculate the error(delta) i.e the (prob of predicted class - 1)
        # print delta[hidden_layers],"Before "
        delta[hidden_layers][range(num_samples), y] -= 1
        # print delta[hidden_layers],"AAfter"
      
        del_Weight[hidden_layers] = (act_val[hidden_layers-1].T).dot(delta[hidden_layers])
        del_bias[hidden_layers] = np.sum(delta[hidden_layers], axis=0, keepdims=True)
        
        for i in reversed(range(hidden_layers-1) ) :
            delta[i+1] = delta[i+2].dot(W[i+2].T) * (1 - np.power(act_val[i+1], 2))
            del_Weight[i+1] = np.dot(act_val[i].T, delta[i+1])
            del_bias[i+1] = np.sum(delta[i+1], axis=0)

        delta[0] = delta[1].dot(W[1].T) * (1 - np.power(act_val[0], 2))
        del_Weight[0] = np.dot(X.T, delta[0])
        del_bias[0] = np.sum(delta[0], axis=0)

        # Add regularization terms 

        for i in range(hidden_layers+1) :
            del_Weight[i] += reg_lambda * W[i]
            W[i] += -epsilon * del_Weight[i]
            b[i] += -epsilon * del_bias[i]

        # print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if j % 1000 == 0:
            act_input = total_error(X, y)
            print("roundCount %i: Error: %f" % (j, act_input))
            if (act_input < 0.1) : 
                print "Error less than minimum, exiting.. "
                break

    return 

def test_network(validationData, validationLabel):

    validCount = 0
    num_layers = len(W)-1
    act_input={}
    act_val={}
    act_input[0] = validationData.dot(W[0]) + b[0]
    act_val[0] = np.tanh(act_input[0])


    for i in range(num_layers-1) :
        act_input[i+1] = act_val[i].dot(W[i+1]) + b[i+1]
        act_val[i+1] = np.tanh(act_input[i+1])

    act_input[num_layers] = act_val[num_layers-1].dot(W[num_layers]) + b[num_layers]

    #print act_input2
    softmax_val = np.exp(act_input[num_layers])
    probs = softmax_val / np.sum(softmax_val, axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)

    for p in xrange(len(predictions)):
        if predictions[p] == validationLabel[p]:
            validCount += 1
    print "correctly predicted:", str(validCount), " --> ", str(validCount * 100.0 / len(validationData)) + "%"

    return validCount / len(validationData) 


def gettraindata():
    data = open("data-train_2.txt")
    features = []
    labels = []
    for line in data:
        line_data = [float(d) for d in line.split()]
        features.append(line_data[1:9])
        labels.append(int(line_data[9]) -1)
    return array(features), array(labels)

def gettestdata():
    data = open("data-train_2.txt")
    features = []
    labels = []
    for line in data:
        line_data = [float(d) for d in line.split()]
        features.append(line_data[1:9])
        labels.append(int(line_data[9]) -1)
    return array(features), array(labels)

def setParams(inputNodes = 8, outputNodes = 4, hiddenNodes = 12, hiddenLayers = 2, alpha = 0.00001, reg = 0.1, passes = 25000):
    input_dim = inputNodes       
    output_dim = outputNodes     
    hidden_dim = hiddenNodes
    hidden_layers = hiddenLayers
    epsilon = alpha               
    reg_lambda = reg         
    num_passes = passes

def main():

    #get the data from file 
    train_data, training_label = gettraindata()
    test_data, test_label = gettestdata()


    #building the model using the entire train dataset
    train_network(train_data[:6000], training_label[:6000], num_passes)
    #test the constructed network with test_data
    test_network(train_data[:6000], training_label[:6000])
    test_network(test_data[6000:], test_label[6000:])

if __name__ == "__main__":
    main()  