import numpy as np
from math import exp
from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

#def get_classes():
	
class NeuralNetwork:
    def __init__(self, layers, activation):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        i = 0
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i]
                                + 1))-1)*0.25)
        self.weights.append((2*np.random.random((layers[i] + 1, layers[i +
                            1]))-1)*0.25)
    
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            #print(a)
            error = y[i] - a[-1]
            #print(y[i])
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def rbfnn(X, y):
    c = len(set(y))
    nodes = 8*c
    kmeans = KMeans(n_clusters=nodes, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    #print(len(centers))
    X_new = []
    for i in range(len(X)):
        current = X[i]
        temp = []
        for j in range(len(centers)):
        #print(X[i]-centers[i])
            temp.append(exp(-1*np.linalg.norm(current-centers[j])))
        X_new.append(temp)
	
    return X_new,nodes
 
 
def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X -= X.min() # normalize the values to bring them into the range 0-1
    X /= X.max()
    X, nodes = rbfnn(X,y)
    X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    labels_train = LabelBinarizer().fit_transform(y_train)
    
    nn = NeuralNetwork([nodes,10],'logistic')
	#labels_test = LabelBinarizer().fit_transform(y_test)
    #print(y_train[10])
    nn.fit(X_train,labels_train,epochs=30000)
    predictions = []
    for i in range(X_test.shape[0]):
        o = nn.predict(X_test[i] )
        #print(o,len(o))
        predictions.append(np.argmax(o))
    temp = confusion_matrix(y_test,predictions)
    print(temp)
    ans = np.array(temp)
    print('correctly matched patterns', ans.trace())
    print('total patterns',sum(sum(ans)))
    print('accuracy', ans.trace()/sum(sum(ans)))
    #print(classification_report(y_test,predictions))

if __name__ == '__main__':
	main()