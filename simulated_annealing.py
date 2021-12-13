import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.datasets import load_digits
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import arff
from collections import defaultdict
from itertools import count
from functools import partial

class SAnneal:
    def __init__(self, X, y, choosen, total, T):
        self.current = random.sample(range(0, total), choosen)
        self.total = range(0, total)
        self.unvisited = list(set(self.total) - set(self.current))
        self.X = X
        self.y = y
        self.T = T
        self.best = list(self.current)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.ans = []
        self.ans.append(self.fitness(self.best))

    def generate_new(self):
        temp1 = random.randrange(0, len(self.current))
        self.unvisited = list(set(self.total) - set(self.current))
        temp2 = random.randrange(0, len(self.unvisited))
        temp = list(self.current)
        temp[temp1], self.unvisited[temp2] = self.unvisited[temp2], temp[temp1]
        self.next = list(temp)
    
    def fitness(self, features):
        X_train = self.X_train[:,features]
        X_test = self.X_test[:,features]
        clf = KNeighborsClassifier(n_neighbors=3)
        #clf = svm.SVC()
        clf.fit(X_train, self.y_train)
        predicted = clf.predict(X_test)
        return accuracy_score(self.y_test, predicted)

    
    def check(self, new, cur):
        t = self.T
        p = np.exp(-1*(cur-new)/t)
        if np.random.rand() <= p:
            return True
        else:
            return False
    
    def sanneal(self, generations):
        counter = []
        counter.append(self.T)
        for i in range(0, generations):

            self.T = 0.995*self.T
            counter.append(self.T)
            self.generate_new()

            new, cur, best = self.fitness(self.next), self.fitness(self.current), self.fitness(self.best)
                      
                
            if new > cur:
                self.current = list(self.next)
                if new > best:
                    self.best = list(self.next)

            elif self.check(new, cur):
                self.current = list(self.next)
            self.ans.append(max(new, cur, best))
        #plt.plot(counter)
        return self.ans

def main():
    digits = load_digits()
    X = digits.data
    y = digits.target
    '''
    data = arff.load(open('check.arff', 'r'))
    data = data[u'data']
    data = np.array(data)
    X = data[:,:-1]
    y = data[:,data.shape[1]-1]
    label_to_number = defaultdict(partial(next, count(1)))
    y = [label_to_number[label] for label in y]
    y = np.array(y)
    '''
    sa = SAnneal(X, y, 8, X.shape[1],100000)

    generations = 1000
    gen = range(0, generations) 
    ans = sa.sanneal(generations)
    print(gen)
    print(ans)
    plt.plot(ans)
    

if __name__ == '__main__':
	main()