'''
Author: Rajeev Baditha
Roll No: CS1523
Description: Genetic Algorithm
'''
import numpy as np
import random
import math
from itertools import zip_longest
import matplotlib.pyplot as plt

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
    
class Genetic:
    def __init__(self, L, M):
        
        self.initial_population = []
        self.fitness = []
        self.L, self.M  = L, M
        
        for i in range(0, M):
            temp = [random.choice('01') for _ in range(L)]
            self.initial_population.append(''.join(temp))
    
    def func(self, x):
        var = math.ceil(x/(2*math.pi))
        return var*math.sin(x)+6
        
    def X(self, chromosome):
        ans = 0
        for i in range(1, self.L+1):
            ans += (2**(5-i))*(ord(chromosome[i-1])-48)
        return ans
            
        
    def fitness_function(self, current):
        fitness = []
        for chromosome in current:
            fitness.append(self.func(self.X(chromosome)))
        
        return fitness
    
    def selection(self):
        self.selected = []
        s = float(sum(self.fitness))
        values = [x/s for x in self.fitness]
        
        for i in range(1, len(values)):
            values[i] += values[i-1]
            
        for i in range(0, self.M):
            p = random.random()
            for i in range(0, len(values)):
                if p <= values[i]:
                    self.selected.append(self.population[i])
                    break
    
    
    def crossover(self):
        population = list(range(0, self.M))
        self.new = []

        random.shuffle(population)
        pairs = []
        for first, second in grouper(population, 2):
            pairs.append((first, second))
        p_cross = 0.85
        
        for a, b in pairs:
            if random.random()  <=  p_cross:
                one = list(self.selected[a])
                two = list(self.selected[b])
                p = random.randrange(0, self.L-1)
                one[0:p], two[0:p] = two[0:p], one[0:p]
                self.new.append(''.join(one))
                self.new.append(''.join(two))
            else:
                self.new.append(self.selected[a])
                self.new.append(self.selected[b])
        
    
    def mutation(self, p_mut):
        
        
        for i in range(0, self.M):
            cur = list(self.new[i])
            cur = list(map(int, cur))
            temp = []
            for gene in cur:
                if random.random() < p_mut:
                    temp.append(1-gene)
                else:
                    temp.append(gene)
            
            self.new[i] = ''.join(map(str,temp))
           
    def genetic(self, generations):
        
        self.population = []
        self.population = list(self.initial_population)
        
        self.fitness = self.fitness_function(self.population)
        ans = max(self.fitness)
        s = self.population[self.fitness.index(ans)]
        num = self.X(s)
        res = []
        res.append(ans)
        
        mut = []
        mut.extend(np.linspace(0.49, 1/self.M, generations//3))
        mut.extend(np.linspace(1/self.M, 0.49, generations//3))
        mut.extend(np.linspace(0.49, 1/self.M, generations//3))
        
        for i in range(0, generations):
            
            self.selection()
            self.crossover()
            self.mutation(mut[i])
            temp = self.fitness_function(self.new)
            if max(temp) > max(self.fitness):
                self.population = list(self.new)
                self.fitness = self.fitness_function(self.population)
            else:
                self.new.pop(random.randrange(len(self.new)))
                temp1 = self.fitness.index(max(self.fitness))
                self.new.append(self.population[temp1])
                self.population = list(self.new)
                self.fitness = self.fitness_function(self.population)
            
            ans = max(self.fitness)
            s = self.population[self.fitness.index(ans)]
            num = self.X(s)
            res.append(ans)
            
        return ans, s, num, res
    
def main():
    N = 600
    name = 'genetic_detailed' + str(N) + '.txt'
    f = open(name, 'w+')
    
    for i in range(10):
        print("Iteration no %s" %(i+1))
        L, M = 15, 10;
        g = Genetic(L, M)
        
        for i in range(10):
            ans, s, num, res = g.genetic(N)
            print(ans, s, num)
            final = []
            final.append(ans)
            final.append(s)
            final.append(num)
            f.write( str(final) )
            f.write("\n")
        #f.write(str(final))
        f.write("\n\n")
        
    f.close()
'''
    plt.axis((0,301,0,12))
    plt.plot(res)
    plt.show()
    '''
     

if __name__ == '__main__':
    main()