from typing import Tuple
import numpy as np
from utils import binary_encode, binary_decode


class SimpleGeneticAlgorithm:
    def __init__(self, func: function, encoder: function = None, 
                 decoder: function = None, init_num: int = 20, mut_prob=0.05,
                 ranges: Tuple[float, float] = (0., 1.), iters: int = 100):
        """Simple Genetic Algorithm for single function and univariate optimize problem without constraint.
        1. initialize individuals
        2. compute fitness
        3. choose individuals
        4. crossover operator
        5. mutation operator
        Args:
            func (function): fitness function
            encoder (function): encode methods
            decoder (function): decode methods
            init_num (int): initialize individuals
            ranges: Tuple(float)
            iters (int): iteration times
        """
        self.func = func
        self.encoder = encoder or binary_encode
        self.decoder = decoder or binary_decode
        self.init_num = init_num
        self.mut_prob = mut_prob
        self.ranges = ranges
        self.iters = iters
    
    def fit(self):
        population = self.__init_individual()
        for _ in range(self.iters):
            scores = self.__evaluate(population)
            population = self.__choose(population, scores)
            population = self.__crossover(population)
            population = self.__mutation(population)
        scores = self.__evaluate(population)
        return population[np.argmax(scores)]
            
    
    def __init_individual(self):
        population = np.random.uniform(self.ranges[0], self.ranges[1], size=self.init_num)
        return population
    
    def __evaluate(self, population):
        scores = []
        population = self.decoder(population)
        for individual in population:
            score = self.func(individual)
            scores.append(score)
        return np.array(scores)
    
    def __choose(self, population, scores):
        probs = scores / np.sum(scores)
        index = np.argsort(probs)
        probs = np.cumsum(probs[index])
        probs = np.concatenate([np.zeros(1), probs])
        out = []
        for i in range(self.init_num // 2):
            p = np.random.uniform(0., 1.)
            for j in range(len(probs)-1):
                if probs[j] <= p < probs[j+1]:
                    break
            out.append(population[j])
        return np.array(out)

    def __crossover(self, population):
        population = self.encoder(population)
        N = len(population)
        index = list(range(N))
        np.random.shuffle(index)
        for i in range(N):
            pass
        return population
    
    def __mutation(self, population):
        for ind in population:
            for i in ind:
                if np.random.uniform(0., 1.) < self.mut_prob:
                    ind[i] = 1 - ind[i]
        population = self.decoder(population)
        return population
    
    
    
    