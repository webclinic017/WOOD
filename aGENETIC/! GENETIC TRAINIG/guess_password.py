# File: guessPasswordTests.py
#    from chapter 1 of _Genetic Algorithms with Python_, an ebook
#    available for purchase at http://leanpub.com/genetic_algorithms_with_python
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Repository: https://drive.google.com/open?id=0B2tHXnhOFnVkRU95SC12alNkU2M
# Copyright (c) 2016 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
import datetime
import genetic
import unittest
import random

def get_fitness(guess, target):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)

def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime 
    print("\r{0}\t{1}\t{2}".format(candidate.Genes, candidate.Fitness, str(timeDiff)),end='')

class GuessPasswordTests():
    geneset = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,"
    
    def Hello_World(self):
        target = "Hello World!"
        self.guess_password(target)
    
    def For_I_am_fearfully_and_wonderfully_made(self): 
        target = "For I am fearfully and wonderfully made." 
        self.guess_password(target)
    
    def guess_password(self, target): 
        startTime = datetime.datetime.now()
    
        def fnGetFitness(genes):
            return get_fitness(genes, target)
        
        def fnDisplay(candidate): 
            display(candidate, startTime)
        
        optimalFitness = len(target)
        best = genetic.get_best(fnGetFitness, len(target),optimalFitness, self.geneset, fnDisplay)
        #self.assertEqual(best.Genes, target)
    
    def test_Random(self): 
        length = 150
        target = ''.join(random.choice(self.geneset) for _ in range(length))
        self.guess_password(target) 
        
    def test_benchmark(self):
        genetic.Benchmark.run(self.test_Random)

if __name__ == '__main__': 
    pass