# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 23:46:48 2016

@author: ajinkya

My Neural Net recreated

"""

import numpy as np
import matplotlib.pyplot as plt
import sys



class NeuralNetwork:
    
    def __init__(self):
        
        self.X = np.array(([3,5], [2,6]), dtype=float)
        self. y = np.array(([23],[29]), dtype=float)
        self.X = self.X/np.amax(self.X, axis=0)
        self.y = self.y/100 #Max test score is 100

        self.inputLayerSize = 2
        self.hiddenLayerSize = 3
        self.outputLayerSize = 1

   
        
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)

        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)    
           
                              
    
        self.myerror = []
        self.myerror2 = []
           
    
    def forward(self,X,y):
                
        #print(self.W1[0].T)

        #print(np.shape(X))


        #print(X, y)

        #print("W1", self.W1)
        #print("W2", self.W2)
          
        self.xj = np.dot(self.X,self.W1)
        
        #xj = (xj[0]+xj[1])
       
        #print("xj", xj)

        
        self.Oj = self.sigmoid(self.xj)
        
        #print("Oj ", self.Oj)

        
        self.xk = np.dot(self.Oj, self.W2)

        self.Ok = self.sigmoid(self.xk)        

        #print("Ok " , self.Ok)
        
            
        
        return self.Ok

    def sigmoid(self, x):
        return (1.0/(1.0+np.exp(-x)));

    def sigmoidPrime(self, x):
        return (self.sigmoid(x)*(1-self.sigmoid(x)))


    def costFunction(self):
        self.Ok = self.forward(self.X, self.y)

        J = 0.5*sum((self.y - self.Ok)**2)
        return J
        
    def backprop(self):
        
        #going forward first        midIJ = np.dot(deltaK, self.W2.T)

        self.Ok = self.forward(self.X, self.y)

        #if dJdW1 = pos, then take step in pos, and accordingly for dJdW2 
        # as long as the eventual error is less than the prescribed value        
        
        #calculating dJdW2
        
        
        
        #print(self.Ok)        
        
        
        
        deltaK = np.multiply( -(self.y-self.Ok), self.sigmoidPrime(self.xk))
        
        #print(np.shape((self.Ok-y).T), np.shape(self.Ok), np.shape(y), np.dot((self.Ok-y).T, self.Ok))        

        #deltaK = np.multiply(, deltaK)
        
        dJdW2 = np.dot(self.Oj.T, deltaK)
        
        
        #print("dJdW2 \n",dJdW2)   
        
        
                
        
        #print(np.shape(midIJ), np.shape(self.Oj))
        
        deltaJ = np.dot(deltaK, self.W2.T)*self.sigmoidPrime(self.xj)
        
        
        dJdW1 = np.dot(self.X.T, deltaJ)
        #print(np.shape(midIJ))        
        
        #print((self.Ok-y).T, self.Ok, np.shape(deltaJ), np.shape(X), np.shape(y), np.shape(self.Ok), np.shape((self.Ok-y).T))
        
        







        #print("dJdW1", dJdW1)
        
        return dJdW1, dJdW2
        
        
    # list to store error values at each iteration
        
      
    # train the network    
        
    def train(self):
        


        maxIter = 500
        stepSize = 0.1
        i = 0
        
        dW1 = []
        dW2 = []
        W1arr = []
        W2arr = []
        #NN.costFunction() > 0.01
        while(NN.costFunction() > 0.0001):
            
            
            
            dJdW1, dJdW2 = NN.backprop()
    
   
            self.W1 -= 10*dJdW1
            self.W2 -= 10*dJdW2
            
            dW1.append(dJdW1)
            dW2.append(dJdW2)
            W1arr.append(self.W1)
            W2arr.append(self.W2)
            print(NN.costFunction())
            print("w1",self.W1)
            print("w2",self.W2)
            print("djdw1",dJdW1)
            print("djdw2",dJdW2)
            print("Ok", self.Ok)
            print(NN.forward(self.X, self.y))
            print("---------------------")
            i += 1
            NN.plotMy(dW1, W1arr, dW2, W2arr)



    def plotMy(self, dW1, W1arr, dW2, W2arr):
        
        plt.subplot(121)
        
        
        plt.scatter(np.array(W1arr), np.array(dW1))


        plt.subplot(122)
        
        
        plt.scatter(np.array(W2arr), np.array(dW2))
        plt.show()
        
#Run the net
        
NN = NeuralNetwork()

NN.train()

print(NN.costFunction(), NN.Ok, NN.y)

#NN.backprop()

#NN.train()