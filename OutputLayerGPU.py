__author__ = 'Thushan Ganegedara'


import numpy as np
from theano import function, config, shared, sandbox
import theano.tensor as T

import numpy.linalg as LA
class OutputLayer(object):

    def sigmoid(self, x):
        return np.exp(x)


    def __init__(self, n_inputs, n_outputs, x_train=None, x_test=None, y=None, W1=None, b1=None, dropout=False,dropout_rate=0.5):

        self.x_train = x_train
        self.x_test = x_test
        self.y = y

        #define global variables for n_inputs and n_hidden
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.dropout = dropout
        self.dropout_rate = dropout_rate

        #generate random weights for W
        if W1 == None:
            W1 = np.random.random_sample((n_inputs,n_outputs))*0.2
            self.W1 = shared(value=W1, name='W1', borrow=True)

        #by introducing *0.05 to b1 initialization got an error dropoff from 360 -> 280
        if b1 == None:
            b1 = np.random.random_sample((n_outputs,)) * 0.2
            self.b1 = shared(value=b1, name='b1', borrow=True)

        #Remember! These are symbolic experessions.
        self.a_train = self.forward_pass(input=self.x_train, training=True)
        self.a_test = self.forward_pass(input=self.x_test, training=False)
        self.pred = self.a_test

        self.theta = [self.W1,self.b1]


    def forward_pass(self,input=None,training=False):
        if self.dropout:
            srng = T.shared_randomstreams.RandomStreams(np.random.randint(999999))
            mask = srng.binomial(n=1, p=1-self.dropout_rate, size=(self.n_inputs,))
            x_tilda = input * T.cast(mask, config.floatX)
            if training:
                a = T.nnet.sigmoid(T.dot(x_tilda, self.W1) + self.b1)
            else:
                a = T.nnet.sigmoid(T.dot(input, self.W1*(1-self.dropout_rate)) + self.b1)

        else:
            a = T.nnet.sigmoid(T.dot(input, self.W1) + self.b1)
        return a

    def get_cost(self,lam,cost_fn='neg_log'):

        #The previous cost function is faulty. It's causing the error to go up
        #instead of reducing
        if cost_fn=='sqr_err':
            cost = 0
        elif cost_fn=='neg_log':
            L = - T.sum(self.y * T.log(self.a_train) + (1 - self.y) * T.log(1 - self.a_train), axis=1)
            cost = T.mean(L) + (lam/2)*T.sum(T.sum(self.W1**2,axis=1))

        return cost

    def get_params(self):
        return self.theta

    def get_error(self,y):
        return T.mean(T.sum((self.pred-y)**2))

    def get_output(self):
        return self.pred