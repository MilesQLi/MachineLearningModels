# coding=utf-8

import cPickle
import gzip
from sklearn import preprocessing
import sys
import time

import numpy
import theano

import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T


def gd(cost, params, learning_rate, decay = 0.1, momentum = 0.1):
            iteration = theano.shared(np.array(0.))  # @UndefinedVariable
            grads = T.grad(cost, params)
            vs = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            lr = learning_rate / (1. + decay*iteration)
            updates =  [(param, param - momentum * v - lr * grad) for param,v, grad in zip(params,vs, grads)]
            updates +=  [(v, momentum * v + lr * grad) for v, grad in zip(vs, grads)]
            updates.append((iteration,iteration+1))
            return updates
def adagrad(cost, params, learning_rate):
            grads = T.grad(cost, params)
            accumulators = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            updates = []
            for p, g, a in zip(params, grads, accumulators):
                a_new = a + g ** 2
                updates.append((a, a_new))
                p_new = p - learning_rate * g / T.sqrt(a_new + 0.01)
                updates.append((p, p_new))
            return updates
        
def rmsprop(cost, params, learning_rate, rho = 0.1):
            grads = T.grad(cost, params)
            accumulators = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            delta_accumulators = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            updates = []
            for p, g, a, d in zip(params, grads, accumulators, delta_accumulators):
                a_new = rho*a +(1-rho)* g ** 2
                updates.append((a, a_new))
                p_new = p - learning_rate*g / T.sqrt(a_new + 0.01)
                updates.append((p, p_new))
            return updates

'''
def adam(cost, params,  learning_rate = 0.001,kappa = 1-1e-8, beta1 = 0.9, beta2 = 0.999, eta = 1e-8):
        grads = T.grad(cost, params)
        iterations = theano.shared(np.array(0.))  # @UndefinedVariable
        updates = [(iterations, iterations+1.)]

        i = iterations
        beta_1_t = beta1 **i

        # the update below seems missing from the paper, but is obviously required
        beta_2_t = beta2 **i 

        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.) # zero init of moment
            v = theano.shared(p.get_value() * 0.) # zero init of velocity

            m_t = (beta_1_t * m) + (1 - beta_1_t) * g
            v_t = (beta_2_t * v) + (1 - beta_2_t) * (g**2)

            m_b_t = m_t / (1 - beta_1_t)
            v_b_t = v_t / (1 - beta_2_t)

            p_t = p - learning_rate * m_b_t / (T.sqrt(v_b_t) + eta)
            
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((p, p_t)) # apply constraints
        return updates
'''
def adam(cost, params,  learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, eta = 1e-8):
            grads = T.grad(cost, params)
            ms = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            vs = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            iteration = theano.shared(np.array(0.))  # @UndefinedVariable
            updates = []
            updates.append((iteration,iteration+1))
            for p, g, m, v in zip(params, grads, ms, vs):
                m_new = beta1*m + (1-beta1) * g
                v_new = beta2*v + (1-beta2)*g**2
                m_ka = m_new / (1-beta1**iteration)
                v_ka = v_new / (1-beta2**iteration)
                delta_theta = learning_rate *m_ka/(T.sqr(v_ka)+eta) 
                
                updates.append((m, m_new))
                updates.append((v, v_new))
                updates.append((p, p - delta_theta))
            return updates

                
def adadelta(cost, params, rho = 0.05):
            grads = T.grad(cost, params)
            accumulators = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            delta_accumulators = [theano.shared(np.array(np.zeros(p.get_value().shape), dtype=theano.config.floatX)) for p in params]  # @UndefinedVariable
            updates = []
            for p, g, a, d in zip(params, grads, accumulators, delta_accumulators):
                a_new = rho*a +(1-rho)* g ** 2
                updates.append((a, a_new))
                delta =   g *T.sqrt(d+0.01)/ T.sqrt(a_new + 0.01)
                p_new = p - delta
                updates.append((p, p_new))
                updates.append((d,rho*d +(1-rho)* delta ** 2 ))
            return updates