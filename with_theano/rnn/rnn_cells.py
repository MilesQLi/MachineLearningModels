import matplotlib.pyplot as plt
import numpy as np


import theano.tensor as T
import theano


class lstm(object):
    def __init__(self, n_in, n_h, emb, pre_fix):     
        
        self.n_in = n_in
        self.n_h = n_h
        
        self.pre_fix = pre_fix
        self.name = pre_fix + 'lstm'   
        self.params = []
        
        bound_x = np.sqrt(6. / (n_in + n_h))
        bound_h = np.sqrt(6. / (n_h + n_h))
        
        self.w_xi = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX), name=self.name+'w_xi', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_xi)
        self.w_hi = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX), name=self.name+'w_hi', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_hi)
        # self.w_ci = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        # self.params.append(self.w_ci)
        self.b_i = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX), name=self.name+'b_i', borrow=True)  # @UndefinedVariable
        self.params.append(self.b_i)
        
        self.w_xf = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX), name=self.name+'w_xf', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_xf)
        self.w_hf = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX), name=self.name+'w_hf', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_hf)
        # self.w_cf = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
        # self.params.append(self.w_cf)
        self.b_f = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX), name=self.name+'b_f', borrow=True)  # @UndefinedVariable
        self.params.append(self.b_f)
        
        self.w_xc = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX), name=self.name+'w_xc', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_xc)
        self.w_hc = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX), name=self.name+'w_hc', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_hc)
        self.b_c = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX), name=self.name+'b_c', borrow=True)  # @UndefinedVariable       
        self.params.append(self.b_c)
        
        self.w_xo = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_in, n_h)), dtype=theano.config.floatX), name=self.name+'w_xo', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_xo)
        self.w_ho = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX), name=self.name+'w_ho', borrow=True)  # @UndefinedVariable
        self.params.append(self.w_ho)
   #     self.w_co = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h, n_h)), dtype=theano.config.floatX))  # @UndefinedVariable
   #     self.params.append(self.w_co)
        self.b_o = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_h,)), dtype=theano.config.floatX), name=self.name+'b_o', borrow=True)  # @UndefinedVariable
        self.params.append(self.b_o)
        
        # self.w_y = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_h, n_out)), dtype=theano.config.floatX))  # @UndefinedVariable
        # self.params.append(self.w_y)
        # self.b_y = theano.shared(np.array(np.random.uniform(low=-bound_h, high=bound_h, size=(n_out,)), dtype=theano.config.floatX))  # @UndefinedVariable
        # self.params.append(self.b_y)
        
        self.c0 = theano.shared(np.array(np.random.uniform(low=-bound_x, high=bound_x, size=(n_h,)), dtype=theano.config.floatX), name=self.name+'c0', borrow=True)  # @UndefinedVariable
        self.params.append(self.c0)
        self.h0 = T.tanh(self.c0)
        self.h0.name = self.name+'h0'
        
        
        def one_step(x, m_, c_t1, h_t1, w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xc, w_hc, b_c, w_xo, w_ho, b_o):
            i = T.nnet.sigmoid(T.dot(x, w_xi) + T.dot(h_t1, w_hi) + b_i)  # + T.dot(c_t1, w_ci)
            f = T.nnet.sigmoid(T.dot(x, w_xf) + T.dot(h_t1, w_hf) + b_f)  # + T.dot(c_t1, w_cf)
            c = f * c_t1 + i * T.tanh(T.dot(x, w_xc) + T.dot(h_t1, w_hc) + b_c)
            ct = m_[:, None] * c + (1 - m_)[:, None] * c_t1
            o = T.nnet.sigmoid(T.dot(x, w_xo) + T.dot(h_t1, w_ho) + b_o)  # + T.dot(ct, w_co)
            h = o * T.tanh(ct)
            ht = m_[:, None] * h + (1 - m_)[:, None] * h_t1
            # y = T.nnet.sigmoid(T.dot(ht, w_y) + b_y)
            return [T.cast(ct, dtype=theano.config.floatX), T.cast(ht, dtype=theano.config.floatX)]  # , y] @UndefinedVariable
        
        self.normal_one_step = one_step
        

        def one_step_sentence(x_t, c_t1, h_t1, w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xc, w_hc, b_c, w_xo, w_ho, b_o):
            i = T.nnet.sigmoid(T.dot(x_t , w_xi) + T.dot(h_t1, w_hi) + b_i)  # + T.dot(c_t1, w_ci) 
            f = T.nnet.sigmoid(T.dot(x_t, w_xf) + T.dot(h_t1, w_hf) + b_f)  # + T.dot(c_t1, w_cf)
            c = f * c_t1 + i * T.tanh(T.dot(x_t, w_xc) + T.dot(h_t1, w_hc) + b_c)
            o = T.nnet.sigmoid(T.dot(x_t, w_xo) + T.dot(h_t1, w_ho) + b_o)  # + T.dot(c, w_co)
            h = o * T.tanh(c)
            return [c, h]  # , y]   , y]  
        
        self.sen2doc_one_step = one_step_sentence
        
        
    def get_sentence_output(self,emb_x,mask):
        [cs, hs], _ = theano.scan(fn=self.normal_one_step, sequences=[emb_x, mask], outputs_info=[T.alloc(self.c0, emb_x.shape[1], self.n_h), T.alloc(self.h0, emb_x.shape[1], self.n_h)], non_sequences=self.params[:-1])  # , None])
        return cs, hs
    
    def get_onedoc_output(self,x):        
        [cs, hs], _ = theano.scan(fn=self.sen2doc_one_step, sequences=x, outputs_info=[self.c0, self.h0], non_sequences=self.params[:-1])  # , None])
        return cs, hs

