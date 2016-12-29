# coding=utf-8
# Author: Qi Li <qi.li@pku.edu.cn>
# License: BSD 3 clause


import numpy as np
import theano.tensor as T
import rnn_cells


class hie_rnn(object):
    def __init__(self, n_in, n_h, emb, pre_fix, rnn_cell='lstm', n_h_s=None):
        self.x = T.imatrix('hie_rnn_x')
        self.mask = T.imatrix('hie_rnn_mask')
        
        
        self.name = 'hie_rnn'
        
        self.inputs = [self.x, self.mask]
        
        self.n_sen = T.iscalar()
        self.n_doc = T.iscalar()
        self.s_mask = T.imatrix('hie_rnn_s_mask')
        
        self.batch_inputs = [self.x, self.mask,self.n_sen,self.n_doc,self.s_mask]
        
        self.emb = emb
        
        
        self.pre_fix = pre_fix
        self.name = pre_fix + rnn_cell   
        
        
        if n_h_s == None:
            n_h_s = n_h
        
        self.sentence_lstm = getattr(rnn_cells, rnn_cell)(n_in, n_h, emb, pre_fix+'sen')
        self.doc_lstm = getattr(rnn_cells, rnn_cell)(n_h, n_h_s, emb, pre_fix+'doc')
        
        self.params = self.sentence_lstm.params+self.doc_lstm.params
        
        _,hs = self.sentence_lstm.get_sentence_output(self.emb[self.x],self.mask)
        
        cs_s, hs_s = self.doc_lstm.get_onedoc_output(hs[-1])
        self.s_ho = hs_s[-1]
        self.s_co = cs_s[-1]
        
        
        to_sentence = hs[-1]
        to_sentence = to_sentence.reshape((self.n_doc,self.n_sen,n_h))
        to_sentence = to_sentence.dimshuffle(1,0,2)
          
          
        self.to_sentence = to_sentence
        
        cs_s, hs_s = self.doc_lstm.get_sentence_output(self.to_sentence, self.s_mask)
         
        self.h_o = hs_s[-1]
        self.c_o = cs_s[-1]      