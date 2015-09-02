#coding=utf-8

'''
objective function for classification:                


        
    def cross_entropy(self,y):
        return T.mean(-y * T.log(self.y) - (1-y) * T.log(1-self.y))
        
    for softmax regression:            
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y)[T.arange(y.shape[0]), y])
    

    for logistic regression:
        def negative_log_likelihood(self,y):
        return -T.mean(T.log(T.abs_(self.y - (1 - y))))   
    
'''

'''
error function for classification:

    def error(self,y):
        return T.mean(T.neq(self.y_pred, y))
'''

'''
for regression:
 
    
'''