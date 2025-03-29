class compute_loss: 
    
    def __init__(self, lasso_lambda, ridge_lambda,model): 
        self.lasso_lambda = lasso_lambda 
        self.ridge_lambda = ridge_lambda 
        self.model = model 
    
    def call(self,y_true, y_pred): 
        
        y_true = tf.cast( y_true, tf.float32) 
        y_pred = tf.cast( y_pred , tf.float32) 

        mse = tf.reduce_mean(tf.square(y_true - y_pred)) 
        
        lasso_loss = tf.constant( 0.0 , dtype = tf.float32) 
        if self.lasso_lambda > 0: 
            lasso_penalty = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.model.trainable_weights if 'kernel' in w.name ]) 
            lasso_loss += self.lasso_lambda * lasso_penalty 

        ridge_loss = tf.constant( 0.0, dtype = tf.float32) 
        
        if self.ridge_lambda > 0: 
            ridge_penalty = tf.add_n([tf.reduce_sum(tf.square(w)) for w in self.model.trainable_weights if 'kernel' in w.name ]) 
        
            ridge_loss += self.ridge_lambda * ridge_penalty 

        return mse + ridge_loss + lasso_loss 




