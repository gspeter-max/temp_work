import numpy as np

class Normalization: 
    def __init__(self, matrix): 
        self.matrix = matrix

    def batch_normalization(self): 
        mean = np.mean(self.matrix, axis=0, keepdims=True) 
        std = np.std(self.matrix, axis=0, keepdims=True)    
        return (self.matrix - mean) / (std + 1e-5)  

    def layer_normalization(self): 
        mean = np.mean(self.matrix, axis=1, keepdims=True)  
        std = np.std(self.matrix, axis=1, keepdims=True)    
        return (self.matrix - mean) / (std + 1e-5)


matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

nor = Normalization(matrix)

result_ln = nor.layer_normalization() 
print("Layer Normalization:\n", result_ln)

# Correct Batch Normalization
result_bn = nor.batch_normalization() 
print("Batch Normalization:\n", result_bn)
