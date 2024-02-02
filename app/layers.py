#Custom L1 Dist Layer Module
#WHY DO WE NEED THIS:its needed to load the custom model

#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


#custom L1 Dist Layer from Jupyter
class L1Dist(Layer):
    #define the base init method inside of a python class - inheritance
    def __init__(self,**kwargs):
        super().__init__()
    #call function is responsible for telling what to do when some data is passed to it    
    def call(self,input_embedding, validation_embedding):
        # similarity Calculation       
        return tf.math.abs(input_embedding - validation_embedding)
        