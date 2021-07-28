import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 
from ConfigStruct import Parameter, Cumulative_Integer_Struct, LTP_Parameter 

"""	TODO
Seperate Pooling and Convolution Layers
Add more convolution operations (kernalSize and maybe stride)
"""

def init_config():

  cs = CS.ConfigurationSpace()

  inputs = Cumulative_Integer_Struct(cs,None,"ops","num_ops","Integer",1,10)
  conv_ops = ["StdConv", "Conv3", "Conv5","MaxPool","AvgPool"]  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        inputs]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,5)

  

  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"normal_cell", "num_conv","Integer", 1, 1).init() 

  conv_ops = ["FactorizedReduce"]
  
  ops_parameters = [
        Parameter("type",               "Categorical", lower_or_constant_value = conv_ops ), 
        LTP_Parameter("input_1",               "Integer", 0,10),
        LTP_Parameter("input_2",               "Integer", 0,10)]
  ops = Cumulative_Integer_Struct(cs,ops_parameters,"ops","num_ops","Integer",1,1)


  conv_parameters = [
        ops]
 
  Cumulative_Integer_Struct(cs,conv_parameters,"reduction_cell", "num_re","Integer", 1, 1).init() 
  layers = CSH.UniformIntegerHyperparameter(name = "layers", lower = 1 ,upper = 5)
    ###Topology Definition]###

  hp_list = [ layers]
  cs.add_hyperparameters(hp_list)
  return cs

if __name__ == "__main__":
	configS = init_config()	
	print(configS.get_hyperparameters())
