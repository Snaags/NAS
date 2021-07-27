import ops1d as ops
import random
#hyperparameter processing
from operator import itemgetter 
import torch.nn as nn

class DataShapeLogger:
  def __init__(self, filename):
    self.filename = filename
    self.log_list = [] 
  def log(self, *string):
    string = [str(x) for x in string ]
    self.log_list.append("".join(string))
  def write(self):   
    MyFile=open(self.filename+".txt",'w')
    
    for element in self.log_list:
         MyFile.write(element)
         MyFile.write('\n')
    MyFile.close()

class Model(nn.Module):
  def __init__(self, input_size, output_size, hyperparameters):
    super(Model,self).__init__()
    self.log_flag = True
    self.logger = DataShapeLogger("logger.txt")
    self.hyperparameters = hyperparameters  
    self.channels = hyperparameters["channels"]
    self.p = hyperparameters["p"]
    self.build_cells(hyperparameters)
    self.in_conv = ops.StdConv(input_size[0], self.channels)
    self.layers = hyperparameters["layers"]
    self.gap = ops.AdaAvgPool() 
    self.fc = nn.Linear(self.channels, output_size)
    self.fcact = nn.Softmax(dim = 1)
  def _build_dict(self,parameters : dict, keyword : str):
    _dictionary = dict()
    keyword_length = len(keyword)
    id_index = keyword_length + 1
    
    for parameter in parameters:
      if parameter[:keyword_length] == keyword:
        cell_id = int(parameter[id_index])

        operation_key = parameter[id_index + 2 : ]
        operation_value = parameters[ parameter ]
        
        if cell_id in _dictionary.keys():        
          _dictionary[ cell_id ][ operation_key ] = operation_value
         
        else: #if dictionary doesnt exist, make it
          _dictionary[ cell_id ] = { operation_key : operation_value }

    return _dictionary
  
  def build_cells(self, parameters): 
    conv_dictionary = self._build_dict(parameters, "normal_cell")
    redu_dictionary = self._build_dict(parameters, "reduction_cell")
    self.cellsconv = Cell(conv_dictionary[1],self.channels,self.p)
    self.cellsredu = Cell(redu_dictionary[1],self.channels,self.p)
  
  def _forward(self,x):
    x = self.in_conv(x)
    for i in range(self.layers):
      x = self.cellsconv(x) 
      if i != (self.layers -1):
        x = self.cellsredu(x)
    x = self.gap(x)
    x = self.fc(x.squeeze())
    x = self.fcact(x)
    return x  

  def _forward_log(self,x):
    self.logger.log("Input Size: ", x.size())
    x = self.in_conv(x)
    self.logger.log("After in_conv: ", x.size() )
    for i in range(self.layers):
      x = self.cellsconv(x) 
      self.logger.log("Data after Normal Cell ", str(i),": ",x.size())
      if i != (self.layers -1):
        x = self.cellsredu(x)
        self.logger.log("Data after Reduction Cell ", str(i),": ",x.size())
    self.logger.log("Size of x after cells: ", x.size())
    x = self.gap(x)
    self.logger.log("Size of x after gap: ", x.size())
    self.logger.log("Size of dense input: ", self.channels)
    x = self.fc(x.squeeze())
    x = self.fcact(x)
    self.logger.write()
    return x  

  def forward(self,x):
    if self.log_flag == True:
      self.log_flag = False
      return self._forward_log(x)
    else:
      return self._forward(x)
class Ops(nn.Module):
  def __init__(self, parameters, channels, p):
    super(Ops,self).__init__()
    self.args = {}
    self.channels = channels
    self.multicompute = False
    self.p = p
    self.input = []
    self.dropout = nn.Dropout(p = 0.2)
    for i in parameters:
      if i == "type":
        self.op = parameters[i]
        self.operation = self.get_operation(parameters[i])
      elif i[:-2] == "input":
          if parameters[i] not in self.input:
            self.input.append(parameters[i])
      else:
          self.args[i] = parameters[i]
    self.compute = nn.ModuleList()
    if len(self.input) > 1:
      self.compute.append(ops.StdAdd())
      self.multicompute = True
      self.pool = nn.AvgPool1d(2)
    self.compute.append(self.operation(**self.args))
  def get_required(self) -> list:
    return self.input

  def get_operation(self, op_key):
    if op_key == "StdConv":
      operation = ops.StdConv
      self.args["C_in"] = self.args["C_out"] = self.channels
      self.args["padding"] = "same" 
    elif op_key == "Conv3":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 3
      self.args["stride"] = 1
      self.args["padding"] = "same"
      self.args["separable"] = False
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "Conv5":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 5
      self.args["stride"] = 1
      self.args["padding"] = "same" 
      self.args["separable"] = False
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "MaxPool":    
      operation = ops.Pool
      self.args["pool_type"] = "max"
      self.args["kernel_size"] = 5
    elif op_key == "AvgPool":    
      operation = ops.Pool
      self.args["pool_type"] = "avg"
      self.args["kernel_size"] = 5
    elif op_key == "FactorizedReduce":
      operation = ops.FactorizedReduce
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "":
      operation = ops.StdConv
    return operation
  def forward( self, x ):
    #print(self.op)
    for count,i in enumerate(self.compute):
      if self.multicompute and count == 0:
        #print("Size Before operation: ", x[0].size(),x[1].size())
        x = i(*x) 
        #print("Size After operation: ", x.size())
      else:
        x = self.dropout(x)
        #print("Size Before operation: ", x.size())
        x = i(x)
        #print("Size After operation: ", x.size())
    if self.multicompute:
      x = self.pool(x)
    return x 
  
  def process( self, x : list):
    return self.forward(itemgetter( *self.get_required() )( x ))  

class Cell(nn.Module):
  """
  Contains a series of operations and information links
  """
  def __init__(self,parameters,channels,p):
    super(Cell, self).__init__() 
    self.ops_id = [] #numerical identifier for each operation 
    self.ops = []
    self.p = p
    self.channels = channels
    self.inputs = []
    self.output_operation = parameters["num_ops"]
    self.build_ops(parameters)
    self.compute_order = nn.ModuleList()
    self.compute_order.extend(self.calculate_compute_order())
  def _build_dict(self,parameters : dict, keyword : str):
    _dictionary = dict()
    keyword_length = len(keyword)
    id_index = keyword_length + 2
    
    for parameter in parameters:
      if parameter[:keyword_length] == keyword:
        cell_id = int(parameter[id_index])

        operation_key = parameter[id_index + 2 : ]
        operation_value = parameters[ parameter ]
        
        if cell_id in _dictionary.keys():        
          _dictionary[ cell_id ][ operation_key ] = operation_value
         
        else: #if dictionary doesnt exist, make it
          _dictionary[ cell_id ] = { operation_key : operation_value }

    return _dictionary
  def build_ops(self, parameters):
    ops_dictionary = self._build_dict(parameters, "op")  
    for i in ops_dictionary:
      
      self.ops.append(Ops(ops_dictionary[i], self.channels, self.p))
      self.ops_id.append(i)
  def calculate_compute_order(self):
    #
    compute_order = []
    #zero is the cell inputs, the first operation is op 1 
    current_ops_done = [0]
    
    while len(compute_order) < len(self.ops):
      for i in self.ops:
          if set(i.get_required()).issubset(current_ops_done):
            compute_order.append(i)
            current_ops_done.append(self.ops_id[self.ops.index(i)])
        
    return compute_order
      
    
    
  def forward(self, x):
    outputs = [x]
    for op in self.compute_order:
      outputs.append(op.process(outputs))  

    return outputs[self.output_operation] 

 
