import ops1d as ops
#hyperparameter processing
from operator import itemgetter 
import torch.nn as nn


class Model(nn.Module):
  def __init__(self, input_size, output_size, hyperparameters):
    super(Model,self).__init__()
    self.hyperparameters = hyperparameters  
    self.channels = hyperparameters["channels"]
    self.cells = nn.ModuleList()
    self.build_cells(hyperparameters)
    self.in_conv = ops.StdConv(input_size[0], self.channels)
    self.gap = ops.AdaAvgPool() 
    self.fc = nn.Linear(self.channels, output_size)
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
    cell_dictionary = self._build_dict(parameters, "cell")
    for i in cell_dictionary:
      self.cells.append(Cell(cell_dictionary[i],self.channels))
  

  def forward(self,x):
    x = self.in_conv(x)
    for i in self.cells:
      x = i(x)
    #print("Size of x after cells: ", x.size())
    x = self.gap(x)
    #print("Size of x after gap: ", x.size())
    #print("Size of dense input: ", self.channels)

    x = self.fc(x.squeeze())
    return x  


class Ops(nn.Module):
  def __init__(self, parameters, channels):
    super(Ops,self).__init__()
    self.args = {}
    self.channels = channels
    self.multicompute = False
    self.input = []
    for i in parameters:
      if i == "type":
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
    self.compute.append(self.operation(**self.args))
  def get_required(self) -> list:
    return self.input

  def get_operation(self, op_key):
    if op_key == "StdConv":
      operation = ops.StdConv
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "Conv3":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 3
      self.args["stride"] = 1
      self.args["padding"] = 0
      self.args["separable"] = False
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "Conv5":
      operation = ops.ConvBranch
      self.args["kernel_size"] = 5
      self.args["stride"] = 1
      self.args["padding"] = 0 
      self.args["separable"] = False
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "MaxPool":    
      operation = ops.Pool
      self.args["pool_type"] = "max"
      self.args["kernel_size"] = 3
    elif op_key == "AvgPool":    
      operation = ops.Pool
      self.args["pool_type"] = "avg"
      self.args["kernel_size"] = 3
    elif op_key == "FactorizedReduce":
      operation = ops.FactorizedReduce
      self.args["C_in"] = self.args["C_out"] = self.channels 
    elif op_key == "":
      operation = ops.StdConv
    return operation
  def forward( self, x ):

    for count,i in enumerate(self.compute):
      if self.multicompute and count == 0:
        x = i(*x)  
      else:
        x = i(x)
    return x 
  
  def process( self, x : list):
    return self.forward(itemgetter( *self.get_required() )( x ))  

class Cell(nn.Module):
  """
  Contains a series of operations and information links
  """
  def __init__(self,parameters,channels):
    super(Cell, self).__init__() 
    self.ops_id = [] #numerical identifier for each operation 
    self.ops = []
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
      
      self.ops.append(Ops(ops_dictionary[i], self.channels))
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

 
