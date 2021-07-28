

class Supernet:

  def __init__(self, config_space):
    self.config_space = config_space
    self.hyperparameter_list = config_space.get_hyperparameters()
    self.normal_cell = {} 
    self.reduction_cell = {} 
  def parse(self):
    for i in self.hyperparameter_list:
     
      
      
