import numpy as np 
import time
import os 
from torchviz import make_dot
import matplotlib
#matplotlib.use("Agg")
import hiddenlayer as hl
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from window import window_array_random
from model_constructor import Model
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random
"""
## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).
The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).

"""

def getConfusionMatrix(model,testLoader,n_classes = 2 ,show_image=False):
  model.eval() #set the model to evaluation mode
  confusion_matrix=torch.zeros((n_classes , n_classes),dtype=int) #initialize a confusion matrix
  
  TP=FP=FN=TN = 0
  correct = 0
  incorrect = 0
  with torch.no_grad(): #disable back prop to test the model
      for i, (inputs, labels) in enumerate(testLoader):

          inputs = inputs.cuda(non_blocking=True)
          labels = labels.cuda(non_blocking=True).long()
          outputs = model(inputs.float())
          preds = torch.argmax(outputs, 1).long()
          #get confusion matrix
          if i == 50:
            break
          for j in range(inputs.size()[0]):
              
              confusion_matrix[preds[j]][labels[j]] += 1 
              if preds[j] == labels[j] != 0:
                  TP += 1
                  correct += 1 
              elif preds[j] != labels[j] and labels[j] == 0:
                  FP += 1
                  incorrect += 1 
              elif preds[j] != labels[j] and preds[j] == 0:
                  FN += 1
                  incorrect += 1 
              elif preds[j] == labels[j] == 0:
                  TN += 1
                  correct += 1
              elif preds[j] != labels[j]:
                  incorrect+=1
              else:
                  print("Unsorted Value!!")  
      #print results
      total_evaluations = incorrect+correct
      accuracy = correct/(total_evaluations)
      PRINT_DATA = False
      print("Accuracy: " , str(100*accuracy),"%")
      if PRINT_DATA == True:
        print("Total Evaluation Samples: ", total_evaluations)
        print("True Positives: " , str(TP))
        print("False Positives: " , str(FP))
        print("True Negatives: " , str(TN))
        print("False Negatives: " , str(FN))
        print("Correct: " , str(correct))
        print("Incorrect: " , str(incorrect))
        print(" ") 
        total = 0
        for i in range(n_classes):
          class_num = confusion_matrix[:,i]
          print("Total samples for class ",str(i),": ",str(torch.sum(class_num).item()))
          print("Correct samples for class ",str(i),": ",str(confusion_matrix[i,i].item()))
          total+= torch.sum(class_num).item()
        print("Total Samples: " ,str(total))
      return accuracy, confusion_matrix, TP, FP, FN , TN


class TEPS(Dataset):
  def __init__(self, window_size, x : str , y : str ):
    path = "datasets/TEPS/"
    self.x = torch.from_numpy(np.reshape(np.load(path+x),(-1,52)))
    self.y = torch.from_numpy(np.load(path+y))
    self.n_samples = self.x.shape[0]
    self.window = window_size
    self.n_classes = len(np.unique(self.y))

  def __getitem__(self, index):
    while index+self.window > self.n_samples:
      index = random.randint(0,self.n_samples)
    x = self.x[index:index+self.window]
    y = self.y[index+self.window-1]
    x = x.reshape(52,self.window)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples - self.n_samples%self.window


class Train_TEPS(TEPS):

  def __init__(self, window_size): 
    super().__init__(window_size, "x_train.npy","y_train.npy")

class Test_TEPS(TEPS):
  def __init__(self, window_size): 
    super().__init__(window_size, "x_test.npy","y_test.npy")

def alloc_gpu(): 
    gpu_mem_free = []
    for i in range(torch.cuda.device_count()):
      r = torch.cuda.memory_reserved(i)
      t = torch.cuda.get_device_properties(i).total_memory
      gpu_mem_free.append( t-r )
    torch.cuda.set_device(gpu_mem_free.index(max(gpu_mem_free)))

def main(hyperparameter,train_dataloader, test_dataloader,budget = 2000):
    alloc_gpu() 
    VISUAL_MODE = False
    def cal_acc(y,t):
        return np.count_nonzero(y==t)/len(y)
    def convert_label_max_only(y):
        y = y.cpu().detach().numpy()
        idx = np.argmax(y,axis = 1)
        out = np.zeros_like(y)
        for count, i in enumerate(idx):
            out[count, i] = 1
        return idx
    batch_size = 256
    num_workers = 1        

    train_dataset = Train_TEPS(hyperparameter["window_size"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        shuffle = True,drop_last=True,pin_memory=True)

    test_dataset = Test_TEPS(hyperparameter["window_size"])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle = True, 
                                     drop_last=True,pin_memory=True 
                                     )

    num_classes = train_dataset.get_n_classes()
    #print("Currently Running Hyperparameter Set: ", hyperparameter)
    model = Model(input_size = train_dataset.x.shape[1:] ,output_size = num_classes,hyperparameters = hyperparameter)
    model = model.cuda()
    """
    ## Train the model
    """
    ###Training Configuration
    max_iter = budget
    n_iter = train_dataset.get_n_samples()/batch_size
    if max_iter < n_iter:
      n_iter = max_iter
    epochs = int(1)
    optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
    criterion = nn.CrossEntropyLoss()
    acc = 0
    OUTPUT_DIR = "TEPS"
    h = hl.History()
    c = hl.Canvas() 
    #m = hl.build_graph(model, torch.zeros([batch_size,52 ,hyperparameter["window_size"]]).cuda())
    for epoch in range(epochs):
      for i, (samples, labels) in enumerate(train_dataloader):
        samples =samples.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(samples.float())
        
        if i == 2: 
         picture = make_dot(outputs.mean(), params=dict(model.named_parameters())    )
         picture.format = "png"
         picture.render("Model")

        # forward + backward + optimize
   
        outputs = outputs
        labels = labels.long()
        #labels = labels.unsqueeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i%10 == 0:
          ##hl logging
          acc += cal_acc(convert_label_max_only(outputs), labels.cpu().detach().numpy())
          acc = acc/2
          h.log(i, loss = loss, accuracy = acc)
            # Save the canvas
          print("Epoch (",str(epoch),"/",str(epochs), ")""Iteration(s) (", str(i),"/",str(n_iter), ") Loss: ","%.2f" % loss.item(), "Accuracy: ","%.2f" % acc , end = '\r')
        if i >= max_iter:
          break
    if VISUAL_MODE == True: 
      c.draw_plot([h["loss"], h["accuracy"]])
      c.save(os.path.join(OUTPUT_DIR,"%.2f" % acc +  "training_progress.png"))
            # You can also save the history to a file to load and inspect layer
    h.save(os.path.join(OUTPUT_DIR, "%.2f" % acc + "training_progress.pkl"))

    """
    ## Evaluate model on test data
    """
    print()
    acc , CM , TP, FP , FN , TN = getConfusionMatrix( model,test_dataloader ,  num_classes  ) 
   
    import seaborn as sns; sns.set_theme()
    fig = plt.figure(figsize = (25,15))
    fig = sns.heatmap(CM,annot=True,fmt='d',cmap="Blues", cbar_kws={'label': 'Samples'})
    fig.set(xlabel='Actual Class', ylabel='Predicted Class')
    fig.set_title("Confusion Matrix", fontdict ={"fontsize": 25, "fontweight": "bold"}, y = 1.05)
    fig.figure.savefig(OUTPUT_DIR+"/TEPS_"+"%.2f" % acc + ".png", format = "png", dpi = 500)


 

    

    return acc 

def wrapper(hyperparameters):
  try:
    return main(hyperparameters)
  except:
    return 0

if __name__ == "__main__":
  hyperparameter = {'channels': 52, 'layers': 5, 'lr': 0.0007560788690317892, 'normal_cell_1_num_ops': 1, 'normal_cell_1_ops_1_input_1': 0, 'normal_cell_1_ops_1_input_2': 0, 'normal_cell_1_ops_1_type': 'MaxPool7', 'normal_cell_1_ops_2_input_1': 0, 'normal_cell_1_ops_2_input_2': 1, 'normal_cell_1_ops_2_type': 'MaxPool7', 'normal_cell_1_ops_3_input_1': 0, 'normal_cell_1_ops_3_input_2': 1, 'normal_cell_1_ops_3_type': 'MaxPool5', 'normal_cell_1_ops_4_input_1': 1, 'normal_cell_1_ops_4_input_2': 3, 'normal_cell_1_ops_4_type': 'SepConv7', 'normal_cell_1_ops_5_input_1': 3, 'normal_cell_1_ops_5_input_2': 3, 'normal_cell_1_ops_5_type': 'Conv3', 'normal_cell_1_ops_6_input_1': 2, 'normal_cell_1_ops_6_input_2': 0, 'normal_cell_1_ops_6_type': 'MaxPool5', 'num_conv': 1, 'num_re': 1, 'p': 0.05, 'reduction_cell_1_num_ops': 1, 'reduction_cell_1_ops_1_input_1': 0, 'reduction_cell_1_ops_1_input_2': 0, 'reduction_cell_1_ops_1_type': 'FactorizedReduce', 'window_size': 100}

  main(hyperparameter )

