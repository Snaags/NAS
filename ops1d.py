import math
import torch
import torch.nn as nn
def min_which(a : int, b : int):
  if a < b:
    return a, 0
  if a == b: 
    return a , None
  else: 
    return b, 1

class StdConv(nn.Module):
    def __init__(self, C_in, C_out,padding = 0):
        super(StdConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(C_in, C_out, 1, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class StdAdd(nn.Module):
  def __init__(self):
    super(StdAdd,self).__init__()
    self.init_flag = False
    self.operation = nn.Linear
  def forward(self, x1,x2):
    #Adds two 2d tensors together ((35,29), (30,35))
    return torch.cat((x1,x2), dim = 2)


class AdaAvgPool(nn.Module):
    def __init__(self):
        super(AdaAvgPool, self).__init__()
        self.layer =  nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.layer(x)



class StdDense(nn.Module):
  def __init__(self, in_size, out_size):
    super(StdDense, self).__init__()
    self.dense = nn.Linear(in_size, out_size)

  def forward(self, x):
    return self.dense(x)
    
class PoolBranch(nn.Module):
    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        self.pool = Pool(pool_type, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        out = self.preproc(x)
        out = self.pool(out)
        out = self.bn(out)
        return out


class SeparableConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(C_in, C_in, kernel_size=kernel_size, padding=padding, stride=stride,
                                   groups=C_in, bias=False)
        self.pointwise = nn.Conv1d(C_in, C_out, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, separable):
        super(ConvBranch, self).__init__()
        print(C_in)
        self.preproc = StdConv(C_in, C_out)
        if separable:
            self.conv = SeparableConv(C_out, C_out, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv1d(C_out, C_out, kernel_size, stride=stride, padding=padding)
        self.postproc = nn.Sequential(
            nn.BatchNorm1d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.preproc(x)
        out = self.conv(out)
        out = self.postproc(out)
        return out


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv1d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv1d(C_in, math.ceil(C_out / 2), 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:,:,:])], dim=1)
        out = self.bn(out)
        return out


class Pool(nn.Module):
    def __init__(self, pool_type, kernel_size, stride = None, padding = 0):
        super().__init__()
        if stride == None:
          stride = 1
           
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding = 2, ceil_mode = True)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding = 2, count_include_pad=False,ceil_mode= True)
        else:
            raise ValueError()

    def forward(self, x):
        return self.pool(x)


class SepConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = SeparableConv(C_in, C_out, kernel_size, 1, padding)
        self.bn = nn.BatchNorm1d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DropPath(nn.Module):
    def __init__(self, p=0.):
        """
        Drop path with probability.

        Parameters
        ----------
        p : float
            Probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.:
            keep_prob = 1. - self.p
            # per data point mask
            mask = torch.zeros((x.size(0), 1, 1), device=x.device).bernoulli_(keep_prob)
            return x / keep_prob * mask

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool with BN. `pool_type` must be `max` or `avg`.
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm1d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out
