import torch.nn as nn
import torch



class DARTSUNet(nn.Module):
    def __init__(self,inch=64,classes=1,step=4):
        super(DARTSUNet, self).__init__()
        self.downlayer=nn.ModuleList()

    def init_alpha(self):

