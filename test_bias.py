import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    model_path = "output/test_barline_s2s/model.pth"
    model = torch.load(model_path).cuda()
    fc_bias = model.fc.bias
    
    print(fc_bias)
    print(F.softmax(fc_bias, dim=0))