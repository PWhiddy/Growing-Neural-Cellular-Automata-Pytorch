import torch.nn as nn
import torch.nn.functional as F
import base64
import json
import numpy as np

class CAModel(nn.Module):
    
    def __init__(self, env_d, hidden_d, device):
        super(CAModel, self).__init__()
        self.env_d = env_d
        self.conv1 = nn.Conv2d(env_d*3, hidden_d, 1).to(device)
        self.conv2 = nn.Conv2d(hidden_d, env_d, 1).to(device)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)
    
    def pack_layer(self, weight, bias, outputType):
      in_ch, out_ch = weight.shape
      assert (in_ch%4==0) 
      assert (out_ch%4==0)
      print(f'in_ch: {in_ch} out_ch: {out_ch}')
      print(f'bias.shape {bias.shape}')
      assert (bias.shape==(out_ch,))
      weight_scale, bias_scale = 1.0, 1.0
      if outputType == np.uint8:
        weight_scale = 2.0*np.abs(weight).max()
        bias_scale = 2.0*np.abs(bias).max()
        weight = np.round((weight/weight_scale+0.5)*255)
        bias = np.round((bias/bias_scale+0.5)*255)
      packed = np.vstack([weight, bias[None,...]])
      packed = packed.reshape(in_ch+1, out_ch//4, 4)
      packed = outputType(packed)
      packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')
      return {'data_b64': packed_b64, 'in_ch': in_ch, 'out_ch': out_ch,
              'weight_scale': weight_scale, 'bias_scale': bias_scale,
              'type': outputType.__name__}

    def export_pytorch_ca_to_webgl_demo(self, outputType=np.uint8):
      # reorder the first layer inputs to meet webgl demo perception layout
      w1 = self.conv1.weight.squeeze().cpu().detach().permute(1,0).numpy()
      w1 = w1.reshape(self.env_d, 3, -1).transpose(1, 0, 2).reshape(3*self.env_d, -1)
      print(f'w1 shape: {w1.shape}')
      w2 = self.conv2.weight.squeeze().cpu().detach().permute(1,0).numpy()

      layers = [
          self.pack_layer(w1, self.conv1.bias.cpu().detach().numpy(), outputType),
          self.pack_layer(w2, self.conv2.bias.cpu().detach().numpy(), outputType)
      ]
      return json.dumps(layers)