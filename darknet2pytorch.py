import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='darknet to PyTorch')
parser.add_argument('--cfg', default='', type=str, metavar='PATH',
                    help='path to darknet cfg (default: none)')
parser.add_argument('--weight', default='', type=str, metavar='PATH',
                    help='path to darknet weight (default: none)')
args = parser.parse_args()
class FCView(nn.Module):
    def __init__(self):
        super(FCView, self).__init__()

    def forward(self, x):
        nB = x.data.size(0)
        x = x.view(nB,-1)
        return x
    def __repr__(self):
        return 'view(nB, -1)'

class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

def parse_cfg(cfgfile):
    def erase_comment(line):
        line = line.split('#')[0]
        return line
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            line = erase_comment(line)
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def create_network(blocks):
    layers = []
    prev_filters = 3
    out_filters =[]
    out_width =[]
    out_height =[]
    conv_id = 0
    prev_width = 0
    prev_height = 0
    for block in blocks:
        if block['type'] == 'net':
            prev_filters = int(block['channels'])
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue

        elif block['type'] == 'convolutional':
            conv_id = conv_id + 1
            batch_normalize = int(block['batch_normalize'])
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0
            activation = block['activation']
            if batch_normalize:
                layers += [nn.Conv2d(prev_filters, filters, kernel_size,stride, pad, bias=False)]
                layers += [nn.BatchNorm2d(filters)]
            else:
                layers += [nn.Conv2d(prev_filters, filters, kernel_size,stride, pad, bias=True)]
            if activation == 'leaky':
                layers += [nn.LeakyReLU(0.1)]
            elif activation == 'relu':
                layers += [nn.ReLU(inplace=True)]
            prev_filters = filters
            prev_width = (prev_width + 2*pad - kernel_size)/stride + 1
            prev_height = (prev_height + 2*pad - kernel_size)/stride + 1
            out_filters.append(prev_filters)
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            padding = 0
            if block.has_key('pad') and int(block['pad']) == 1:
                padding = int((pool_size-1)/2)
            if stride > 1:
                layers += [nn.MaxPool2d(pool_size, stride, padding=padding)]
            else:
                layers += [MaxPoolStride1()]
            out_filters.append(prev_filters)
            if stride > 1:
                prev_width = (prev_width - kernel_size + 1)/stride + 1
                prev_height = (prev_height - kernel_size + 1)/stride + 1
            else:
                prev_width = prev_width - kernel_size + 1
                prev_height = prev_height - kernel_size + 1
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'avgpool':
            layers += [GlobalAvgPool2d()]
            out_filters.append(prev_filters)
            prev_width = 1
            prev_height = 1
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'softmax':
            layers += [nn.Softmax()]
            prev_width = 1
            prev_height = 1
            out_filters.append(prev_filters)
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'cost':
            if block['_type'] == 'sse':
                layers += [nn.MSELoss(size_average=True)]
            elif block['_type'] == 'L1':
                layers += [nn.L1Loss(size_average=True)]
            elif block['_type'] == 'smooth':
                layers += [nn.SmoothL1Loss(size_average=True)]
            prev_width = 1
            prev_height = 1
            out_filters.append(1)
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            prev_filters = stride * stride * prev_filters
            prev_width = prev_width / stride
            prev_height = prev_height / stride
            out_filters.append(prev_filters)
            out_width.append(prev_width)
            out_height.append(prev_height)
            layers += [Reorg(stride)]

        elif block['type'] == 'dropout':
            ind = len(models)
            ratio = float(block['probability'])
            prev_filters = out_filters[ind-1]
            prev_width = out_width[ind-1]
            prev_height = out_height[ind-1]
            out_filters.append(prev_filters)
            out_width.append(prev_width)
            out_height.append(prev_height)
            layers +=  [nn.Dropout2d(ratio)]

        elif block['type'] == 'connected':
            prev_filters = prev_filters * prev_width * prev_height
            filters = int(block['output'])
            is_first = (prev_width * prev_height != 1)
            if block['activation'] == 'linear':
                if is_first:
                    layers +=  [nn.Sequential(FCView(), nn.Linear(prev_filters, filters))]
                else:
                    layers +=  [nn.Linear(prev_filters, filters)]

            elif block['activation'] == 'leaky':
                if is_first:
                    layers += [nn.Sequential(FCView(),nn.Linear(prev_filters, filters),nn.LeakyReLU(0.1, inplace=True))]
                else:
                    layers +=  [nn.Sequential(nn.Linear(prev_filters, filters),nn.LeakyReLU(0.1, inplace=True))]

            elif block['activation'] == 'relu':
                if is_first:
                    layers +=  [nn.Sequential(FCView(),nn.Linear(prev_filters, filters),nn.ReLU(inplace=True))]
                else:
                    layers +=  [nn.Sequential(nn.Linear(prev_filters, filters),nn.ReLU(inplace=True))]
            prev_filters = filters
            prev_width = 1
            prev_height = 1
            out_filters.append(prev_filters)
            out_width.append(prev_width)
            out_height.append(prev_height)

        elif block['type'] == 'region':
            continue

    else:
        print('unknown type %s' % (block['type']))

    return nn.Sequential(*layers)

def load_conv(buf, start, layer):
    num_b = layer.bias.data.cpu().numpy().size
    layer.bias.data=torch.from_numpy(buf[start:start+num_b]).cuda();   start = start + num_b
    num_w = layer.weight.data.cpu().numpy().size
    shape_w = layer.weight.data.cpu().numpy().shape
    weight = buf[start:start+num_w].reshape(shape_w)
    layer.weight.data = torch.from_numpy(weight).cuda();   start = start + num_w 
    return start

def load_bn(buf, start, layer):
    num_b = layer.bias.data.cpu().numpy().size
    layer.bias.data = torch.from_numpy(buf[start:start+num_b]).cuda();        start = start + num_b
    layer.weight.data = torch.from_numpy(buf[start:start+num_b]).cuda();      start = start + num_b
    layer.running_mean.data = torch.from_numpy(buf[start:start+num_b]).cuda();start = start + num_b
    layer.running_var.data = torch.from_numpy(buf[start:start+num_b]).cuda(); start = start + num_b
    return start

def load_conv_bn(buf, start, conv_layer, bn_layer):

    num_b = bn_layer.bias.data.cpu().numpy().size
    bn_layer.bias.data = torch.from_numpy(buf[start:start+num_b]).cuda();        start = start + num_b
    bn_layer.weight.data = torch.from_numpy(buf[start:start+num_b]).cuda();      start = start + num_b
    bn_layer.running_mean.data = torch.from_numpy(buf[start:start+num_b]).cuda();start = start + num_b
    bn_layer.running_var.data = torch.from_numpy(buf[start:start+num_b]).cuda(); start = start + num_b
    
    num_w = conv_layer.weight.data.cpu().numpy().size
    weight = buf[start:start+num_w].reshape(conv_layer.weight.data.cpu().numpy().shape)
    conv_layer.weight.data = torch.from_numpy(weight).cuda();   start = start + num_w 
    return start

def load_fc(buf, start, layer):
    num_b = layer.bias.data.cpu().numpy().size
    layer.bias.data = torch.from_numpy(buf[start:start+num_b]).cuda();     start = start + num_b
    num_w = layer.weight.data.cpu().numpy().size
    shape_w = layer.weight.data.cpu().numpy().shape
    weight = buf[start:start+num_w].reshape(shape_w)
    layer.weight.data = torch.from_numpy(weight).cuda();   start = start + num_w 
    return start

def load_weight(model, weightfile):
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
    header = torch.from_numpy(header)
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()
    
    start = 0
    for layer_index, layer in model._modules.items():

        if isinstance(layer, nn.Conv2d):
            _,layer_next = model._modules.items()[int(layer_index)+1]
            if isinstance(layer_next, nn.BatchNorm2d):
                start = load_conv_bn(buf,start,layer,layer_next)
                print 'load ' + str(layer) + ' over'
                print 'load ' + str(layer_next) + ' over'
                
            else:
                start = load_conv(buf, start, layer)
                print 'load ' + str(layer) + ' over'

blocks = parse_cfg(args.cfg)
model = create_network(blocks)
load_weight(model,args.weight)

torch.save(model, '%s.pth'%args.cfg[:args.cfg.rfind('.')])
print 'save model %s.pth '%args.cfg[:args.cfg.rfind('.')]
