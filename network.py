import torch
import torch.nn as nn
import torch.nn.functional as F
from ScaleLayer import ScaleLayer

defaultcfg = [32, 64, 64, 128]

def conv_bn_elu_mp(in_planes, out_planes, pooling_size):
    return [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ELU(alpha=1.0),
        nn.MaxPool2d(kernel_size=pooling_size)]

def conv_bn_scale_elu_mp(in_planes, out_planes, pooling_size):
    return [nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        ScaleLayer(out_planes, 0.5),
        nn.ELU(alpha=1.0),
        nn.MaxPool2d(kernel_size=pooling_size)]

class CompactCNNPruneScaling(nn.Module):
    def __init__(self, cfg=None, filter_percent=None):
        super(CompactCNNPruneScaling, self).__init__()
        if cfg is None:
            cfg = defaultcfg
        self.feature = self.make_layers(cfg)
        self.predict = nn.Sequential(nn.Dropout(p=0.2),
                                    nn.Linear(6*cfg[-2], cfg[-1]),
                                    ScaleLayer(cfg[-1], 0.5, dim=2),
                                    nn.Linear(cfg[-1], 8))
        self.filter_percent = filter_percent

    def make_layers(self, cfg):
        layers = []
        in_planes = 1
        pooling_sizes = [(2, 4), (3, 4), (2, 5)]
        layers += [nn.BatchNorm2d(in_planes)]

        for idx, v in enumerate(cfg[:-1]):
            layers += conv_bn_scale_elu_mp(in_planes, v, pooling_sizes[idx])
            in_planes = v

        return nn.Sequential(*layers)

    def forward(self, x, pruning_rate=0, is_training=True):
        if is_training is True:
            cc = torch.zeros([1,1], dtype=torch.float32).cuda()
            layer_cnt = 0
            total = 0
            index = 0
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    w_size = m.weight.data.size()
                    w = m.weight.data
                    m_padding = 0
                    m_stride = 1

                    if w.size()[0]>1:
                        layer_cnt += 1
                        if isinstance(m, nn.Conv2d):
                            m_padding = m.padding
                            m_stride = m.stride
                            m.padding = 0
                            m.stride = 1
                            w = w.view(w.size()[0], -1)

                        w_n = torch.norm(w, p=2, dim=1).view(-1,1).repeat(1, w.size()[1])
                        w = w / (w_n+1e-9)
                        w = w.view(w_size)
                        m.weight.data = w
                        if m.bias is not None:
                            bias_m = m.bias.data.clone()
                            m.bias.data = torch.zeros(bias_m.size()).cuda()
                        w_c = m(w)
                        cc += ( torch.sum(torch.abs(w_c)) - torch.cuda.FloatTensor([[w.size()[0]]] ) ) / (w.size()[0]*(w.size()[0]-1) )
                        
                        m.weight.data = w * (w_n+1e-9).view(w_size)
                        if m.bias is not None:
                            m.bias.data = bias_m
                        if isinstance(m, nn.Conv2d):
                            m.padding = m_padding
                            m.stride = m_stride

                elif isinstance(m, ScaleLayer):
                    total += m.scale.data.shape[0]
            
            div_loss = cc/float(layer_cnt)

            pr_loss = torch.zeros([1,1], dtype=torch.float32).cuda()
            if self.filter_percent>0 and pruning_rate>0:
                bn = torch.zeros(total)
                index = 0
                for m in self.modules():
                    if isinstance(m, ScaleLayer):
                        size = m.scale.data.shape[0]
                        bn[index:(index+size)] = torch.abs(m.scale.data)
                        index += size

                z, j = torch.sort(bn) # ascending order
                thre_index = int(total * self.filter_percent)
                thre = z[int(total * pruning_rate)]
                #thre = z[thre_index]

                if thre < z[-1]:
                    tol_bn = bn.sum()
                    pr_bn = z[:thre_index].sum()
                    pr_loss += float(pr_bn) / float(tol_bn)

                    #dropout_ws = []
                    #dropout_bs = []
                    for k, m in enumerate(self.modules()):
                        if isinstance(m, ScaleLayer):
                            weight_copy = m.scale.data.abs().clone()
                            retained_mask = weight_copy.gt(thre.cuda()).float().cuda()
                            #dropout_ws.append(torch.mul(1-retained_mask, m.scale.data))
                            m.scale.data.mul_(retained_mask)
        else:
            div_loss = torch.zeros([1,1], dtype=torch.float32).cuda()
            pr_loss = torch.zeros([1,1], dtype=torch.float32).cuda()

        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predict(x)

        return x, div_loss, pr_loss

class CompactCNN(nn.Module):
    def __init__(self, cfg=None):
        super(CompactCNN, self).__init__()
        if cfg is None:
            cfg = defaultcfg
        self.feature = self.make_layers(cfg)
        #self.filter_percent = args.filter_percent
        self.predict = nn.Sequential(nn.Dropout(p=0.2),
                                    nn.Linear(6*cfg[-2], cfg[-1]),
                                    nn.Linear(cfg[-1], 8))

    def make_layers(self, cfg):
        layers = []
        in_planes = 1
        pooling_sizes = [(2, 4), (3, 4), (2, 5)]
        layers += [nn.BatchNorm2d(in_planes)]

        for idx, v in enumerate(cfg[:-1]):
            layers += conv_bn_elu_mp(in_planes, v, pooling_sizes[idx])
            in_planes = v

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predict(x)

        return x

class CompactCNNPrune(nn.Module):
    def __init__(self, cfg=None, filter_percent=None):
        super(CompactCNNPrune, self).__init__()
        if cfg is None:
            cfg = defaultcfg
        self.feature = self.make_layers(cfg)
        self.predict = nn.Sequential(nn.Dropout(p=0.2),
                                    nn.Linear(6*cfg[-2], cfg[-1]),
                                    nn.Linear(cfg[-1], 8))
        self.filter_percent = filter_percent

    def make_layers(self, cfg):
        layers = []
        in_planes = 1
        pooling_sizes = [(2, 4), (3, 4), (2, 5)]
        layers += [nn.BatchNorm2d(in_planes)]

        for idx, v in enumerate(cfg[:-1]):
            layers += conv_bn_elu_mp(in_planes, v, pooling_sizes[idx])
            in_planes = v

        return nn.Sequential(*layers)

    def forward(self, x, pruning_rate=0, is_training=True):
        if is_training is True:
            cc = torch.zeros([1,1], dtype=torch.float32).cuda()
            layer_cnt = 0
            total = 0
            index = 0
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    w_size = m.weight.data.size()
                    w = m.weight.data
                    m_padding = 0
                    m_stride = 1

                    if w.size()[0]>1:
                        layer_cnt += 1
                        if isinstance(m, nn.Conv2d):
                            m_padding = m.padding
                            m_stride = m.stride
                            m.padding = 0
                            m.stride = 1
                            w = w.view(w.size()[0], -1)

                        w_n = torch.norm(w, p=2, dim=1).view(-1,1).repeat(1, w.size()[1])
                        w = w / (w_n+1e-9)
                        w = w.view(w_size)
                        m.weight.data = w
                        if m.bias is not None:
                            bias_m = m.bias.data.clone()
                            m.bias.data = torch.zeros(bias_m.size()).cuda()
                        w_c = m(w)
                        cc += ( torch.sum(torch.abs(w_c)) - torch.cuda.FloatTensor([[w.size()[0]]] ) ) / (w.size()[0]*(w.size()[0]-1) )
                        
                        m.weight.data = w * (w_n+1e-9).view(w_size)
                        if m.bias is not None:
                            m.bias.data = bias_m
                        if isinstance(m, nn.Conv2d):
                            m.padding = m_padding
                            m.stride = m_stride

                elif isinstance(m, nn.BatchNorm2d):
                    total += m.weight.data.shape[0]
            
            div_loss = cc/float(layer_cnt)

            pr_loss = torch.zeros([1,1], dtype=torch.float32).cuda()
            if self.filter_percent>0 and pruning_rate>0:
                bn = torch.zeros(total)
                index = 0
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        size = m.weight.data.shape[0]
                        bn[index:(index+size)] = torch.abs(m.weight.data)#.abs().clone()
                        index += size

                z, j = torch.sort(bn) # ascending order
                thre_index = int(total * self.filter_percent)
                thre = z[int(total * pruning_rate)]
                #thre = z[thre_index]

                if thre < z[-1]:
                    tol_bn = bn.sum()
                    pr_bn = z[:thre_index].sum()
                    pr_loss += float(pr_bn) / float(tol_bn)

                    dropout_ws = []
                    dropout_bs = []
                    for k, m in enumerate(self.modules()):
                        if isinstance(m, nn.BatchNorm2d):
                            weight_copy = m.weight.data.abs().clone()
                            retained_mask = weight_copy.gt(thre.cuda()).float().cuda()
                            dropout_ws.append(torch.mul(1-retained_mask, m.weight.data))
                            dropout_bs.append(torch.mul(1-retained_mask, m.bias.data))
                            m.weight.data.mul_(retained_mask)
                            m.bias.data.mul_(retained_mask)
        else:
            div_loss = torch.zeros([1,1], dtype=torch.float32).cuda()
            pr_loss = torch.zeros([1,1], dtype=torch.float32).cuda()

        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.predict(x)

        return x, div_loss, pr_loss

    
