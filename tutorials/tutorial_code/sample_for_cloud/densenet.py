# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Densenet."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from collections import OrderedDict
from mindspore.common.initializer import TruncatedNormal

def weight_variable():
    """Weight variable."""
    return TruncatedNormal(0.02)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def _conv(in_channel, out_channel, kernel_size, stride, padding = 0):
    weight_shape = (out_channel, in_channel, kernel_size, kernel_size)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, stride=stride, pad_mode='pad', weight_init=weight_variable(), padding=padding)

def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

class _DenseLayer(nn.Cell):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()

        print('    ', num_input_features)
        self.norm1 = _bn(num_input_features)
        self.relu1 = nn.ReLU()
        self.conv1 = _conv(num_input_features, bn_size * growth_rate, kernel_size = 1, stride = 1)
        self.norm2 = _bn(bn_size * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = _conv(bn_size * growth_rate, growth_rate, kernel_size = 3, stride = 1, padding = 1)
        
        self.drop_rate = drop_rate

    def construct(self, x):
        #out = self.norm1(x)
        out = self.relu1(x)
        out = self.conv1(out)
        #out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return out



class _DenseBlock(nn.Cell):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()

        self.num_layers = num_layers
        print('dense block num_layers:', num_layers)
        block_layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features = num_input_features + i * growth_rate,
                growth_rate = growth_rate,
                bn_size = bn_size,
                drop_rate = drop_rate
            )
            block_layers.append(layer)
        
        self.block_layers = block_layers

        self.layer0 = block_layers[0]
        self.layer1 = block_layers[1]
        self.layer2 = block_layers[2]
        self.layer3 = block_layers[3]
        self.layer4 = block_layers[4]
        self.layer5 = block_layers[5]

        self.concat = P.Concat(axis=1)

    def construct(self, x):
        midOut = []

        for i in range(6, self.num_layers):
            input = x
            for out in midOut:
                input = self.concat((input, out))
            output = self.block_layers[i](input)
            midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer0(input)
        midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer1(input)
        midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer2(input)
        midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer3(input)
        midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer4(input)
        midOut.append(output)

        input = x
        for out in midOut:
            input = self.concat((input, out))
        output = self.layer5(input)
        midOut.append(output)

        output = x
        for out in midOut:
            output = self.concat((output, out))
        return output



class _Transition(nn.Cell):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()

        self.norm = _bn(num_input_features)
        self.relu = nn.ReLU()
        self.conv = _conv(num_input_features, num_output_features, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        #out = self.norm(x)
        out = self.relu(x)
        out = self.conv(out)
        out = self.pool(out)

        return out



class Densenet(nn.Cell):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):
        super(Densenet, self).__init__()

        self.feature = nn.SequentialCell(OrderedDict([
            ('conv0', _conv(3, num_init_features, kernel_size=7, stride=2, padding=3)),
            #('norm0', _bn(num_init_features)),
            ('relu0', nn.ReLU())
        ]))

        self.pad = P.Pad(((0,0),(0,0),(1,1), (1,1)))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        num_features = num_init_features
        blocks = []
        for i, num_layers in enumerate(block_config):
            print('num_features', num_features)
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                print('transition num_input_features:', num_features, '  num_output_features:', num_features // 2)
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                blocks.append(trans)
                num_features = num_features // 2

        self.block0 = blocks[0]
        self.block1 = blocks[1]
        self.block2 = blocks[2]
        self.block3 = blocks[3]
        self.block4 = blocks[4]
        '''self.block5 = blocks[5]
        self.block6 = blocks[6]'''
        self.norm5 = _bn(num_features)
        self.classifier = nn.Dense(num_features, num_classes)
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = P.Flatten()
        self.print = P.Print()

    def construct(self, x):
        out = self.feature(x)
        out = self.pad(out)
        out = self.pool(out)
        out = self.block0(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        '''out = self.block5(out)
        out = self.block6(out)'''
        #out = self.norm5(out)
        out = self.relu(out)
        out = self.mean(out, (2, 3))
        out = self.flatten(out)
        out = self.classifier(out)

        return out


def densenet121(class_num=10):
    return Densenet(32, (6, 6, 6), 64, num_classes=class_num)
