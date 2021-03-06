# 算子

<!-- TOC -->

- [算子](#算子)
  - [概述](#概述)
  - [张量操作](#张量操作)
    - [标量运算](#标量运算)
      - [加法](#加法)
      - [Element-wise乘法](#element-wise乘法)
      - [求三角函数](#求三角函数)
    - [向量运算](#向量运算)
      - [Squeeze](#squeeze)
      - [求Sparse2Dense](#求sparse2dense)
    - [矩阵运算](#矩阵运算)
      - [矩阵乘法](#矩阵乘法)
      - [广播机制](#广播机制)
  - [网络操作](#网络操作)
    - [特征提取](#特征提取)
      - [卷积操作](#卷积操作)
      - [卷积的反向传播算子操作](#卷积的反向传播算子操作)
    - [激活函数](#激活函数)
    - [LossFunction](#lossfunction)
      - [L1Loss](#l1loss)
    - [优化算法](#优化算法)
      - [SGD](#sgd)
  - [数组操作](#数组操作)
    - [DType](#dtype)
    - [Cast](#cast)
    - [Shape](#shape)
  - [图像操作](#图像操作)
  - [编码运算](#编码运算)
    - [BoundingBoxEncode](#boundingboxencode)
    - [BoundingBoxDecode](#boundingboxdecode)
    - [IOU计算](#iou计算)
  - [调试操作](#调试操作)
    - [Debug](#debug)
    - [HookBackward](#hookbackward)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/api/source_zh_cn/programming_guide/operator.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

算子组件包含了常用的算子及其操作，按功能大致可分为张量操作、网络操作、数组操作、图像操作、编码操作、调试操作和量化操作七个模块。所有的算子在Ascend AI处理器、GPU和CPU的支持情况，参见[算子支持列表](https://www.mindspore.cn/docs/zh-CN/master/operator_list.html)。

## 张量操作

张量操作包括张量的结构操作和张量的数学运算。

张量结构操作有：张量创建、索引切片、维度变换和合并分割。

张量数学运算有：标量运算、向量运算和矩阵运算。

这里以张量的数学运算和运算的广播机制为例，介绍使用方法。

### 标量运算

张量的数学运算符可以分为标量运算符、向量运算符以及矩阵运算符。

加减乘除乘方，以及三角函数、指数、对数等常见函数，逻辑比较运算符等都是标量运算符。

标量运算符的特点是对张量实施逐元素运算。

有些标量运算符对常用的数学运算符进行了重载。并且支持类似NumPy的广播特性。

以下代码实现了对input_x作乘方数为input_y的乘方操作：
```python
import numpy as np            
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P
input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
print(input_x**input_y)
```

输出如下：
```
[ 1.  8. 64.]
```

#### 加法

上述代码中`input_x`和`input_y`的相加实现方式如下：
```python
print(input_x + input_y)
```

输出如下：
```
[4.0 5.0 7.0]
```

#### Element-wise乘法

以下代码实现了Element-wise乘法示例：
```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

input_x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
input_y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
mul = P.Mul()
res = mul(input_x, input_y)

print(res)
```

输出如下：
```
[4. 10. 18]
```

#### 求三角函数

以下代码实现了Acos：
```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

acos = P.ACos()
input_x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
output = acos(input_x)
print(output)
```

输出如下：
```
[0.7377037, 1.5307858, 1.2661037，0.97641146]
```
### 向量运算

向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量。

#### Squeeze

以下代码实现了压缩第3个通道维度为1的通道：
```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
squeeze = P.Squeeze(2)
output = squeeze(input_tensor)

print(output)
```

输出如下：
```
[[1. 1.]
 [1. 1.]
 [1. 1.]]
```
#### 求Sparse2Dense

以下代码实现了对Sparse2Dense示例：
```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.ops.operations as P

indices = Tensor([[0, 1], [1, 2]])
values = Tensor([1, 2], dtype=ms.float32)
dense_shape = (3, 4)
out = P.SparseToDense()(indices, values, dense_shape)

print(out)
```

输出如下：
```
[[0, 1, 0, 0],
 [0, 0, 2, 0],
 [0, 0, 0, 0]]
```

### 矩阵运算

矩阵运算包括矩阵乘法、矩阵范数、矩阵行列式、矩阵求特征值、矩阵分解等运算。

#### 矩阵乘法

以下代码实现了input_x 和 input_y的矩阵乘法：
```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as P

input_x = Tensor(np.ones(shape=[1, 3]), mindspore.float32)
input_y = Tensor(np.ones(shape=[3, 4]), mindspore.float32)
matmul = P.MatMul()
output = matmul(input_x, input_y)

print(output)
```

输出如下：
```
[[3. 3. 3. 3.]]
```

#### 广播机制

广播表示输入各变量channel数目不一致时，改变他们的channel 数以得到结果。

以下代码实现了广播机制的示例：
```python
from mindspore import Tensor
from mindspore.communication import init
import mindspore.nn as nn
import mindspore.ops.operations as P
import numpy as np

init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.broadcast = P.Broadcast(1)

    def construct(self, x):
        return self.broadcast((x,))

input_ = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_)
```

## 网络操作

网络操作包括特征提取、激活函数、LossFunction、优化算法等。

### 特征提取

特征提取是机器学习中的常见操作，核心是提取比原输入更具代表性的Tensor。

#### 卷积操作

以下代码实现了常见卷积操作之一的2D convolution 操作：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

input = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
conv2d = P.Conv2D(out_channel=32, kernel_size=3)
conv2d(input, weight)
```

#### 卷积的反向传播算子操作

以下代码实现了反向梯度算子传播操作的具体代码，输出存于dout， weight：

```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore
import mindspore.ops.functional as F

dout = Tensor(np.ones([10, 32, 30, 30]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
x = Tensor(np.ones([10, 32, 32, 32]))
conv2d_backprop_input = P.Conv2DBackpropInput(out_channel=32, kernel_size=3)
conv2d_backprop_input(dout, weight, F.shape(x))
```

### 激活函数

以下代码实现Softmax激活函数计算：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
softmax = P.Softmax()
res = softmax(input_x)

print(res)
```

输出如下：
```
[0.01165623 0.03168492 0.08612854 0.23412167 0.6364086]
```

### LossFunction

#### L1Loss

以下代码实现了L1 loss function：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

loss = P.SmoothL1Loss()
input_data = Tensor(np.array([1, 2, 3]), mindspore.float32)
target_data = Tensor(np.array([1, 2, 2]), mindspore.float32)
loss(input_data, target_data)
```

输出如下：
```
[0.  0.  0.5]
```

### 优化算法

#### SGD

以下代码实现了SGD梯度下降算法的具体实现，输出是result：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

sgd = P.SGD()
parameters = Tensor(np.array([2, -0.5, 1.7, 4]), mindspore.float32)
gradient = Tensor(np.array([1, -1, 0.5, 2]), mindspore.float32)
learning_rate = Tensor(0.01, mindspore.float32)
accum = Tensor(np.array([0.1, 0.3, -0.2, -0.1]), mindspore.float32)
momentum = Tensor(0.1, mindspore.float32)
stat = Tensor(np.array([1.5, -0.3, 0.2, -0.7]), mindspore.float32)
result = sgd(parameters, gradient, learning_rate, accum, momentum, stat)

print(result)
```

输出如下：
```
[0.  0.  0.  0.]
```

## 数组操作

数组操作指操作对象是一些数组的操作。

### DType

返回跟输入的数据类型一致的并且适配Mindspore的Tensor变量，常用于Mindspore工程内。

```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
typea = P.DType()(input_tensor)

print(typea)
```

输出如下：
```
Float32
```

### Cast

转换输入的数据类型并且输出与目标数据类型相同的变量。

```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
input_x = Tensor(input_np)
type_dst = mindspore.float16
cast = P.Cast()
result = cast(input_x, type_dst)
print(result.type())
```

输出结果:
```
mindspore.float16
```

### Shape

返回输入数据的形状。

以下代码实现了返回输入数据input_tensor的操作：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

input_tensor = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
shape = P.Shape()
output = shape(input_tensor)
print(output)
```

输出如下：
```
[3, 2, 1]
```

## 图像操作

图像操作包括图像预处理操作，如图像剪切（Crop，便于得到大量训练样本）和大小变化（Reise，用于构建图像金子塔等）。

以下代码实现了Crop和Resize操作：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore.common.dtype as mstype
from mindpore.ops import composite as C

class CropAndResizeNet(nn.Cell):
    def __init__(self, crop_size):
        super(CropAndResizeNet, self).__init__()
        self.crop_and_resize = P.CropAndResize()
        self.crop_size = crop_size

    def construct(self, x, boxes, box_index):
        return self.crop_and_resize(x, boxes, box_index, self.crop_size)

BATCH_SIZE = 1
NUM_BOXES = 5
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 3
image = np.random.normal(size=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS]).astype(np.float32)
boxes = np.random.uniform(size=[NUM_BOXES, 4]).astype(np.float32)
box_index = np.random.uniform(size=[NUM_BOXES], low=0, high=BATCH_SIZE).astype(np.int32)
crop_size = (24, 24)
crop_and_resize = CropAndResizeNet(crop_size=crop_size)
output = crop_and_resize(Tensor(image), Tensor(boxes), Tensor(box_index))
print(output.asnumpy())
```

## 编码运算

编码运算包括BoundingBox Encoding、BoundingBox Decoding、IOU计算等。

### BoundingBoxEncode

对物体所在区域方框进行编码，得到类似PCA的更精简信息，以便做后续类似特征提取，物体检测，图像恢复等任务。

以下代码实现了对anchor_box和groundtruth_box的boundingbox encode：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
groundtruth_box = Tensor([[3,1,2,2],[1,2,1,4]],mindspore.float32)
boundingbox_encode = P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
res = boundingbox_encode(anchor_box, groundtruth_box)
print(res)
```

输出如下:
```
[[5.0000000e-01  5.0000000e-01  -6.5504000e+04  6.9335938e-01]
 [-1.0000000e+00  2.5000000e-01  0.0000000e+00  4.0551758e-01]]
```

### BoundingBoxDecode

编码器对区域位置信息解码之后，用此算子进行解码。

以下代码实现了：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

anchor_box = Tensor([[4,1,2,1],[2,2,2,3]],mindspore.float32)
deltas = Tensor([[3,1,2,2],[1,s2,1,4]],mindspore.float32)
boundingbox_decode = P.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), max_shape=(768, 1280), wh_ratio_clip=0.016)
res = boundingbox_decode(anchor_box, deltas)
print(res)
```

输出如下：
```
[[4.1953125  0.  0.  5.1953125]
 [2.140625  0.  3.859375  60.59375]]
```

### IOU计算

计算预测的物体所在方框和真实物体所在方框的交集区域与并集区域的占比大小，常作为一种损失函数，用以优化模型。

以下代码实现了计算两个变量anchor_boxes和gt_boxes之间的IOU，以out输出：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore

iou = P.IOU()
anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
out = iou(anchor_boxes, gt_boxes)
print(out)
```

输出如下：
```
[[0.  -0.  0.]
 [0.  -0.  0.]
 [0.   0.  0.]]
```

## 调试操作

调试操作指的是用于调试网络的一些常用算子及其操作，例如Debug等, 此操作非常方便，对入门深度学习重要，极大提高学习者的学习体验。

### Debug

输出Tensor变量的数值，方便用户随时随地打印想了解或者debug必需的某变量数值。

以下代码实现了输出x这一变量的值：
```python
from mindspore import nn

class DebugNN(nn.Cell):
    def __init__(self,):
        self.debug = nn.Debug()

    def construct(self, x, y):
        self.debug()
        x = self.add(x, y)
        self.debug(x)
        return x
```

### HookBackward

打印中间变量的梯度，是比较常用的算子，目前仅支持Pynative模式。

以下代码实现了打印中间变量(例中x,y)的梯度：
```python
from mindspore import Tensor
import mindspore.ops.operations as P
import numpy as np
import mindspore.common.dtype as mstype
from mindpore.ops import composite as C

def hook_fn(grad_out):
    print(grad_out)

grad_all = C.GradOperation(get_all=True)
hook = P.HookBackward(hook_fn)

def hook_test(x, y):
    z = x * y
    z = hook(z)
    z = z * y
    return z

def backward(x, y):
    return grad_all(hook_test)(Tensor(x, mstype.float32), Tensor(y, mstype.float32))

backward(1, 2)
```
