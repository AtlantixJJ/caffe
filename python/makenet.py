import sys
import caffe
from caffe import layers as cl

def create_neural_net(input_file, batch_size=128):
    net = caffe.NetSpec()

    net.data, net.label = cl.HDF5Data(batch_size=batch_size, source=input_file,
            ntop=2, include=dict(phase=caffe.TRAIN))

    ## Convolutional Layer 1
    net.conv1 = cl.Convolution(net.data, num_output=64, kernel_size=9, stride=1,
            pad=0, weight_filler=dict(type='gaussian', std=0.001),
            param=[{'lr_mult':1},{'lr_mult':0.1}],
            bias_filler=dict(type='constant', value=0))
    net.relu1 = cl.ReLU(net.conv1, in_place=True)

    ## Convolutional Layer 2
    net.conv2 = cl.Convolution(net.relu1, num_output=32, kernel_size=1, stride=1,
            pad=0, weight_filler=dict(type='gaussian', std=0.001),
            param=[{'lr_mult':1},{'lr_mult':0.1}],
            bias_filler=dict(type='constant', value=0))
    net.relu2 = cl.ReLU(net.conv2, in_place=True)

    ## Convolutional Layer 3
    net.conv3 = cl.Convolution(net.relu2, num_output=1, kernel_size=5, stride=1,
            pad=0, weight_filler=dict(type='gaussian', std=0.001),
            param=[{'lr_mult':0.1},{'lr_mult':0.1}],
            bias_filler=dict(type='constant', value=0))
    net.relu3 = cl.ReLU(net.conv3, in_place=True)

    ## Euclidean Loss
    net.loss = cl.EuclideanLoss(net.conv3, net.label)

    return net.to_proto()

train_h5list_file = "a"
output_file = "b"
# batch_size = 50
with open(output_file, 'w') as f:
    f.write(str(create_neural_net(train_h5list_file)))