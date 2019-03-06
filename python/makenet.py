import sys
import caffe
from caffe import layers as cl

cifar_lmda_dir = "examples/cifar10/cifar10_train_lmdb"

def create_cifar10_ae(input_file, batch_size=128):
    net = caffe.NetSpec()

    net.data = cl.Data(batch_size=batch_size, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN))

    ## Convolutional Layer 1
    net.conv1 = cl.Convolution(net.data, num_output=64, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu1 = cl.ReLU(net.conv1, in_place=True, negative_slope=0.2)
    # 16x16

    ## Convolutional Layer 2
    net.conv2 = cl.Convolution(net.relu1, num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu2 = cl.ReLU(net.conv2, in_place=True, negative_slope=0.2)
    # 8x8

    ## Convolutional Layer 3
    net.conv3 = cl.Convolution(net.relu2, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu3 = cl.ReLU(net.conv3, in_place=True, negative_slope=0.2)
    # 4x4

    net.deconv3 = cl.Deconvolution(net.relu2, num_output=256, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu3 = cl.ReLU(net.deconv3, in_place=True)
    # 8x8

    net.deconv2 = cl.Deconvolution(net.relu3, num_output=128, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu2 = cl.ReLU(net.deconv2, in_place=True)
    # 16x16

    net.deconv1 = cl.Deconvolution(net.relu2, num_output=64, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu1 = cl.ReLU(net.deconv2, in_place=True)
    # 32x32

    net.conv_output = cl.Convolution(net.relu1, num_output=3, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.output = cl.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = cl.EuclideanLoss(net.output, net.data)

    return net.to_proto()

train_h5list_file = "a"
output_file = "b"
# batch_size = 50
with open(output_file, 'w') as f:
    f.write(str(create_neural_net(train_h5list_file)))