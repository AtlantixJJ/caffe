import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

cifar_lmda_dir = "examples/cifar10/cifar10_train_lmdb"

def relu(x):
    return L.ReLU(x, in_place=True)

def lrelu(x):
    return L.ReLU(x, in_place=True, negative_slope=0.2)

def simple_residual_block(name, net, x, dim, activation_fn):
    conv1 = L.Convolution(x, num_output=dim, kernel_size=3, stride=1, pad=1,
        weight_filler=dict(type="gaussian", std=0.02))
    bn1 = L.BatchNorm(conv1)
    relu1 = activation_fn(bn1)
    conv2 = L.Convolution(relu1, num_output=dim, kernel_size=3, stride=1, pad=1,
        weight_filler=dict(type="gaussian", std=0.02))
    bn2 = L.BatchNorm(conv2)
    add = L.Eltwise(x, bn2)
    names = ['conv1', 'bn1', 'relu1', 'conv2', 'bn2', 'add']
    layers = [conv1, bn1, relu1, conv2, bn2, add]
    for n,l in zip(names, layers):
        setattr(net, name + "_" + n, l)
    return add

def create_cifar10_res_g(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.RandVec(randvec_param={
        'batch_size': 64,
        'dim': 128,
        'lower': -1.0,
        'upper': 1.0})
    #net.data = L.Data(batch_size=batch_size, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN), transform_param=dict(scale=1/128,mean_value=127.5))

    net.fc = L.InnerProduct(net.data, num_output=4*4*1024, weight_filler=dict(type='gaussian', std=0.02))
    net.reshape = L.Reshape(net.fc, reshape_param=dict(shape={'dim': [batch_size, 1024, 4, 4]}))
    net.bn = L.BatchNorm(net.reshape)
    net.relu = L.RelU(net.bn, in_place=True)

    net.deconv1 = L.Deconvolution(net.relu, convolution_param=dict(num_output=512, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.bn1 = L.BatchNorm(net.deconv1)
    net.relu1 = L.RelU(net.bn1, in_place=True) # 8x8

    net.deconv2 = L.Deconvolution(net.deconv1, convolution_param=dict(num_output=256, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.bn2 = L.BatchNorm(net.deconv1)
    net.relu2 = L.RelU(net.bn2, in_place=True) # 16x16

    net.res1_out = simple_residual_block("res1", net, net.relu2, 256, relu)
    net.bn3 = L.BatchNorm(net.res1_out)
    net.relu3 = L.RelU(net.bn3, in_place=True)

    net.res2_out = simple_residual_block("res2", net, net.relu3, 256, relu)
    net.bn4 = L.BatchNorm(net.res2_out)
    net.relu4 = L.RelU(net.bn4, in_place=True)

    net.deconv3 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=128, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.bn5 = L.BatchNorm(net.deconv3)
    net.relu5 = L.RelU(net.bn5, in_place=True) # 32x32

    net.conv_output = L.Convolution(net.relu5, num_output=3, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.output = L.TanH(net.conv_output)

    return net.to_proto()

def create_cifar10_ae(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN),
        transform_param=dict(scale=1/128,mean_value=127.5))

    ## Convolutional Layer 1
    net.conv1 = L.Convolution(net.data, num_output=64, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu1 = L.ReLU(net.conv1, in_place=True, negative_slope=0.2)
    # 16x16

    ## Convolutional Layer 2
    net.conv2 = L.Convolution(net.relu1, num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu2 = L.ReLU(net.conv2, in_place=True, negative_slope=0.2)
    # 8x8

    ## Convolutional Layer 3
    net.conv3 = L.Convolution(net.relu2, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.relu3 = L.ReLU(net.conv3, in_place=True, negative_slope=0.2)
    # 4x4

    net.deconv3 = L.Deconvolution(net.relu3, convolution_param=dict(num_output=256, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.relu4 = L.ReLU(net.deconv3, in_place=True)
    # 8x8

    net.deconv2 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=256, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.relu5 = L.ReLU(net.deconv2, in_place=True)
    # 16x16

    net.deconv1 = L.Deconvolution(net.relu5, convolution_param=dict(num_output=128, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02)))
    net.relu6 = L.ReLU(net.deconv1, in_place=True)
    # 32x32

    net.conv_output = L.Convolution(net.relu6, num_output=3, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='gaussian', std=0.02))
    net.output = L.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = L.EuclideanLoss(net.output, net.data)

    return net.to_proto()

output_file = sys.argv[1]
# batch_size = 50
with open(output_file, 'w') as f:
    f.write(str(create_cifar10_res_g()))