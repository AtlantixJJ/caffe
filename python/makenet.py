import sys
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

cifar_lmda_dir = "examples/cifar10/cifar10_train_lmdb"
vsp_lmdb_dir = "examples/vsp/layer2_2channel256-4590-float_train_lmdb"

def partial(func, **kwargs):
    def _func(x):
        return func(x, **kwargs)
    return _func

def relu(x):
    return L.ReLU(x, in_place=True)

def lrelu(x):
    return L.ReLU(x, in_place=True, negative_slope=0.2)

def add_layers(net, layers, layer_names):
    for n,l in zip(layer_names, layers):
        setattr(net, n, l)
"""
def downsample_d(downsample=4, batch_size=256):
    net = caffe.NetSpec()
    x = L.Data(batch_size=batch_size, backend=caffe.params.Data.LMDB, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN), transform_param=dict(scale=1/127.5,mean_value=127.5))
    net.disc_label = L.Input(input_param=dict(shape={'dim': [batch_size, 1]}))

    x = L.Convolution(net.data, num_output=128, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu1 = L.ReLU(net.conv1, in_place=True, negative_slope=0.2) # 32x32
"""

def upsample_g(upsample=4, batch_size=256):
    net = caffe.NetSpec()

    x = L.RandVec(randvec_param={
        'batch_size': batch_size,
        'dim': 128,
        'lower': -1.0,
        'upper': 1.0})

    layers = [x]
    layer_names = ["data"]

    x = L.InnerProduct(x, num_output=4*4*1024, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    layers.append(x); layer_names.append("fc1")
    x = L.Reshape(x, reshape_param=dict(shape={'dim': [batch_size, 1024, 4, 4]}))
    layers.append(x); layer_names.append("reshape1")
    x = L.BatchNorm(x)
    layers.append(x); layer_names.append("bn1")
    x = L.ReLU(x, in_place=True)
    layers.append(x); layer_names.append("relu1")
    
    lower_dim = [1024, 512, 256, 128, 64] # 8x upsample
    upper_dim = lower_dim[1:][:upsample] # from 4x4 to 32x32
    lower_dim = lower_dim[:upsample]
    for i in range(len(lower_dim)):
        ind = i + 2
        x = L.Deconvolution(x, convolution_param=dict(bias_term=False, num_output=lower_dim[i], kernel_size=2, stride=2, pad=0, weight_filler=dict(type='bilinear')), param=dict(lr_mult=0, decay_mult=0))
        layers.append(x); layer_names.append("upsample%d"%ind)
        x = L.Convolution(x, num_output=upper_dim[i], kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        layers.append(x); layer_names.append("conv%d"%ind)
        x = L.BatchNorm(x)
        layers.append(x); layer_names.append("bn%d"%ind)
        x = L.ReLU(x, in_place=True)
        layers.append(x); layer_names.append("relu%d"%ind)

    x = L.Convolution(x, num_output=3, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    layers.append(x); layer_names.append("conv_out")
    x = L.TanH(x)
    layers.append(x); layer_names.append("output")

    add_layers(net, layers, layer_names)
    return net.to_proto()

def mnist_upsample_g(batch_size=256):
    net = caffe.NetSpec()

    x = L.RandVec(randvec_param={
        'batch_size': batch_size,
        'dim': 128,
        'lower': -1.0,
        'upper': 1.0})

    layers = [x]
    layer_names = ["data"]

    x = L.InnerProduct(x, num_output=7*7*256, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    layers.append(x); layer_names.append("fc1")
    x = L.Reshape(x, reshape_param=dict(shape={'dim': [-1, 256, 7, 7]}))
    layers.append(x); layer_names.append("reshape1")
    x = L.BatchNorm(x, [dict()])
    layers.append(x); layer_names.append("bn1")
    x = L.ReLU(x, in_place=True)
    layers.append(x); layer_names.append("relu1")
    
    lower_dim = [256, 256] # 4x upsample
    upper_dim = lower_dim[1:] + [128] # from 7x7 to 28x28
    for i in range(len(lower_dim)):
        ind = i + 2
        x = L.Deconvolution(x, convolution_param=dict(bias_term=False, num_output=lower_dim[i], kernel_size=2, stride=2, pad=0, weight_filler=dict(type='bilinear')), param=dict(lr_mult=0, decay_mult=0))
        layers.append(x); layer_names.append("upsample%d"%ind)
        x = L.Convolution(x, num_output=upper_dim[i], kernel_size=3, stride=1, pad=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
        layers.append(x); layer_names.append("conv%d"%ind)
        x = L.BatchNorm(x)
        layers.append(x); layer_names.append("bn%d"%ind)
        x = L.ReLU(x, in_place=True)
        layers.append(x); layer_names.append("relu%d"%ind)

    x = L.Convolution(x, num_output=1, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
    layers.append(x); layer_names.append("conv_out")
    x = L.TanH(x)
    layers.append(x); layer_names.append("output")

    add_layers(net, layers, layer_names)
    return net.to_proto()

def simple_residual_block(name, net, x, dim, activation_fn, use_bn=True):
    conv1 = L.Convolution(x, num_output=dim, kernel_size=3, stride=1, pad=1,
        weight_filler=dict(type="gaussian", std=0.02))
    if use_bn:
        bn1 = L.BatchNorm(conv1)
        relu1 = activation_fn(bn1)
    else:
        relu1 = activation_fn(conv1)
    conv2 = L.Convolution(relu1, num_output=dim, kernel_size=3, stride=1, pad=1,
        weight_filler=dict(type="gaussian", std=0.02))
    if use_bn:
        bn2 = L.BatchNorm(conv2)
        add = L.Eltwise(x, bn2)
        names = ['conv1', 'bn1', 'act1', 'conv2', 'bn2', 'add']
        layers = [conv1, bn1, relu1, conv2, bn2, add]
    else:
        add = L.Eltwise(x, conv2)
        names = ['conv1', 'act1', 'conv2', 'add']
        layers = [conv1, relu1, conv2, add]

    names = [name + i_ for i_ in names]
    add_layers(net, layers, names)

def cifar10_res_g(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.RandVec(randvec_param={
        'batch_size': batch_size,
        'dim': 128,
        'lower': -1.0,
        'upper': 1.0})

    net.fc = L.InnerProduct(net.data, num_output=4*4*1024, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.reshape = L.Reshape(net.fc, reshape_param=dict(shape={'dim': [batch_size, 1024, 4, 4]}))
    net.bn = L.BatchNorm(net.reshape)
    net.relu = L.ReLU(net.bn, in_place=True)

    net.deconv1 = L.Deconvolution(net.relu, convolution_param=dict(num_output=512, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn1 = L.BatchNorm(net.deconv1)
    net.relu1 = L.ReLU(net.bn1, in_place=True) # 8x8

    net.deconv2 = L.Deconvolution(net.relu1, convolution_param=dict(num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn2 = L.BatchNorm(net.deconv2)
    net.relu2 = L.ReLU(net.bn2, in_place=True) # 16x16

    net.res1_out = simple_residual_block("res1", net, net.relu2, 256, relu)
    net.bn3 = L.BatchNorm(net.res1_out)
    net.relu3 = L.ReLU(net.bn3, in_place=True)

    net.res2_out = simple_residual_block("res2", net, net.relu3, 256, relu)
    net.bn4 = L.BatchNorm(net.res2_out)
    net.relu4 = L.ReLU(net.bn4, in_place=True)

    net.deconv3 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn5 = L.BatchNorm(net.deconv3)
    net.relu5 = L.ReLU(net.bn5, in_place=True) # 32x32

    net.conv_output = L.Convolution(net.relu5, num_output=3, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.output = L.TanH(net.conv_output)

    return net.to_proto()

def cifar10_res_d(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, backend=caffe.params.Data.LMDB, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN), transform_param=dict(scale=1/128,mean_value=127.5))
    net.disc_label = L.Input(input_param=dict(shape={'dim': [batch_size, 1]}))

    net.conv1 = L.Convolution(net.data, num_output=128, kernel_size=3, stride=1,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu1 = L.ReLU(net.conv1, in_place=True, negative_slope=0.2) # 32x32

    net.conv2 = L.Convolution(net.relu1, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu2 = L.ReLU(net.conv2, in_place=True, negative_slope=0.2) # 16x16

    net.res1_out = simple_residual_block("res1", net, net.relu2, 256, lrelu, use_bn=False)
    net.relu3 = L.ReLU(net.res1_out, in_place=True, negative_slope=0.2)

    net.res2_out = simple_residual_block("res2", net, net.relu3, 256, lrelu, use_bn=False)
    net.relu4 = L.ReLU(net.res2_out, in_place=True)

    net.conv3 = L.Convolution(net.relu4, num_output=512, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu5 = L.ReLU(net.conv3, in_place=True) # 8x8

    net.conv4 = L.Convolution(net.relu5, num_output=1024, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu6 = L.ReLU(net.conv4, in_place=True) # 4x4

    net.conv5 = L.Convolution(net.relu6, num_output=1, kernel_size=4, stride=1,
            pad=0, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))

    net.disc_loss = L.SigmoidCrossEntropyLoss(net.conv5, net.disc_label)
    return net.to_proto()

def cifar10_ae(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN),
        transform_param=dict(scale=1/128,mean_value=127.5))

    ## Convolutional Layer 1
    net.conv1 = L.Convolution(net.data, num_output=64, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu1 = L.ReLU(net.conv1, in_place=True, negative_slope=0.2)
    # 16x16

    ## Convolutional Layer 2
    net.conv2 = L.Convolution(net.relu1, num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu2 = L.ReLU(net.conv2, in_place=True, negative_slope=0.2)
    # 8x8

    ## Convolutional Layer 3
    net.conv3 = L.Convolution(net.relu2, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.relu3 = L.ReLU(net.conv3, in_place=True, negative_slope=0.2)
    # 4x4

    net.deconv3 = L.Deconvolution(net.relu3, convolution_param=dict(num_output=256, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.relu4 = L.ReLU(net.deconv3, in_place=True)
    # 8x8

    net.deconv2 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=128, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.relu5 = L.ReLU(net.deconv2, in_place=True)
    # 16x16

    net.deconv1 = L.Deconvolution(net.relu5, convolution_param=dict(num_output=64, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.relu6 = L.ReLU(net.deconv1, in_place=True)
    # 32x32

    net.conv_output = L.Convolution(net.relu6, num_output=3, kernel_size=3, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.output = L.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = L.EuclideanLoss(net.output, net.data)

    return net.to_proto()

def cifar10_ae256x256(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, source=cifar_lmda_dir, include=dict(phase=caffe.TRAIN),
        transform_param=dict(scale=1/128,mean_value=127.5))

    ## Convolutional Layer 1
    net.conv1 = L.Convolution(net.data, num_output=64, kernel_size=5, stride=1,
            pad=2, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn1   = L.BatchNorm(net.conv1)
    net.relu1 = L.ReLU(net.bn1, in_place=True)
    # 256x256

    ## Convolutional Layer 2
    net.conv2 = L.Convolution(net.relu1, num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn2   = L.BatchNorm(net.conv2)
    net.relu2 = L.ReLU(net.bn2, in_place=True)
    # 128x128

    ## Convolutional Layer 3
    net.conv3 = L.Convolution(net.relu2, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn3   = L.BatchNorm(net.conv3)
    net.relu3 = L.ReLU(net.bn3, in_place=True)
    # 64x64

    ## Convolutional Layer 4
    net.conv4 = L.Convolution(net.relu3, num_output=512, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn4   = L.BatchNorm(net.conv4)
    net.relu4 = L.ReLU(net.bn4, in_place=True)
    # 32x32

    net.deconv3 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn5   = L.BatchNorm(net.deconv3)
    net.relu5 = L.ReLU(net.bn5, in_place=True)
    # 64x64

    net.deconv2 = L.Deconvolution(net.relu5, convolution_param=dict(num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn6   = L.BatchNorm(net.deconv2)
    net.relu6 = L.ReLU(net.bn6, in_place=True)
    # 128x128

    net.deconv1 = L.Deconvolution(net.relu6, convolution_param=dict(num_output=64, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn7   = L.BatchNorm(net.deconv1)
    net.relu7 = L.ReLU(net.bn7, in_place=True)
    # 256x256

    net.conv_output = L.Convolution(net.relu7, num_output=3, kernel_size=5, stride=1,
            pad=2, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.output = L.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = L.EuclideanLoss(net.output, net.data)

    return net.to_proto()

def vsp(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, backend=caffe.params.Data.LMDB, source=vsp_lmdb_dir, include=dict(phase=caffe.TRAIN))

    net.data_A, net.data_B = L.Slice(net.data, ntop=2, slice_param=dict(axis=1, slice_point=1))

    ## Convolutional Layer 1
    net.conv1 = L.Convolution(net.data_A, num_output=64, kernel_size=5, stride=1,
            pad=2, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn1   = L.BatchNorm(net.conv1, in_place=True)
    net.relu1 = L.ReLU(net.bn1, in_place=True)
    # 256x256

    ## Convolutional Layer 2
    net.conv2 = L.Convolution(net.relu1, num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn2   = L.BatchNorm(net.conv2, in_place=True)
    net.relu2 = L.ReLU(net.bn2, in_place=True)
    # 128x128

    ## Convolutional Layer 3
    net.conv3 = L.Convolution(net.relu2, num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn3   = L.BatchNorm(net.conv3, in_place=True)
    net.relu3 = L.ReLU(net.bn3, in_place=True)
    # 64x64

    ## Convolutional Layer 4
    net.conv4 = L.Convolution(net.relu3, num_output=512, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.bn4   = L.BatchNorm(net.conv4, in_place=True)
    net.relu4 = L.ReLU(net.bn4, in_place=True)
    # 32x32

    net.deconv3 = L.Deconvolution(net.relu4, convolution_param=dict(num_output=256, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn5   = L.BatchNorm(net.deconv3, in_place=True)
    net.relu5 = L.ReLU(net.bn5, in_place=True)
    # 64x64

    net.deconv2 = L.Deconvolution(net.relu5, convolution_param=dict(num_output=128, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn6   = L.BatchNorm(net.deconv2, in_place=True)
    net.relu6 = L.ReLU(net.bn6, in_place=True)
    # 128x128

    net.deconv1 = L.Deconvolution(net.relu6, convolution_param=dict(num_output=64, kernel_size=4, stride=2,
            pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
    net.bn7   = L.BatchNorm(net.deconv1, in_place=True)
    net.relu7 = L.ReLU(net.bn7, in_place=True)
    # 256x256

    net.conv_output = L.Convolution(net.relu7, num_output=1, kernel_size=5, stride=1,
            pad=2, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.output = L.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = L.EuclideanLoss(net.output, net.data_B)

    return net.to_proto()

class UNetSkipConnectBlock(object):
    def __init__(self, name, net, in_dim, out_dim, submodule=None, outer_most=False):
        self.name = name
        self.net = net
        self.outer_most = outer_most
        
        downconv = partial(L.Convolution, num_output=in_dim, kernel_size=4, stride=2,
                pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
        downnorm = partial(L.BatchNorm, in_place=True)
        downact = partial(L.ReLU, in_place=True, negative_slope=0.2)
        
        upconv = partial(L.Deconvolution, convolution_param=dict(num_output=out_dim, kernel_size=4, stride=2,
                pad=1, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant')))
        upnorm = partial(L.BatchNorm, in_place=True)

        upact = partial(L.ReLU, in_place=True)

        if submodule is None:
            fn_seq = [downconv, downnorm, downact, upconv, upnorm, upact]
            fn_name= ['downconv', 'downnorm', 'downact', 'upconv', 'upnorm', 'upact']
            fn_name= [name + '_' + n for n in fn_name]
        else:
            fn_seq = [downconv, downnorm, downact] + [submodule] + [upconv, upnorm, upact]
            name1 = ['downconv', 'downnorm', 'downact']
            name2 = ['upconv', 'upnorm', 'upact']
            name1 = [name + "_" + n for n in name1]
            name2 = [name + "_" + n for n in name2]
            fn_name = name1 + ["sub"] + name2

        self.fn_seq = fn_seq
        self.fn_name = fn_name

    def __call__(self, x):
        """
        Build the graph
        """
        for f, n in zip(self.fn_seq, self.fn_name):
            if isinstance(f, UNetSkipConnectBlock):
                X = f(x)
                if not self.outer_most:
                    l = L.Concat(x, X, concat_param=dict(axis=1))
                    setattr(self.net, self.name + "_concat", l)
                    x = l
                else:
                    x = X
            else:
                setattr(self.net, n, f(x))
                x = getattr(self.net, n)
        return x
        #fn_seq += partial(L.Concat, concat_param=dict(axis=1))

def vsp_unet(batch_size=128):
    net = caffe.NetSpec()

    net.data = L.Data(batch_size=batch_size, backend=caffe.params.Data.LMDB, source=vsp_lmdb_dir, include=dict(phase=caffe.TRAIN))

    net.data_A, net.data_B = L.Slice(net.data, ntop=2, slice_param=dict(axis=1, slice_point=1))

    ch = 64
    chs = [ch * 8, ch * 4, ch * 2, ch]
    sub = None
    sub = UNetSkipConnectBlock("layer%d" % 1, net, ch * 8, ch * 4, sub, outer_most=False)
    sub = UNetSkipConnectBlock("layer%d" % 2, net, ch * 4, ch * 4, sub, outer_most=False)
    sub = UNetSkipConnectBlock("layer%d" % 3, net, ch * 2, ch * 2, sub, outer_most=False)
    sub = UNetSkipConnectBlock("layer%d" % 4, net, ch * 1, ch * 1, sub, outer_most=False)
    
    x = sub(net.data_A)

    net.conv_output = L.Convolution(x, num_output=1, kernel_size=5, stride=1,
            pad=2, weight_filler=dict(type='xavier') , bias_filler=dict(type='constant'))
    net.output = L.TanH(net.conv_output)

    ## Euclidean Loss
    net.loss = L.EuclideanLoss(net.output, net.data_B)

    return net.to_proto()
        

func = sys.argv[1]
output_file = sys.argv[2]
batch_size = sys.argv[3]
# batch_size = 50
with open(output_file, 'w') as f:
    f.write(str(locals()[func](int(batch_size))))
