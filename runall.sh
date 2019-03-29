CUDA_VISIBLE_DEVICES=2 caffe/examples/cifar_gan/run.sh pool deconv
CUDA_VISIBLE_DEVICES=3 caffe/examples/cifar_gan/run.sh pool pxs
CUDA_VISIBLE_DEVICES=4 caffe/examples/mnist_gan/run.sh pool deconv
CUDA_VISIBLE_DEVICES=5 caffe/examples/mnist_gan/run.sh pool pxs