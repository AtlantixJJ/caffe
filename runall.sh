CUDA_VISIBLE_DEVICES=0 bash examples/cifar_gan/run.sh pool deconv &
CUDA_VISIBLE_DEVICES=2 bash examples/cifar_gan/run.sh pool pxs &
CUDA_VISIBLE_DEVICES=3 bash examples/cifar_gan/run.sh pool upsample &
CUDA_VISIBLE_DEVICES=4 bash examples/mnist_gan/run.sh pool deconv &
CUDA_VISIBLE_DEVICES=6 bash examples/mnist_gan/run.sh pool pxs &
CUDA_VISIBLE_DEVICES=2 bash examples/mnist_gan/run.sh pool upsample &
CUDA_VISIBLE_DEVICES=3 bash examples/mnist_gan/run.sh pool upsample_scale &
CUDA_VISIBLE_DEVICES=4 bash examples/cifar_gan/run.sh pool upsample_scale &