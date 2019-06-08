CUDA_VISIBLE_DEVICES=2 bash examples/cifar_gan/run_flags.sh pool deconv "--timing 1" &
CUDA_VISIBLE_DEVICES=3 bash examples/cifar_gan/run_flags.sh pool pxs "--timing 1" &
CUDA_VISIBLE_DEVICES=4 bash examples/cifar_gan/run_flags.sh pool upsample "--timing 1" &
CUDA_VISIBLE_DEVICES=2 bash examples/cifar_gan/run_flags.sh res res "--timing 1" &
CUDA_VISIBLE_DEVICES=3 bash examples/mnist_gan/run_flags.sh pool deconv "--timing 1" &
CUDA_VISIBLE_DEVICES=4 bash examples/mnist_gan/run_flags.sh pool pxs "--timing 1" &
CUDA_VISIBLE_DEVICES=2 bash examples/mnist_gan/run_flags.sh pool upsample "--timing 1" &