mkdir log/
mkdir log/mnist_pool
build/tools/caffe_gan train \
    --d_solver=examples/mnist_gan/d_pool_solver_mnist.prototxt \
    --g_solver=examples/mnist_gan/g_pxs_solver_mnist.prototxt