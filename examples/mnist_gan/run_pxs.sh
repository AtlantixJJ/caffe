build/tools/caffe_gan train \
    --d_solver=examples/mnist_gan/d_pxs_solver_mnist.prototxt \
    --g_solver=examples/mnist_gan/g_pxs_solver_mnist.prototxt 2>&1 | tee log/mnist_pxs/train.log
