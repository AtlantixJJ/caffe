mkdir log
mkdir log/cifar_$1
build/tools/caffe_gan train \
    --g_solver=examples/cifar_gan/g_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt