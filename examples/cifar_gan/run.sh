mkdir log
mkdir log/cifar_gan
build/tools/caffe_gan train --g_solver=examples/cifar_gan/g_solver.prototxt --d_solver=examples/cifar_gan/d_solver.prototxt