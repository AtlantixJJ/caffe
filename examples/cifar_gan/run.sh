rm -r log/cifar_$1
mkdir log
mkdir log/cifar_$1
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo net: snapshot_prefix: \"log/cifar_pxs\" >> examples/cifar_gan/d_solver.prototxt

cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar_gan/g_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt