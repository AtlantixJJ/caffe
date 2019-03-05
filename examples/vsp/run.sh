rm -r log/vsp_$1
mkdir log
mkdir log/vsp_$1
echo net: \"examples/vsp/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo snapshot_prefix: \"log/cifar_pxs\" >> examples/cifar_gan/d_solver.prototxt

cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar_gan/g_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt