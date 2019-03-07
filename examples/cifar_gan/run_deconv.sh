# examples/cifar_gan/run.sh pool
rm -r log/cifar_deconv_$1

mkdir log
mkdir log/cifar_deconv_$1
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo snapshot_prefix: \"log/cifar_deconv_$1\" >> examples/cifar_gan/d_solver.prototxt

cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar_gan/deconv_g_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt