# examples/cifar_gan/run_aegan.sh pool
rm -r log/cifar_aegan_$1
mkdir log
mkdir log/cifar_aegan_$1
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo snapshot_prefix: \"log/cifar_aegan_$1\" >> examples/cifar_gan/d_solver.prototxt

cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar10/cifar10_autoencoder_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt $2