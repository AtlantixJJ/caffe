# examples/cifar_gan/run_aegan.sh <d proto>
rm -r log/cifar_pix2pix_$1
mkdir log
mkdir log/cifar_pix2pix_$1
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo snapshot_prefix: \"log/cifar_pix2pix_$1\" >> examples/cifar_gan/d_solver.prototxt

cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar10/cifar10_pix2pix_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt $2