# ./run <d_proto> <g_proto>
rm -r log/cifar_$1_$2
mkdir log
mkdir log/cifar_$1_$2
# make d solver prototxt
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/d_solver.prototxt
echo snapshot_prefix: \"log/cifar_$1_$2\" >> examples/cifar_gan/d_solver.prototxt
cat examples/cifar_gan/d_solver >> examples/cifar_gan/d_solver.prototxt
# make g solver prototxt
echo net: \"examples/cifar_gan/$1_d.prototxt\" > examples/cifar_gan/g_solver.prototxt
cat examples/cifar_gan/g_solver >> examples/cifar_gan/g_solver.prototxt

build/tools/caffe_gan train \
    --g_solver=examples/cifar_gan/g_solver.prototxt \
    --d_solver=examples/cifar_gan/d_solver.prototxt