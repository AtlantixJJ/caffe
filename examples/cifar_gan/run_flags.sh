# ./run.sh <pool/pxs> <deconv/pxs>
mkdir log
rm -rf log/cifar_$1_$2
mkdir log/cifar_$1_$2

# make d solver
cat examples/cifar_gan/d_solver > examples/cifar_gan/$1_$2_d_solver.prototxt
echo "" >> examples/cifar_gan/$1_$2_d_solver.prototxt
echo net: \"examples/cifar_gan/$1_d.prototxt\" >> examples/cifar_gan/$1_$2_d_solver.prototxt
echo snapshot_prefix: \"log/cifar_$1_$2/\" >> examples/cifar_gan/$1_$2_d_solver.prototxt

# make g solver
cat examples/cifar_gan/g_solver > examples/cifar_gan/$1_$2_g_solver.prototxt
echo "" >> examples/cifar_gan/$1_$2_g_solver.prototxt
echo net: \"examples/cifar_gan/$2_g.prototxt\" >> examples/cifar_gan/$1_$2_g_solver.prototxt
echo snapshot_prefix: \"log/cifar_$1_$2/\" >> examples/cifar_gan/$1_$2_g_solver.prototxt

build/tools/caffe_gan train $3\
    --d_solver=examples/cifar_gan/$1_$2_d_solver.prototxt \
    --g_solver=examples/cifar_gan/$1_$2_g_solver.prototxt \
    2>&1 | tee log/cifar_$1_$2/train.log