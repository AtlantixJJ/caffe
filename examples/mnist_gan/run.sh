# ./run.sh <pool> <deconv/upsample/pxs>
mkdir log
rm -rf log/mnist_$1_$2
mkdir log/mnist_$1_$2

# make d solver
cat examples/mnist_gan/d_solver > examples/mnist_gan/$1_$2_d_solver.prototxt
echo "" >> examples/mnist_gan/$1_$2_d_solver.prototxt
echo net: \"examples/mnist_gan/$1_d.prototxt\" >> examples/mnist_gan/$1_$2_d_solver.prototxt
echo snapshot_prefix: \"log/mnist_$1_$2/\" >> examples/mnist_gan/$1_$2_d_solver.prototxt

# make g solver
cat examples/mnist_gan/g_solver > examples/mnist_gan/$1_$2_g_solver.prototxt
echo "" >> examples/mnist_gan/$1_$2_g_solver.prototxt
echo net: \"examples/mnist_gan/$2_g.prototxt\" >> examples/mnist_gan/$1_$2_g_solver.prototxt
echo snapshot_prefix: \"log/mnist_$1_$2/\" >> examples/mnist_gan/$1_$2_g_solver.prototxt

build/tools/caffe_gan train --timing 1\
    --d_solver=examples/mnist_gan/$1_$2_d_solver.prototxt \
    --g_solver=examples/mnist_gan/$1_$2_g_solver.prototxt \
    2>&1 | tee log/mnist_$1_$2/train.log