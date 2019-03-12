# ./run.sh <pool/pxs> <deconv/pxs>
mkdir log
rm -rf log/mnist_$1_$2
mkdir log/mnist_$1_$2

# make d solver
cat d_solver > $1_$2_d_solver.prototxt
echo "" >> $1_$2_d_solver.prototxt
echo net: \"examples/mnist_gan/$1_d.prototxt\" >> $1_$2_d_solver.prototxt
echo snapshot_prefix: \"log/mnist_$1_$2/\" >> $1_$2_d_solver.prototxt

# make g solver
cat g_solver > $1_$2_g_solver.prototxt
echo "" >> $1_$2_g_solver.prototxt
echo net: \"examples/mnist_gan/$2_g.prototxt\" >> $1_$2_g_solver.prototxt
echo snapshot_prefix: \"log/mnist_$1_$2/\" >> $1_$2_g_solver.prototxt

build/tools/caffe_gan train \
    --d_solver=$1_$2_d_solver.prototxt \
    --g_solver=$1_$2_g_solver.prototxt \
    2>&1 | tee log/mnist_$1_$2/train.log