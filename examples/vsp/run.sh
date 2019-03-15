# ./run.sh <pool/pxs> <deconv/pxs>
mkdir log
rm -rf log/vsp_pool_$1
mkdir log/vsp_pool_$1

# make d solver
cat examples/vsp/d_solver > examples/vsp/pool_$1_d_solver.prototxt
echo "" >> examples/vsp/pool_$1_d_solver.prototxt
echo net: \"examples/vsp/pool_d.prototxt\" >> examples/vsp/pool_$1_d_solver.prototxt
echo snapshot_prefix: \"log/vsp_pool_$1/\" >> examples/vsp/pool_$1_d_solver.prototxt

# make g solver
cat examples/vsp/d_solver > examples/vsp/pool_$1_g_solver.prototxt
echo "" >> examples/vsp/pool_$1_g_solver.prototxt
echo net: \"examples/vsp/$1_g.prototxt\" >> examples/vsp/pool_$1_g_solver.prototxt
echo snapshot_prefix: \"log/vsp_pool_$1/\" >> examples/vsp/pool_$1_g_solver.prototxt

build/tools/caffe_gan train \
    --d_solver=examples/vsp/pool_$1_d_solver.prototxt \
    --g_solver=examples/vsp/pool_$1_g_solver.prototxt \
    2>&1 | tee log/vsp_pool_$1/train.log