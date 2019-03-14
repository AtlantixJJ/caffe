# ./run.sh <pool/pxs> <deconv/pxs>
mkdir log
rm -rf log/vsp_pool_ae
mkdir log/vsp_pool_ae

# make d solver
cat examples/vsp/d_solver > examples/vsp/pool_ae_d_solver.prototxt
echo "" >> examples/vsp/pool_ae_d_solver.prototxt
echo net: \"examples/vsp/pool_d.prototxt\" >> examples/vsp/pool_ae_d_solver.prototxt
echo snapshot_prefix: \"log/vsp_pool_ae/\" >> examples/vsp/pool_ae_d_solver.prototxt

# make g solver
cat examples/vsp/d_solver > examples/vsp/pool_ae_g_solver.prototxt
echo "" >> examples/vsp/pool_ae_g_solver.prototxt
echo net: \"examples/vsp/ae_g.prototxt\" >> examples/vsp/pool_ae_g_solver.prototxt
echo snapshot_prefix: \"log/vsp_pool_ae/\" >> examples/vsp/pool_ae_g_solver.prototxt

build/tools/caffe_gan train \
    --d_solver=examples/vsp/pool_ae_d_solver.prototxt \
    --g_solver=examples/vsp/pool_ae_g_solver.prototxt \
    2>&1 | tee log/vsp_pool_ae/train.log