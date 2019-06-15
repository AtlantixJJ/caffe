# ./run.sh <pool> <ae/unet>
mkdir log
rm -rf log/vsp_$1_$2
mkdir log/vsp_$1_$2

# make d solver
cat examples/vsp/d_solver > examples/vsp/$1_$2_d_solver.prototxt
echo "" >> examples/vsp/$1_$2_d_solver.prototxt
echo net: \"examples/vsp/$1_d.prototxt\" >> examples/vsp/$1_$2_d_solver.prototxt
echo snapshot_prefix: \"log/vsp_$1_$2/\" >> examples/vsp/$1_$2_d_solver.prototxt

# make g solver
cat examples/vsp/d_solver > examples/vsp/$1_$2_g_solver.prototxt
echo "" >> examples/vsp/$1_$2_g_solver.prototxt
echo net: \"examples/vsp/$2_g.prototxt\" >> examples/vsp/$1_$2_g_solver.prototxt
echo snapshot_prefix: \"log/vsp_$1_$2/\" >> examples/vsp/$1_$2_g_solver.prototxt

build/tools/caffe_gan train --timing 1\
    --d_solver=examples/vsp/$1_$2_d_solver.prototxt \
    --g_solver=examples/vsp/$1_$2_g_solver.prototxt \
    2>&1 | tee log/vsp_$1_$2/train.log