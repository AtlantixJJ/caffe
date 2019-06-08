mkdir $4
echo CUDA_VISIBLE_DEVICES=$1 build/tools/caffe_gan test \
    --gpu 0 \
    --iterations 1000 \
    --g_model $2 \
    --g_weights $3 \
    --output $4
CUDA_VISIBLE_DEVICES=$1 build/tools/caffe_gan test \
    --gpu 0 \
    --iterations 1000 \
    --g_model $2 \
    --g_weights $3 \
    --output $4