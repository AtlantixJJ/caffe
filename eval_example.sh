build/tools/caffe_gan test \
    --g_model examples/cifar_gan/deconv_g.prototxt \
    --g_weights log/store/cifar_pool_deconv/pool_deconv_d_solver_iter_2000.caffemodel \
    --output log/eval/cifar_pool_deconv/