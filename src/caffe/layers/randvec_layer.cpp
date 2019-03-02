#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/randvec_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX_RANDOM 10000

namespace caffe {

template <typename Dtype>
Dtype RandVecLayer<Dtype>::GetRandom(const Dtype lower, const Dtype upper) {
    CHECK(data_rng_);
    CHECK_LT(lower, upper) << "Upper bound must be greater than lower bound!";
    caffe::rng_t* data_rng =
        static_cast<caffe::rng_t*>(data_rng_->generator());
    return static_cast<Dtype>(((*data_rng)()) % static_cast<unsigned int>(
        (upper - lower) * MAX_RANDOM)) / static_cast<Dtype>(MAX_RANDOM)+lower;
}

template <typename Dtype>
void RandVecLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    RandVecParameter randvec_param = this->layer_param_.randvec_param();
    batch_size_ = randvec_param.batch_size();
    dim_ = randvec_param.dim();
    height_ = randvec_param.height();
    width_ = randvec_param.width();
    lower_ = randvec_param.lower();
    upper_ = randvec_param.upper();
    iter_idx_ = 1;
    vector<int> top_shape(2);
    top_shape[0] = batch_size_;
    top_shape[1] = dim_;
    if (height_ >0 && width_>0) {
        top_shape.resize(4);
        top_shape[0] = batch_size_;
        top_shape[1] = dim_;
        top_shape[2] = height_;
        top_shape[3] = width_;
    }
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RandVecLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    caffe_rng_gaussian<Dtype>(top[0]->count(), 0.0, 2.0, top[0]->mutable_cpu_data());
}

template <typename Dtype>
void RandVecLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /*
    if (!propagate_down[0]) return;
    LOG(INFO) << "RandVec backward";
    LOG(INFO) << "top shape: " << top[0]->shape_string();
    LOG(INFO) << "bottom shape: " << bottom[0]->shape_string();
    LOG(INFO) << "Read mutable";
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    LOG(INFO) << "Read mutable done";

    const Dtype* top_diff = top[0]->cpu_diff();
    for (int i = 0; i < top[0]->count(); i ++) {
        bottom_diff[i] = top_diff[i];
    }
    */
    //LOG(INFO) << "RandVec backward done";
}

INSTANTIATE_CLASS(RandVecLayer);
REGISTER_LAYER_CLASS(RandVec);

}  // namespace caffe