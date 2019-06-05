#ifndef CAFFE_GAN_SOLVER_HPP_
#define CAFFE_GAN_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype> class Solver;


/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on GAN%s.
 *
 * Consist of two solver for generator and discriminator individually.
 */
template <typename Dtype>
class GANSolver {
 public:
  explicit GANSolver(const SolverParameter& g_param, const SolverParameter& d_param);

  shared_ptr<caffe::Solver<Dtype> > getDiscriminatorSolver() {return d_solver;}
  shared_ptr<caffe::Solver<Dtype> > getGeneratorSolver() {return g_solver;}

  void SetActionFunction(ActionCallback func);

  void Restore(const char* resume_file) {
    // TODO
  }

  /// Atlantix: set the debug verbose level
  void set_debug(int val) {
    debug = val;
    d_solver->net_->set_debug(val);
    g_solver->net_->set_debug(val);
  }
  
  /// Atlantix: set timing flag
  void set_timing(int val) {
    timing = val;
    d_solver->net_->set_timing(val);
    g_solver->net_->set_timing(val);
  }

  void tile(const vector<cv::Mat> &src, cv::Mat &dst, int grid_x, int grid_y) {
    // patch size
    int width  = dst.cols/grid_x;
    int height = dst.rows/grid_y;
    // iterate through grid
    int k = 0;
    for(int i = 0; i < grid_y; i++) {
      for(int j = 0; j < grid_x; j++) {
        cv::Mat s = src[k++];
        cv::resize(s,s,cv::Size(width,height));
        s.copyTo(dst(cv::Rect(j*width,i*height,width,height)));
      }
    }
  }

  /// Show 3 channel or 1 channel blob; batch size >= 16; scale between (-1, 1)
  cv::Mat* blob2cvgrid(Blob<Dtype> *blob) {
    // LOG(INFO) << "Shape " << blob->shape_string();
    int width = blob->width(), height = blob->height(), channels = blob->channels();
    Dtype* input_data = blob->mutable_cpu_data();
    vector<cv::Mat> src;
    for (int i = 0; i < 16; i ++) {
      cv::Mat image;
      vector<cv::Mat> color_channel;
      for(int i = 0; i < channels; i ++) {
        cv::Mat _ch(height, width, CV_32FC1, input_data);
        color_channel.push_back(_ch);
        input_data += height * width;
      }
      cv::merge(color_channel, image);
      src.push_back(image);
    }
    cv::Mat *grid = new cv::Mat(height * 4, width * 4, (channels == 1 ? CV_32FC1: CV_32FC3));
    tile(src, *grid, 4, 4);

    //double min, max;
    //cv::minMaxLoc(*grid, &min, &max);
    //LOG(INFO) << "Min " <<  min << " Max " << max;

    *grid = (*grid + 1) * 127.5; 
    return grid;
  }

  void summary_time() {
    LOG(INFO) << "Summary timing:";
    LOG(INFO) << "Generator";
    vector<string>& g_names = g_solver->net_->layer_names_;
    for (int i = 0; i < g_names.size(); i ++) {
      LOG(INFO) << g_names[i] << "Forward:\t" << g_solver->net_->forward_time[i] / g_solver->net_->forward_count[i] << "\tBackward:\t" << g_solver->net_->backward_time[i] / g_solver->net_->backward_count[i];
    }
    LOG(INFO) << "Discriminator";
    vector<int>& d_counts = d_solver->net_->forward_count;
    vector<string>& d_names = d_solver->net_->layer_names_;
    for (int i = 0; i < d_names.size(); i ++) {
      LOG(INFO) << d_names[i] << "Forward:\t" << d_solver->net_->forward_time[i] / d_solver->net_->forward_count[i] << "\tBackward:\t" << d_solver->net_->backward_time[i] / d_solver->net_->backward_count[i];
    }
  }

  void TestAll() {
    if (timing > 0) {
      summary_time();
    }

    LOG(INFO) << "Save image";

    string name;
    // save input if pix2pix
    if (g_solver->net_->layers()[0]->type() != "RandVec") {
      // quite dirty, must be pix2pix here
      // data_A
      int g_input_layer = g_solver->net_->layerid_by_name("data_split");
      cv::Mat *x_input_grid = blob2cvgrid(g_solver->net_->top_vecs()[g_input_layer][0]);
      name = d_solver->param_.snapshot_prefix() + "x_input_" + caffe::format_int(iter_) + ".png";
      cv::imwrite(name.c_str(), *x_input_grid);
      delete x_input_grid;

      // data_B
      cv::Mat *x_target_grid = blob2cvgrid(g_solver->net_->top_vecs()[g_input_layer][1]);
      name = d_solver->param_.snapshot_prefix() + "x_target_" + caffe::format_int(iter_) + ".png";
      cv::imwrite(name.c_str(), *x_target_grid);
      delete x_target_grid;
    }

    int g_output_layer = g_solver->net_->layerid_by_name("output");
    // Must be float
    cv::Mat *x_fake_grid = blob2cvgrid(g_solver->net_->top_vecs()[g_output_layer][0]);
    
    name = d_solver->param_.snapshot_prefix() + "x_fake_" + caffe::format_int(iter_) + ".png";
    cv::imwrite(name.c_str(), *x_fake_grid);
    delete x_fake_grid;

    d_solver->net_->Forward();
    int ind = d_solver->net_->base_layer_index();

    cv::Mat *x_real_grid = blob2cvgrid(d_solver->net_->bottom_vecs()[ind][0]);
    name = d_solver->param_.snapshot_prefix() + "x_real_" + caffe::format_int(iter_) + ".png";
    cv::imwrite(name.c_str(), *x_real_grid);
    delete x_real_grid;
  }

  SolverAction::Enum GetRequestedAction();
  
  void Step(int iters);
  void Step_sw(int iters);

  virtual void Solve(const char* resume_file = NULL);

 private:
  int debug, timing;

  shared_ptr<caffe::Solver<Dtype> > g_solver, d_solver;

  int iter_;
  int current_step_;

  vector<Dtype> d_losses_, g_losses_;
  Dtype d_smoothed_loss_, g_smoothed_loss_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(GANSolver);
};

}  // namespace caffe

#endif  // CAFFE_GAN_SOLVER_HPP_
