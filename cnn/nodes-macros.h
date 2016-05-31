#ifndef CNN_NODE_MACROS_H_
#define CNN_NODE_MACROS_H_

// A macro to dispatch things to the appropriate device
#ifdef HAVE_CUDA
#define CNN_NODE_DEFINE_DEV_IMPL() \
  std::string as_string(const std::vector<std::string>& arg_names) const override; \
  Dim dim_forward(const std::vector<Dim>& xs) const override; \
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override { \
    assert(fx.device); \
    if(fx.device->type == DeviceType::CPU) { forward_dev_impl<cnn::Device_CPU>(*(cnn::Device_CPU*)fx.device,xs,fx); } \
    else if(fx.device->type == DeviceType::GPU) { forward_dev_impl<cnn::Device_GPU>(*(cnn::Device_GPU*)fx.device,xs,fx); } \
    else { abort(); } \
  } \
  template <class MyDevice> \
  void forward_dev_impl(const MyDevice & dev, const std::vector<const Tensor*>& xs, Tensor& fx) const; \
  void backward_impl(const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const override { \
    assert(fx.device); \
    if(fx.device->type == DeviceType::CPU) { backward_dev_impl<cnn::Device_CPU>(*(cnn::Device_CPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else if(fx.device->type == DeviceType::GPU) { backward_dev_impl<cnn::Device_GPU>(*(cnn::Device_GPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else { abort(); } \
  } \
  template <class MyDevice> \
  void backward_dev_impl( \
                const MyDevice & dev, \
                const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const;
#else
#define CNN_NODE_DEFINE_DEV_IMPL() \
  std::string as_string(const std::vector<std::string>& arg_names) const override; \
  Dim dim_forward(const std::vector<Dim>& xs) const override; \
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override { \
    assert(fx.device); \
    if(fx.device->type == DeviceType::CPU) { forward_dev_impl<cnn::Device_CPU>(*(cnn::Device_CPU*)fx.device,xs,fx); } \
    else { abort(); } \
  } \
  template <class MyDevice> \
  void forward_dev_impl(const MyDevice & dev, const std::vector<const Tensor*>& xs, Tensor& fx) const; \
  void backward_impl(const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const override { \
    assert(fx.device); \
    if(fx.device->type == DeviceType::CPU) { backward_dev_impl<cnn::Device_CPU>(*(cnn::Device_CPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else { abort(); } \
  } \
  template <class MyDevice> \
  void backward_dev_impl( \
                const MyDevice & dev, \
                const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const;
#endif

// A macro to instantiate templated device functions
// If the implementation is the same for both devices (using Eigen Tensors),
//  then this will instantiate both CPU and GPU implementations, and the
//  code can be the same.
// If the implementation is different for both devices, use #ifdef __CUDACC__
//  within the function, and create alternative code paths for CPU and GPU implementations
#ifdef __CUDACC__
#define CNN_NODE_INST_DEV_IMPL(MyNode) \
  template void MyNode::forward_dev_impl<Device_GPU>(const Device_GPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  template void MyNode::backward_dev_impl<Device_GPU>(const Device_GPU & dev, \
                                           const vector<const Tensor*>& xs, \
                                           const Tensor& fx, \
                                           const Tensor& dEdf, \
                                           unsigned i, \
                                           Tensor& dEdxi) const;
#else
#define CNN_NODE_INST_DEV_IMPL(MyNode) \
  template void MyNode::forward_dev_impl<Device_CPU>(const Device_CPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  template void MyNode::backward_dev_impl<Device_CPU>(const Device_CPU & dev, \
                                           const vector<const Tensor*>& xs, \
                                           const Tensor& fx, \
                                           const Tensor& dEdf, \
                                           unsigned i, \
                                           Tensor& dEdxi) const;
#endif

#endif
