#ifndef CNN_CONV_H_
#define CNN_CONV_H_

#include "cnn/cnn.h"

namespace cnn {

struct AddVectorToAllColumns : public Node {
  explicit AddVectorToAllColumns(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

struct KMaxPooling : public Node {
  explicit KMaxPooling(const std::initializer_list<VariableIndex>& a, unsigned k = 1) : Node(a), k(k) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  size_t aux_storage_size() const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  unsigned k;
};

struct FoldRows : public Node {
  explicit FoldRows(const std::initializer_list<VariableIndex>& a, unsigned nrows) : Node(a), nrows(nrows) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
  unsigned nrows;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DNarrow : public Node {
  explicit Conv1DNarrow(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

// y = x_1 *conv x_2
// x_1 \in R^{d x s} (input)
// x_2 \in R^{d x m} (filter)
struct Conv1DWide : public Node {
  explicit Conv1DWide(const std::initializer_list<VariableIndex>& a) : Node(a) {}
  std::string as_string(const std::vector<std::string>& arg_names) const override;
  Dim dim_forward(const std::vector<Dim>& xs) const override;
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override;
  void backward_impl(const std::vector<const Tensor*>& xs,
                const Tensor& fx,
                const Tensor& dEdf,
                unsigned i,
                Tensor& dEdxi) const override;
};

} // namespace cnn

#endif
