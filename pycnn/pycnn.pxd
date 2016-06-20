from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef float real

cdef extern from "cnn/init.h" namespace "cnn":
    cdef void Initialize(int& argc, char **& argv, unsigned random_seed)

cdef extern from "cnn/dim.h" namespace "cnn":
    cdef cppclass CDim "cnn::Dim":
        CDim() except +
        #CDim(int m) except +
        #CDim(int m, int n) except +
        CDim(vector[long]& ds) except +
        #CDim(std::initializer_list[long] x) except +
        int size()
        int sum_dims()
        CDim truncate()
        void resize(unsigned i)
        int ndims()
        int rows()
        int cols()
        void set(unsigned i, unsigned s)
        int size(unsigned i)
        CDim transpose()

cdef extern from "cnn/tensor.h" namespace "cnn":
    cdef cppclass CTensor "cnn::Tensor": 
        CDim d
        float* v
        pass
    float c_as_scalar "cnn::as_scalar" (CTensor& t)
    vector[float] c_as_vector "cnn::as_vector" (CTensor& t)

cdef extern from "cnn/model.h" namespace "cnn":
    cdef cppclass CParameters "cnn::Parameters":
        CParameters()
        CTensor values
        #void scale_parameters(float a) # override;
        #void squared_l2norm(float* sqnorm) const # override;
        #void g_squared_l2norm(float* sqnorm) const # override;
        #size_t size() const # override;
        #void accumulate_grad(const Tensor& g)
        #void clear()
        CDim dim
        pass

    cdef cppclass CLookupParameters "cnn::LookupParameters":
        CLookupParameters()
        vector[CTensor] values
        #void scale_parameters(float a) # override;
        #void squared_l2norm(float* sqnorm) const # override;
        #void g_squared_l2norm(float* sqnorm) const # override;
        #size_t size() const # override;
        #void accumulate_grad(const Tensor& g)
        #void clear()
        CDim dim
        void Initialize(unsigned index, const vector[float]& val)
        pass

    cdef cppclass CModel "cnn::Model":
        CModel()
        #float gradient_l2_norm() const
        CParameters* add_parameters(CDim& d, float scale = 0.0)
        CLookupParameters* add_lookup_parameters(unsigned n, const CDim& d)
        #void save(string fname)
        #void load(string fname)

    void load_cnn_model "cnn::load_cnn_model" (string filename, CModel *model)
    void save_cnn_model "cnn::save_cnn_model" (string filename, CModel *model)

cdef extern from "cnn/cnn.h" namespace "cnn":
    ctypedef unsigned VariableIndex
    cdef cppclass CComputationGraph "cnn::ComputationGraph":
        CComputationGraph() except +
        # Inputs
        VariableIndex add_input(real s)
        VariableIndex add_input(const real* ps)
        VariableIndex add_input(const CDim& d, const vector[float]* pdata)

        # Parameters
        VariableIndex add_parameters(CParameters* p)
        VariableIndex add_lookup(CLookupParameters* p, const unsigned* pindex)
        VariableIndex add_lookup(CLookupParameters* p, unsigned index)
        VariableIndex add_const_lookup(CLookupParameters* p, const unsigned* pindex)
        VariableIndex add_const_lookup(CLookupParameters* p, unsigned index)
        
        const CTensor& forward()
        const CTensor& incremental_forward()
        const CTensor& get_value(VariableIndex i)
        void invalidate()
        void backward()
        void backward(VariableIndex i)

        void PrintGraphviz() const

cdef extern from "cnn/training.h" namespace "cnn":
    cdef cppclass CSimpleSGDTrainer "cnn::SimpleSGDTrainer":
        CSimpleSGDTrainer(CModel* m, float lam, float e0)
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CMomentumSGDTrainer "cnn::MomentumSGDTrainer":
        CMomentumSGDTrainer(CModel* m, float lam, float e0, float mom)
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdagradTrainer "cnn::AdagradTrainer":
        CAdagradTrainer(CModel* m, float lam, float e0, float eps)
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdadeltaTrainer "cnn::AdadeltaTrainer":
        CAdadeltaTrainer(CModel* m, float lam, float eps, float rho)
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdamTrainer "cnn::AdamTrainer":
        CAdamTrainer(CModel* m, float lam, float alpha, float beta_1, float beta_2, float eps)
        void update(float s)
        void update_epoch(float r)
        void status()


cdef extern from "cnn/expr.h" namespace "cnn::expr":
    cdef cppclass CExpression "cnn::expr::Expression":
        CExpression()
        CExpression(CComputationGraph *pg, VariableIndex i)
        CComputationGraph *pg
        long i
    #CExpression c_input "cnn::expr::input" (CComputationGraph& g, float s)   #
    CExpression c_input "cnn::expr::input" (CComputationGraph& g, float *ps) #
    CExpression c_input "cnn::expr::input" (CComputationGraph& g, CDim& d, vector[float]* pdata)
    CExpression c_parameter "cnn::expr::parameter" (CComputationGraph& g, CParameters* p) #
    #CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned* pindex) #
    CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, vector[unsigned]* pindices) #
    #CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned* pindex) #
    CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, vector[unsigned]* pindices) #

    # identity function, but derivative is not propagated through it
    CExpression c_nobackprop "cnn::expr::nobackprop" (CExpression& x) #

    CExpression c_op_neg "cnn::expr::operator-" (CExpression& x) #
    CExpression c_op_add "cnn::expr::operator+" (CExpression& x, CExpression& y) #
    CExpression c_op_scalar_add "cnn::expr::operator+" (CExpression& x, float y) #
    CExpression c_op_mul "cnn::expr::operator*" (CExpression& x, CExpression& y) #
    CExpression c_op_scalar_mul "cnn::expr::operator*" (CExpression& x, float y) #
    CExpression c_op_scalar_div "cnn::expr::operator/" (CExpression& x, float y) #
    CExpression c_op_scalar_sub "cnn::expr::operator-" (float y, CExpression& x) #

    CExpression c_bmax "cnn::expr::max" (CExpression& x, CExpression& y) #
    CExpression c_bmin "cnn::expr::min" (CExpression& x, CExpression& y) #

    CExpression c_cdiv "cnn::expr::cdiv" (CExpression& x, CExpression& y) #
    CExpression c_colwise_add "cnn::expr::colwise_add" (CExpression& x, CExpression& bias) #

    CExpression c_tanh "cnn::expr::tanh" (CExpression& x) #
    CExpression c_exp "cnn::expr::exp" (CExpression& x) #
    CExpression c_square "cnn::expr::square" (CExpression& x) #
    CExpression c_cube "cnn::expr::cube" (CExpression& x) #
    CExpression c_log "cnn::expr::log" (CExpression& x) #
    CExpression c_logistic "cnn::expr::logistic" (CExpression& x) #
    CExpression c_rectify "cnn::expr::rectify" (CExpression& x) #
    #CExpression c_hinge "cnn::expr::hinge" (CExpression& x, unsigned index, float m=?) #
    CExpression c_hinge "cnn::expr::hinge" (CExpression& x, unsigned* pindex, float m) #
    CExpression c_log_softmax "cnn::expr::log_softmax" (CExpression& x) #
    CExpression c_log_softmax "cnn::expr::log_softmax" (CExpression& x, vector[unsigned]& restriction) #?
    CExpression c_softmax "cnn::expr::softmax" (CExpression& x) #
    CExpression c_softsign "cnn::expr::softsign" (CExpression& x) #
    CExpression c_bmin "cnn::expr::min" (CExpression& x, CExpression& y) #
    CExpression c_bmax "cnn::expr::max" (CExpression& x, CExpression& y) #
    CExpression c_noise "cnn::expr::noise" (CExpression& x, float stddev) #
    CExpression c_dropout "cnn::expr::dropout" (CExpression& x, float p) #
    CExpression c_block_dropout "cnn::expr::block_dropout" (CExpression& x, float p) #

    CExpression c_reshape "cnn::expr::reshape" (CExpression& x, CDim& d) #?
    CExpression c_transpose "cnn::expr::transpose" (CExpression& x) #

    CExpression c_affine_transform "cnn::expr::affine_transform" (const vector[CExpression]& xs)

    CExpression c_trace_of_product "cnn::expr::trace_of_product" (CExpression& x, CExpression& y);
    CExpression c_cwise_multiply "cnn::expr::cwise_multiply" (CExpression& x, CExpression& y) #

    CExpression c_dot_product "cnn::expr::dot_product" (CExpression& x, CExpression& y) #
    CExpression c_squared_distance "cnn::expr::squared_distance" (CExpression& x, CExpression& y) #
    CExpression c_huber_distance "cnn::expr::huber_distance" (CExpression& x, CExpression& y, float c) #
    CExpression c_l1_distance "cnn::expr::l1_distance" (CExpression& x, CExpression& y) #
    CExpression c_binary_log_loss "cnn::expr::binary_log_loss" (CExpression& x, CExpression& y) #
    CExpression c_pairwise_rank_loss "cnn::expr::pairwise_rank_loss" (CExpression& x, CExpression& y, float m) #
    CExpression c_poisson_loss "cnn::expr::poisson_loss" (CExpression& x, unsigned y)

    CExpression c_conv1d_narrow "cnn::expr::conv1d_narrow" (CExpression& x, CExpression& f) #
    CExpression c_conv1d_wide "cnn::expr::conv1d_wide" (CExpression& x, CExpression& f) #
    CExpression c_kmax_pooling "cnn::expr::kmax_pooling" (CExpression& x, unsigned k) #
    CExpression c_fold_rows "cnn::expr::fold_rows" (CExpression& x, unsigned nrows) #
    CExpression c_sum_cols "cnn::expr::sum_cols" (CExpression& x)               #
    CExpression c_kmh_ngram "cnn::expr::kmh_ngram" (CExpression& x, unsigned n) #

    CExpression c_sum_batches "cnn::expr::sum_batches" (CExpression& x)

    #CExpression c_pick "cnn::expr::pick" (CExpression& x, unsigned v)   #
    CExpression c_pick "cnn::expr::pick" (CExpression& x, unsigned* pv) #
    CExpression c_pick "cnn::expr::pick" (CExpression& x, vector[unsigned]* pv) #
    CExpression c_pickrange "cnn::expr::pickrange" (CExpression& x, unsigned v, unsigned u) #

    CExpression c_pickneglogsoftmax "cnn::expr::pickneglogsoftmax" (CExpression& x, unsigned v) #
    CExpression c_pickneglogsoftmax "cnn::expr::pickneglogsoftmax" (CExpression& x, vector[unsigned] vs) #

    # expecting a vector of CExpression
    CExpression c_average     "cnn::expr::average" (vector[CExpression]& xs)
    CExpression c_concat_cols "cnn::expr::concatenate_cols" (vector[CExpression]& xs)
    CExpression c_concat      "cnn::expr::concatenate" (vector[CExpression]& xs)

    CExpression c_sum      "cnn::expr::sum" (vector[CExpression]& xs)
    CExpression c_max      "cnn::expr::vmax" (vector[CExpression]& xs)


#cdef extern from "cnn/model.h" namespace "cnn":
#    cdef cppclass Model:

cdef extern from "cnn/rnn.h" namespace "cnn":
    cdef cppclass CRNNPointer "cnn::RNNPointer":
        CRNNPointer()
        CRNNPointer(int i)

    cdef cppclass CRNNBuilder "cnn::RNNBuilder":
        void new_graph(CComputationGraph &cg)
        void start_new_sequence(vector[CExpression] ces)
        CExpression add_input(CExpression &x)
        CExpression add_input(CRNNPointer prev, CExpression &x)
        void rewind_one_step()
        CExpression back()
        vector[CExpression] final_h()
        vector[CExpression] final_s()
        vector[CExpression] get_h(CRNNPointer i)
        vector[CExpression] get_s(CRNNPointer i)
        CRNNPointer state()

# TODO unify with LSTMBuilder using inheritance
cdef extern from "cnn/rnn.h" namespace "cnn":
    #cdef cppclass RNNBuilder "cnn::RNNBuilder":
    cdef cppclass CSimpleRNNBuilder  "cnn::SimpleRNNBuilder" (CRNNBuilder):
        CSimpleRNNBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel *model)
        #void new_graph(CComputationGraph &cg)
        #void start_new_sequence(vector[CExpression] ces)
        #CExpression add_input(CExpression &x)
        #CExpression add_input(CRNNPointer prev, CExpression &x)
        #void rewind_one_step()
        #CExpression back()
        #vector[CExpression] final_h()
        #vector[CExpression] final_s()
        #vector[CExpression] get_h(CRNNPointer i)
        #vector[CExpression] get_s(CRNNPointer i)
        #CRNNPointer state()

cdef extern from "cnn/lstm.h" namespace "cnn":
    cdef cppclass CLSTMBuilder "cnn::LSTMBuilder" (CRNNBuilder):
        CLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel *model)
        #void new_graph(CComputationGraph &cg)
        #void start_new_sequence(vector[CExpression] ces)
        #CExpression add_input(CExpression &x)
        #CExpression add_input(CRNNPointer prev, CExpression &x)
        #void rewind_one_step()
        #CExpression back()
        #vector[CExpression] final_h()
        #vector[CExpression] final_s()
        #vector[CExpression] get_h(CRNNPointer i)
        #vector[CExpression] get_s(CRNNPointer i)
        #CRNNPointer state()

cdef extern from "cnn/fast-lstm.h" namespace "cnn":
    cdef cppclass CFastLSTMBuilder "cnn::FastLSTMBuilder" (CRNNBuilder):
        CFastLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel *model)
        #void new_graph(CComputationGraph &cg)
        #void start_new_sequence(vector[CExpression] ces)
        #CExpression add_input(CExpression &x)
        #CExpression add_input(CRNNPointer prev, CExpression &x)
        #void rewind_one_step()
        #CExpression back()
        #vector[CExpression] final_h()
        #vector[CExpression] final_s()
        #vector[CExpression] get_h(CRNNPointer i)
        #vector[CExpression] get_s(CRNNPointer i)
        #CRNNPointer state()

