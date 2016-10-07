from libcpp.vector cimport vector
from libcpp.string cimport string

ctypedef float real

cdef extern from "cnn/init.h" namespace "cnn":
    cdef void initialize(int& argc, char **& argv, unsigned random_seed)

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
    cdef cppclass CParameterStorage "cnn::ParameterStorage":
        CParameterStorage()
        CTensor values
        CDim dim

    cdef cppclass CLookupParameterStorage "cnn::LookupParameterStorage":
        CLookupParameterStorage()
        vector[CTensor] values
        CDim dim

    cdef cppclass CParameters "cnn::Parameter":
        CParameters()
        CParameterStorage *get()
        void zero()

    cdef cppclass CLookupParameters "cnn::LookupParameter":
        CLookupParameters()
        CLookupParameterStorage *get()
        CDim dim
        void initialize(unsigned index, const vector[float]& val)
        void zero()

    cdef cppclass CModel "cnn::Model":
        CModel()
        #float gradient_l2_norm() const
        CParameters add_parameters(CDim& d, float scale = 0.0)
        CLookupParameters add_lookup_parameters(unsigned n, const CDim& d)
        vector[CParameterStorage] parameters_list()

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
        
        const CTensor& forward(VariableIndex index)
        const CTensor& incremental_forward(VariableIndex index)
        const CTensor& get_value(VariableIndex i)
        void invalidate()
        void backward(VariableIndex i)

        # checkpointing
        void checkpoint()
        void revert()

        void print_graphviz() const

cdef extern from "cnn/training.h" namespace "cnn":
    cdef cppclass CSimpleSGDTrainer "cnn::SimpleSGDTrainer":
        #CSimpleSGDTrainer(CModel* m, float lam, float e0)
        CSimpleSGDTrainer(CModel* m, float e0) # TODO removed lam, update docs.
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CMomentumSGDTrainer "cnn::MomentumSGDTrainer":
        CMomentumSGDTrainer(CModel* m, float e0, float mom) # TODO removed lam, update docs
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdagradTrainer "cnn::AdagradTrainer":
        CAdagradTrainer(CModel* m, float e0, float eps) # TODO removed lam, update docs

        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdadeltaTrainer "cnn::AdadeltaTrainer":
        CAdadeltaTrainer(CModel* m, float eps, float rho) # TODO removed lam, update docs

        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdamTrainer "cnn::AdamTrainer":
        CAdamTrainer(CModel* m, float alpha, float beta_1, float beta_2, float eps) # TODO removed lam, update docs

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
    CExpression c_parameter "cnn::expr::parameter" (CComputationGraph& g, CParameters p) #
    #CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) #
    CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) #
    #CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) #
    CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) #

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
    CExpression c_filter1d_narrow "cnn::expr::filter1d_narrow" (CExpression& x, CExpression& f) #
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
        void set_dropout(float f)
        void disable_dropout()

# TODO unify with LSTMBuilder using inheritance
cdef extern from "cnn/rnn.h" namespace "cnn":
    #cdef cppclass RNNBuilder "cnn::RNNBuilder":
    cdef cppclass CSimpleRNNBuilder  "cnn::SimpleRNNBuilder" (CRNNBuilder):
        CSimpleRNNBuilder()
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

cdef extern from "cnn/gru.h" namespace "cnn":
    cdef cppclass CGRUBuilder "cnn::GRUBuilder" (CRNNBuilder):
        CGRUBuilder()
        CGRUBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel *model)
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
        CLSTMBuilder()
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

cdef extern from "pybridge.h" namespace "pycnn":
    cdef cppclass CModelSaver "pycnn::ModelSaver":
        CModelSaver(string filename, CModel *model)
        CModelSaver add_parameter(CParameters p)
        CModelSaver add_lookup_parameter(CLookupParameters lp)
        CModelSaver add_gru_builder(CGRUBuilder b)
        CModelSaver add_lstm_builder(CLSTMBuilder b)
        CModelSaver add_srnn_builder(CSimpleRNNBuilder b)
        void done()

    cdef cppclass CModelLoader "pycnn::ModelLoader":
        CModelLoader(string filename, CModel *model)
        CModelSaver fill_parameter(CParameters p)
        CModelSaver fill_lookup_parameter(CLookupParameters lp)
        CModelSaver fill_gru_builder(CGRUBuilder lp)
        CModelSaver fill_lstm_builder(CLSTMBuilder lp)
        CModelSaver fill_srnn_builder(CSimpleRNNBuilder lp)
        void done()
