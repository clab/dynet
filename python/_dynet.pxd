from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

ctypedef float real

cdef extern from "dynet/init.h" namespace "dynet":
    cdef void initialize(int& argc, char **& argv, unsigned random_seed)

cdef extern from "dynet/dim.h" namespace "dynet":
    cdef cppclass CDim "dynet::Dim":
        CDim() except +
        #CDim(int m) except +
        #CDim(int m, int n) except +
        CDim(vector[long]& ds) except +
        CDim(vector[long]& ds, unsigned int bs) except +
        #CDim(std::initializer_list[long] x) except +
        int size()
        unsigned int batch_elems()
        int sum_dims()
        CDim truncate()
        void resize(unsigned i)
        int ndims()
        int rows()
        int cols()
        void set(unsigned i, unsigned s)
        int size(unsigned i)
        CDim transpose()

cdef extern from "dynet/tensor.h" namespace "dynet":
    cdef cppclass CTensor "dynet::Tensor": 
        CDim d
        float* v
        pass
    float c_as_scalar "dynet::as_scalar" (CTensor& t)
    vector[float] c_as_vector "dynet::as_vector" (CTensor& t)

cdef extern from "dynet/model.h" namespace "dynet":
    cdef cppclass CParameterStorage "dynet::ParameterStorage":
        CParameterStorage()
        CTensor values
        CDim dim

    cdef cppclass CLookupParameterStorage "dynet::LookupParameterStorage":
        CLookupParameterStorage()
        vector[CTensor] values
        CDim dim

    cdef cppclass CParameters "dynet::Parameter":
        CParameters()
        CParameterStorage *get()
        void zero()
        void set_updated(bool b)
        bool is_updated()

    cdef cppclass CLookupParameters "dynet::LookupParameter":
        CLookupParameters()
        CLookupParameterStorage *get()
        CDim dim
        void initialize(unsigned index, const vector[float]& val)
        void zero()
        void set_updated(bool b)
        bool is_updated()

    cdef cppclass CParameterInit "dynet::ParameterInit":
        pass

    cdef cppclass CParameterInitNormal "dynet::ParameterInitNormal" (CParameterInit):
        CParameterInitNormal(float m, float v) # m = 0, v=1

    cdef cppclass CParameterInitUniform "dynet::ParameterInitUniform" (CParameterInit):
        CParameterInitUniform(float scale)

    cdef cppclass CParameterInitConst "dynet::ParameterInitConst" (CParameterInit):
        CParameterInitConst(float c)

    cdef cppclass CParameterInitIdentity "dynet::ParameterInitIdentity" (CParameterInit):
        CParameterInitIdentity()

    cdef cppclass CParameterInitGlorot "dynet::ParameterInitGlorot" (CParameterInit):
        CParameterInitGlorot(bool is_lookup) # is_lookup = False

    cdef cppclass CParameterInitSaxe "dynet::ParameterInitSaxe" (CParameterInit):
        ParameterInitSaxe()

    cdef cppclass CParameterInitFromFile "dynet::ParameterInitFromFile" (CParameterInit):
        CParameterInitFromFile(string filename)

    cdef cppclass CParameterInitFromVector "dynet::ParameterInitFromVector" (CParameterInit):
        CParameterInitFromVector(vector[float] void)

    cdef cppclass CModel "dynet::Model":
        CModel()
        #float gradient_l2_norm() const
        CParameters add_parameters(CDim& d)
        CParameters add_parameters(CDim& d, CParameterInit initializer)
        #CParameters add_parameters(CDim& d, CParameterInitNormal initializer)
        #CParameters add_parameters(CDim& d, CParameterInitUniform initializer)
        #CParameters add_parameters(CDim& d, CParameterInitConst initializer)
        CLookupParameters add_lookup_parameters(unsigned n, const CDim& d)
        CLookupParameters add_lookup_parameters(unsigned n, const CDim& d, CParameterInit initializer)
        vector[CParameterStorage] parameters_list()

    void load_dynet_model "dynet::load_dynet_model" (string filename, CModel *model)
    void save_dynet_model "dynet::save_dynet_model" (string filename, CModel *model)

cdef extern from "dynet/dynet.h" namespace "dynet":
    ctypedef unsigned VariableIndex

    cdef cppclass CComputationGraph "dynet::ComputationGraph":
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

cdef extern from "dynet/training.h" namespace "dynet":
    cdef cppclass CSimpleSGDTrainer "dynet::SimpleSGDTrainer":
        #CSimpleSGDTrainer(CModel& m, float lam, float e0)
        CSimpleSGDTrainer(CModel& m, float e0, float edecay) # TODO removed lam, update docs.
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CMomentumSGDTrainer "dynet::MomentumSGDTrainer":
        CMomentumSGDTrainer(CModel& m, float e0, float mom, float edecay) # TODO removed lam, update docs
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdagradTrainer "dynet::AdagradTrainer":
        CAdagradTrainer(CModel& m, float e0, float eps, float edecay) # TODO removed lam, update docs
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdadeltaTrainer "dynet::AdadeltaTrainer":
        CAdadeltaTrainer(CModel& m, float eps, float rho, float edecay) # TODO removed lam, update docs
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s)
        void update_epoch(float r)
        void status()

    cdef cppclass CAdamTrainer "dynet::AdamTrainer":
        CAdamTrainer(CModel& m, float alpha, float beta_1, float beta_2, float eps, float edecay) # TODO removed lam, update docs
        float clip_threshold
        bool clipping_enabled
        bool sparse_updates_enabled
        void update(float s)
        void update_epoch(float r)
        void status()


cdef extern from "dynet/expr.h" namespace "dynet::expr":
    cdef cppclass CExpression "dynet::expr::Expression":
        CExpression()
        CExpression(CComputationGraph *pg, VariableIndex i)
        CComputationGraph *pg
        long i
        CDim dim()
    #CExpression c_input "dynet::expr::input" (CComputationGraph& g, float s)   #
    CExpression c_input "dynet::expr::input" (CComputationGraph& g, float *ps) #
    CExpression c_input "dynet::expr::input" (CComputationGraph& g, CDim& d, vector[float]* pdata)
    CExpression c_parameter "dynet::expr::parameter" (CComputationGraph& g, CParameters p) #
    #CExpression c_lookup "dynet::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_lookup "dynet::expr::lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) #
    CExpression c_lookup "dynet::expr::lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) #
    #CExpression c_const_lookup "dynet::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_const_lookup "dynet::expr::const_lookup" (CComputationGraph& g, CLookupParameters p, unsigned* pindex) #
    CExpression c_const_lookup "dynet::expr::const_lookup" (CComputationGraph& g, CLookupParameters p, vector[unsigned]* pindices) #
    CExpression c_zeroes "dynet::expr::zeroes" (CComputationGraph& g, CDim& d) #
    CExpression c_random_normal "dynet::expr::random_normal" (CComputationGraph& g, CDim& d) #
    CExpression c_random_bernoulli "dynet::expr::random_bernoulli" (CComputationGraph& g, CDim& d, float p, float scale)
    CExpression c_random_uniform "dynet::expr::random_uniform" (CComputationGraph& g, CDim& d, float left, float right) #

    # identity function, but derivative is not propagated through it
    CExpression c_nobackprop "dynet::expr::nobackprop" (CExpression& x) #

    CExpression c_op_neg "dynet::expr::operator-" (CExpression& x) #
    CExpression c_op_add "dynet::expr::operator+" (CExpression& x, CExpression& y) #
    CExpression c_op_scalar_add "dynet::expr::operator+" (CExpression& x, float y) #
    CExpression c_op_mul "dynet::expr::operator*" (CExpression& x, CExpression& y) #
    CExpression c_op_scalar_mul "dynet::expr::operator*" (CExpression& x, float y) #
    CExpression c_op_scalar_div "dynet::expr::operator/" (CExpression& x, float y) #
    CExpression c_op_scalar_sub "dynet::expr::operator-" (float y, CExpression& x) #

    CExpression c_bmax "dynet::expr::max" (CExpression& x, CExpression& y) #
    CExpression c_bmin "dynet::expr::min" (CExpression& x, CExpression& y) #

    CExpression c_cdiv "dynet::expr::cdiv" (CExpression& x, CExpression& y) #
    CExpression c_cmult "dynet::expr::cmult" (CExpression& x, CExpression& y) #
    CExpression c_colwise_add "dynet::expr::colwise_add" (CExpression& x, CExpression& bias) #

    CExpression c_tanh "dynet::expr::tanh" (CExpression& x) #
    CExpression c_exp "dynet::expr::exp" (CExpression& x) #
    CExpression c_square "dynet::expr::square" (CExpression& x) #
    CExpression c_sqrt "dynet::expr::sqrt" (CExpression& x) #
    CExpression c_erf "dynet::expr::erf" (CExpression& x) #
    CExpression c_cube "dynet::expr::cube" (CExpression& x) #
    CExpression c_log "dynet::expr::log" (CExpression& x) #
    CExpression c_lgamma "dynet::expr::lgamma" (CExpression& x) #
    CExpression c_logistic "dynet::expr::logistic" (CExpression& x) #
    CExpression c_rectify "dynet::expr::rectify" (CExpression& x) #
    #CExpression c_hinge "dynet::expr::hinge" (CExpression& x, unsigned index, float m=?) #
    CExpression c_hinge "dynet::expr::hinge" (CExpression& x, unsigned* pindex, float m) #
    CExpression c_log_softmax "dynet::expr::log_softmax" (CExpression& x) #
    CExpression c_log_softmax "dynet::expr::log_softmax" (CExpression& x, vector[unsigned]& restriction) #?
    CExpression c_softmax "dynet::expr::softmax" (CExpression& x) #
    CExpression c_sparsemax "dynet::expr::sparsemax" (CExpression& x) #
    CExpression c_softsign "dynet::expr::softsign" (CExpression& x) #
    CExpression c_pow "dynet::expr::pow" (CExpression& x, CExpression& y) #
    CExpression c_bmin "dynet::expr::min" (CExpression& x, CExpression& y) #
    CExpression c_bmax "dynet::expr::max" (CExpression& x, CExpression& y) #
    CExpression c_noise "dynet::expr::noise" (CExpression& x, float stddev) #
    CExpression c_dropout "dynet::expr::dropout" (CExpression& x, float p) #
    CExpression c_block_dropout "dynet::expr::block_dropout" (CExpression& x, float p) #

    CExpression c_reshape "dynet::expr::reshape" (CExpression& x, CDim& d) #?
    CExpression c_transpose "dynet::expr::transpose" (CExpression& x) #

    CExpression c_affine_transform "dynet::expr::affine_transform" (const vector[CExpression]& xs)

    CExpression c_trace_of_product "dynet::expr::trace_of_product" (CExpression& x, CExpression& y);

    CExpression c_dot_product "dynet::expr::dot_product" (CExpression& x, CExpression& y) #
    CExpression c_squared_distance "dynet::expr::squared_distance" (CExpression& x, CExpression& y) #
    CExpression c_squared_norm "dynet::expr::squared_norm" (CExpression& x) #
    CExpression c_huber_distance "dynet::expr::huber_distance" (CExpression& x, CExpression& y, float c) #
    CExpression c_l1_distance "dynet::expr::l1_distance" (CExpression& x, CExpression& y) #
    CExpression c_binary_log_loss "dynet::expr::binary_log_loss" (CExpression& x, CExpression& y) #
    CExpression c_pairwise_rank_loss "dynet::expr::pairwise_rank_loss" (CExpression& x, CExpression& y, float m) #
    CExpression c_poisson_loss "dynet::expr::poisson_loss" (CExpression& x, unsigned y)

    CExpression c_conv1d_narrow "dynet::expr::conv1d_narrow" (CExpression& x, CExpression& f) #
    CExpression c_conv1d_wide "dynet::expr::conv1d_wide" (CExpression& x, CExpression& f) #
    CExpression c_filter1d_narrow "dynet::expr::filter1d_narrow" (CExpression& x, CExpression& f) #
    CExpression c_kmax_pooling "dynet::expr::kmax_pooling" (CExpression& x, unsigned k) #
    CExpression c_fold_rows "dynet::expr::fold_rows" (CExpression& x, unsigned nrows) #
    CExpression c_sum_cols "dynet::expr::sum_cols" (CExpression& x)               #
    CExpression c_kmh_ngram "dynet::expr::kmh_ngram" (CExpression& x, unsigned n) #

    CExpression c_sum_batches "dynet::expr::sum_batches" (CExpression& x)

    #CExpression c_pick "dynet::expr::pick" (CExpression& x, unsigned v)   #
    CExpression c_select_rows "dynet::expr::select_rows" (CExpression& x, vector[unsigned] rs)
    CExpression c_select_cols "dynet::expr::select_cols" (CExpression& x, vector[unsigned] cs)
    CExpression c_pick "dynet::expr::pick" (CExpression& x, unsigned* pv) #
    CExpression c_pick "dynet::expr::pick" (CExpression& x, vector[unsigned]* pv) #
    CExpression c_pickrange "dynet::expr::pickrange" (CExpression& x, unsigned v, unsigned u) #

    CExpression c_pickneglogsoftmax "dynet::expr::pickneglogsoftmax" (CExpression& x, unsigned v) #
    CExpression c_pickneglogsoftmax "dynet::expr::pickneglogsoftmax" (CExpression& x, vector[unsigned] vs) #

    # expecting a vector of CExpression
    CExpression c_average     "dynet::expr::average" (vector[CExpression]& xs)
    CExpression c_concat_cols "dynet::expr::concatenate_cols" (vector[CExpression]& xs)
    CExpression c_concat      "dynet::expr::concatenate" (vector[CExpression]& xs)

    CExpression c_sum            "dynet::expr::sum" (vector[CExpression]& xs)
    CExpression c_max            "dynet::expr::vmax" (vector[CExpression]& xs)
    CExpression c_logsumexp      "dynet::expr::logsumexp" (vector[CExpression]& xs)


#cdef extern from "dynet/model.h" namespace "dynet":
#    cdef cppclass Model:

cdef extern from "dynet/rnn.h" namespace "dynet":
    cdef cppclass CRNNPointer "dynet::RNNPointer":
        CRNNPointer()
        CRNNPointer(int i)

    cdef cppclass CRNNBuilder "dynet::RNNBuilder":
        void new_graph(CComputationGraph &cg)
        void start_new_sequence(vector[CExpression] ces)
        CExpression add_input(CExpression &x)
        CExpression add_input(CRNNPointer prev, CExpression &x)
        CExpression set_h(CRNNPointer prev, vector[CExpression] ces)
        CExpression set_s(CRNNPointer prev, vector[CExpression] ces)
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
cdef extern from "dynet/rnn.h" namespace "dynet":
    #cdef cppclass RNNBuilder "dynet::RNNBuilder":
    cdef cppclass CSimpleRNNBuilder  "dynet::SimpleRNNBuilder" (CRNNBuilder):
        CSimpleRNNBuilder()
        CSimpleRNNBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)
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

cdef extern from "dynet/gru.h" namespace "dynet":
    cdef cppclass CGRUBuilder "dynet::GRUBuilder" (CRNNBuilder):
        CGRUBuilder()
        CGRUBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)
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

cdef extern from "dynet/lstm.h" namespace "dynet":
    cdef cppclass CLSTMBuilder "dynet::LSTMBuilder" (CRNNBuilder):
        CLSTMBuilder()
        CLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)
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

    cdef cppclass CVanillaLSTMBuilder "dynet::VanillaLSTMBuilder" (CRNNBuilder):
        CVanillaLSTMBuilder()
        CVanillaLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel *model)

cdef extern from "dynet/fast-lstm.h" namespace "dynet":
    cdef cppclass CFastLSTMBuilder "dynet::FastLSTMBuilder" (CRNNBuilder):
        CFastLSTMBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, CModel &model)
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

cdef extern from "python/pybridge.h" namespace "pydynet":
    cdef cppclass CModelSaver "pydynet::ModelSaver":
        CModelSaver(string filename, CModel *model)
        CModelSaver add_parameter(CParameters p)
        CModelSaver add_lookup_parameter(CLookupParameters lp)
        CModelSaver add_gru_builder(CGRUBuilder b)
        CModelSaver add_lstm_builder(CLSTMBuilder b)
        CModelSaver add_vanilla_lstm_builder(CVanillaLSTMBuilder b)
        CModelSaver add_srnn_builder(CSimpleRNNBuilder b)
        void done()

    cdef cppclass CModelLoader "pydynet::ModelLoader":
        CModelLoader(string filename, CModel *model)
        CModelSaver fill_parameter(CParameters p)
        CModelSaver fill_lookup_parameter(CLookupParameters lp)
        CModelSaver fill_gru_builder(CGRUBuilder lp)
        CModelSaver fill_lstm_builder(CLSTMBuilder lp)
        CModelSaver fill_vanilla_lstm_builder(CVanillaLSTMBuilder lp)
        CModelSaver fill_srnn_builder(CSimpleRNNBuilder lp)
        void done()
