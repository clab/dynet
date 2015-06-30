from libcpp.vector cimport vector

ctypedef float real

cdef extern from "cnn/init.h" namespace "cnn":
    cdef void Initialize(int& argc, char **& argv)

cdef extern from "cnn/dim.h" namespace "cnn":
    cdef cppclass CDim "cnn::Dim":
        CDim() except +
        CDim(int m) except +
        CDim(int m, int n) except +
        #CDim(std::initializer_list[long] x) except +
        int size()
        int sum_dims()
        int ndims()
        int rows()
        int cols()
        int size(unsigned i)
        CDim transpose()

cdef extern from "cnn/tensor.h" namespace "cnn":
    cdef cppclass CTensor "cnn::Tensor": 
        pass
    float c_as_scalar "cnn::as_scalar" (CTensor& t)
    vector[float] c_as_vector "cnn::as_vector" (CTensor& t)

cdef extern from "cnn/model.h" namespace "cnn":
    cdef cppclass CParameters "cnn::Parameters":
        CParameters()
        #void scale_parameters(float a) # override;
        #void squared_l2norm(float* sqnorm) const # override;
        #void g_squared_l2norm(float* sqnorm) const # override;
        #size_t size() const # override;
        #void accumulate_grad(const Tensor& g)
        #void clear()
        pass

    cdef cppclass CLookupParameters "cnn::LookupParameters":
        CLookupParameters()
        #void scale_parameters(float a) # override;
        #void squared_l2norm(float* sqnorm) const # override;
        #void g_squared_l2norm(float* sqnorm) const # override;
        #size_t size() const # override;
        #void accumulate_grad(const Tensor& g)
        #void clear()
        #void Initialize(unsigned index, const vector[float]& val)
        pass

    cdef cppclass CModel "cnn::Model":
        CModel()
        #float gradient_l2_norm() const
        CParameters* add_parameters(CDim& d, float scale = 0.0)
        CLookupParameters* add_lookup_parameters(unsigned n, const CDim& d)

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
        #const CTensor& get_value(VariableIndex i)
        void backward()

        void PrintGraphviz() const

cdef extern from "cnn/training.h" namespace "cnn":
    cdef cppclass CSimpleSGDTrainer "cnn::SimpleSGDTrainer":
        CSimpleSGDTrainer(CModel* m, float lam, float e0)
        void update(float s)
        void update_epoch(float r)


cdef extern from "cnn/expr.h" namespace "cnn::expr":
    cdef cppclass CExpression "cnn::expr::Expression":
        CExpression()
        CExpression(CComputationGraph *pg, long i)
        CComputationGraph *pg
        long i
    #CExpression c_input "cnn::expr::input" (CComputationGraph& g, float s)   #
    CExpression c_input "cnn::expr::input" (CComputationGraph& g, float *ps) #
    CExpression c_input "cnn::expr::input" (CComputationGraph& g, CDim& d, vector[float]* pdata)
    CExpression c_parameter "cnn::expr::parameter" (CComputationGraph& g, CParameters* p) #
    #CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_lookup "cnn::expr::lookup" (CComputationGraph& g, CLookupParameters* p, unsigned* pindex) #
    #CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned index)   #
    CExpression c_const_lookup "cnn::expr::const_lookup" (CComputationGraph& g, CLookupParameters* p, unsigned* pindex) #

    CExpression c_op_neg "cnn::expr::operator-" (CExpression& x) #
    CExpression c_op_add "cnn::expr::operator+" (CExpression& x, CExpression& y) #
    CExpression c_op_mul "cnn::expr::operator*" (CExpression& x, CExpression& y) #
    CExpression c_op_scalar_mul "cnn::expr::operator*" (CExpression& x, float y) #
    CExpression c_op_scalar_sub "cnn::expr::operator-" (float y, CExpression& x) #

    CExpression c_cdiv "cnn::expr::cdiv" (CExpression& x, CExpression& y) #
    CExpression c_colwise_add "cnn::expr::colwise_add" (CExpression& x, CExpression& bias) #

    CExpression c_tanh "cnn::expr::tanh" (CExpression& x) #
    CExpression c_exp "cnn::expr::exp" (CExpression& x) #
    CExpression c_log "cnn::expr::log" (CExpression& x) #
    CExpression c_logistic "cnn::expr::logistic" (CExpression& x) #
    CExpression c_rectify "cnn::expr::rectify" (CExpression& x) #
    #CExpression c_hinge "cnn::expr::hinge" (CExpression& x, unsigned index, float m=?) #
    CExpression c_hinge "cnn::expr::hinge" (CExpression& x, unsigned* pindex, float m) #
    CExpression c_log_softmax "cnn::expr::log_softmax" (CExpression& x) #
    CExpression c_log_softmax "cnn::expr::log_softmax" (CExpression& x, vector[unsigned]& restriction) #?
    CExpression c_softmax "cnn::expr::softmax" (CExpression& x) #
    CExpression c_softsign "cnn::expr::softsign" (CExpression& x) #
    CExpression c_noise "cnn::expr::noise" (CExpression& x, float stddev) #
    CExpression c_dropout "cnn::expr::dropout" (CExpression& x, float p) #

    CExpression c_reshape "cnn::expr::reshape" (CExpression& x, CDim& d) #?
    CExpression c_transpose "cnn::expr::transpose" (CExpression& x) #

    #CExpression c_affine_transform "cnn::expr::affine_transform" (const std::initializer_list<Expression>& xs)
    CExpression c_cwise_multiply "cnn::expr::cwise_multiply" (CExpression& x, CExpression& y) #

    CExpression c_dot_product "cnn::expr::dot_product" (CExpression& x, CExpression& y) #
    CExpression c_squared_distance "cnn::expr::squared_distance" (CExpression& x, CExpression& y) #
    CExpression c_huber_distance "cnn::expr::huber_distance" (CExpression& x, CExpression& y, float c) #
    CExpression c_l1_distance "cnn::expr::l1_distance" (CExpression& x, CExpression& y) #
    CExpression c_binary_log_loss "cnn::expr::binary_log_loss" (CExpression& x, CExpression& y) #
    CExpression c_pairwise_rank_loss "cnn::expr::pairwise_rank_loss" (CExpression& x, CExpression& y, float m) #

    CExpression c_conv1d_narrow "cnn::expr::conv1d_narrow" (CExpression& x, CExpression& f) #
    CExpression c_conv1d_wide "cnn::expr::conv1d_wide" (CExpression& x, CExpression& f) #
    CExpression c_kmax_pooling "cnn::expr::kmax_pooling" (CExpression& x, unsigned k) #
    CExpression c_fold_rows "cnn::expr::fold_rows" (CExpression& x, unsigned nrows) #

    #CExpression c_pick "cnn::expr::pick" (CExpression& x, unsigned v)   #
    CExpression c_pick "cnn::expr::pick" (CExpression& x, unsigned* pv) #
    CExpression c_pickrange "cnn::expr::pickrange" (CExpression& x, unsigned v, unsigned u) #

    CExpression c_pickneglogsoftmax "cnn::expr::pickneglogsoftmax" (CExpression& x, unsigned v) #

    CExpression c_sum_cols "cnn::expr::sum_cols" (CExpression& x)               #
    CExpression c_kmh_ngram "cnn::expr::kmh_ngram" (CExpression& x, unsigned n) #

#cdef extern from "cnn/model.h" namespace "cnn":
#    cdef cppclass Model:



