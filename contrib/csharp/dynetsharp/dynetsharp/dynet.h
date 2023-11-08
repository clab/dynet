#pragma once

#include "dynet/except.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/training.h"
#include "dynet/init.h"
#include "dynet/param-init.h"
#include "dynet/rnn.h"
#include "dynet/lstm.h"
#include "dynet/gru.h"
#include "dynet/fast-lstm.h"
#include "dynet/model.h"
#include "dynet/grad-check.h"
#include "dynet/globals.h"
#include "dynet/tensor.h"
#include "dynet/mem.h"
#include "dynet/devices.h"
#include "dynet/device-structs.h"

//using namespace std;
using namespace dynet;
using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;

#define ExceptionWrap(msg) \
	try { msg } \
	catch (out_of_memory &ex) { \
		throw gcnew OutOfMemoryException(gcnew String(ex.what())); \
	} \
	catch (cuda_not_implemented &ex) { \
		throw gcnew NotImplementedException(gcnew String(ex.what())); \
	} \
	catch (std::invalid_argument &ex) { \
		throw gcnew ArgumentException(gcnew String(ex.what())); \
	} \
	catch (std::runtime_error &ex) { \
		throw gcnew SystemException(gcnew String(ex.what())); \
	} \
	catch (std::exception &ex) { \
		throw gcnew Exception(gcnew String(ex.what())); \
	};/* \
	catch (...) { \
		throw gcnew Exception("Unknown Exception"); \
	};*/

namespace dynetsharp {
	bool fInitialized = false;
	void CheckForInitialized() {
		if (!fInitialized)
			throw gcnew SystemException("DynetParams.Initialize() has not been called yet. Please do so before using any Dynet functions.");
	}

	static ComputationGraph *cg = new ComputationGraph();
	static int _cg_version = 0;
	static std::vector<float *> _floatInputs;
	static std::vector<std::vector<float> *> _vecInputs;
	dynet::Device *str2dev(String ^name);
	// For clearing out memory
	static size_t maxOverallMemory = 0;
	static size_t initialMemorySize = 0;
	// Extra functions
	dynet::Dim ConvertArrToDim(array<long>^ arr);
	array<long> ^ConvertDimToArr(dynet::Dim d);
	std::vector<unsigned int> VecToUInt(std::vector<int> vec);
	template<typename T> std::vector<T> ConvertArrayToVector(array<T> ^arr);
	template<typename T> array<T> ^ConvertVectorToArray(std::vector<T> vec);
	void ResetDynetMemory(size_t newMemSize);
	
	public ref class Dim {
	private:
		dynet::Dim *_dim;
	internal:
		Dim(dynet::Dim dim) {
			_dim = new dynet::Dim(dim);
		}
		Dim(std::vector<long> dims, int batch_size) {
			_dim = new dynet::Dim(dims, batch_size);
		}
		dynet::Dim get_cdim() {
			return *_dim;
		}
	public:
		Dim(array<long> ^dims) {
			_dim = new dynet::Dim(ConvertArrayToVector<long>(dims));
		}
		Dim(array<long> ^dims, int batchSize) {
			_dim = new dynet::Dim(ConvertArrayToVector<long>(dims), batchSize);
		}
		~Dim() {
			this->!Dim();
		}
		!Dim() {
			if (_dim != NULL) 
				delete _dim;
			_dim = NULL;
		}
		property int NDims {
			int get() {
				return (int)_dim->ndims();
			}
		}
		property int BatchSize {
			int get() {
				return _dim->batch_elems();
			}
		}
		/// <summary>
		/// <para>Returns the list of dimensions sizes, batch size is the last dimension</para>
		/// </summary>
		property array<long> ^Dims {
			array<long> ^get() {
				array<long> ^ret = gcnew array<long>(NDims + (_dim->batch_elems() > 1 ? 1 : 0));
				for (int iDim = 0; iDim < NDims; iDim++)
					ret[iDim] = (*_dim)[iDim];
				if (_dim->batch_elems() > 1)
					ret[NDims] = _dim->batch_elems();
				return ret;
			}
		}
		property long default[int] {
			long get(int index) {
				ExceptionWrap(
					return (*_dim)[index];
				)
			}
		};

	};

	public ref class Tensor {
	private:
		int GetActualPosFromArr(array<int> ^arr);
		int GetActualPosFromArr(std::vector<int> arr);
		int GetActualPosFromArr(std::vector<int> arr, bool fCheckBoundaries);
		Dim ^_dim = nullptr;
	internal:
		dynet::Tensor *__thisptr;
		std::vector<float> *_vec;
		Tensor(dynet::Tensor _tensor) {
			ExceptionWrap(
				_dim = gcnew Dim(_tensor.d);
			// Copy in the vector
			_vec = new std::vector<float>(as_vector(_tensor));
			__thisptr = new dynet::Tensor(_tensor);
			)
		}
		~Tensor() {
			this->!Tensor();
		}
		!Tensor() {
			if (__thisptr != NULL) {
				delete __thisptr;
				delete _vec;
			}
			__thisptr = NULL;
			_vec = NULL;
		}
		property int ndims {
			int get() {
				return (int)_dim->NDims + (_dim->BatchSize > 1 ? 1 : 0);
			}
		}
	public:
		Tensor(array<float> ^arr, array<long> ^shape);
		Tensor(array<float> ^arr);
		Tensor(array<array<float> ^> ^arr);
		Tensor(array<array<array<float> ^> ^> ^arr);
		Tensor(array<array<array<array<float> ^> ^> ^> ^arr);
		void SetBatched(bool fBatched);

		int NDims() { return ndims; }
		Dim ^Shape() { return _dim; }

		array<float> ^GetFlatVector();
		array<float> ^Get1DVector();
		array<array<float>^> ^Get2DVector();
		array<array<array<float>^>^> ^Get3DVector();
		array<array<array<array<float> ^>^>^> ^Get4DVector();
		void SetValue(array<int> ^pos, float value);
		void SetRowValue(array<int> ^pos, array<float> ^value);
		void SetRowValue(array<int> ^pos, array<float> ^value, int batchDim);
		float GetValue(array<int> ^pos);
		array<float> ^GetRowValue(array<int> ^pos);
		
		// Equals method
		// Check pointers. Since this object can be initialized in C#, compare hashcode at the end
		static bool operator==(Tensor^ t1, Tensor ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			// Case and compare pointers
			if (t1->__thisptr)
				return t1->__thisptr->v == t2->__thisptr->v;
			// Return if it's the same
			return t1->GetHashCode() == t2->GetHashCode();
		}
	};

	/// <summary>
	/// <para>Expressions are the building block of a Dynet computation graph.</para>
	/// <para>Expressions are the main data types being manipulated in a DyNet program. Each expression represents a sub-computation in a computation graph.</para>
	/// </summary>
	public ref class Expression {
	private:
		unsigned variableIndex;
		const dynet::Tensor *val;
		int self_cg_version;
		void GetValue();

		// For static graphs, we want to allow for float, vector pointers
		float *floatVal = NULL;
		bool fHasFloatVal = false;
		std::vector<float> *floatArrVal = NULL;
		bool fHasFloatArrVal = false;
			
		// Initialize
		void Init(dynet::Expression &inexpr) {
			CheckForInitialized();
			__thisptr = inexpr;
			val = NULL;
			self_cg_version = _cg_version;
		}
	internal:
		property dynet::Expression __thisptr {
			dynet::Expression get() {
				if (self_cg_version != _cg_version)
					throw gcnew Exception(gcnew String("Stale Expression (created before renewing the Computation Graph)."));
				return dynet::Expression(cg, variableIndex);
			}
			void set(dynet::Expression input) {
				variableIndex = input.i;
			}
		};
		Expression ^_multiply(Expression ^other);
		Expression ^_add(Expression ^other);
		Expression ^_divide(Expression ^other);
		Expression ^_subtract(Expression ^other);
		Expression ^_multiply(float other);
		Expression ^_add(float other);
		Expression ^_divide(float other);
		Expression ^_subtract(float other);
		Expression(dynet::Expression inexpr) {
			Init(inexpr);
		}
		Expression(dynet::Expression inexpr, float *floatValue) {
			Init(inexpr);
			fHasFloatVal = true;
			floatVal = floatValue;
		}
		Expression(dynet::Expression inexpr, std::vector<float> *floatArrValue) {
			Init(inexpr);
			fHasFloatArrVal = true;
			floatArrVal = floatArrValue;
		}
		~Expression() {
			this->!Expression();
		}
		!Expression() {
			//if (__thisactualptr != NULL)
			//	delete __thisactualptr;
			if (val != NULL)
				delete val;
			//if (fHasFloatVal && floatVal != NULL)
			//	delete floatVal;
			//if (fHasFloatArrVal && floatArrVal != NULL)
			//	delete floatArrVal;

			floatVal = NULL;
			floatArrVal = NULL;
			//__thisptr = NULL;
			val = NULL;
		}
	public:
		bool IsStale() {
			return self_cg_version != _cg_version;
		}
		
		static Expression ^operator-(Expression ^x, Expression ^other) { return x->_subtract(other); }
		static Expression ^operator*(Expression ^x, Expression ^other) { return x->_multiply(other); }
		static Expression ^operator/(Expression ^x, Expression ^other) { return x->_divide(other); }
		static Expression ^operator+(Expression ^x, Expression ^other) { return x->_add(other); }
		static Expression ^operator*(Expression ^x, float other) { return x->_multiply(other); }
		static Expression ^operator-(Expression ^x, float other) { return x->_subtract(other); }
		static Expression ^operator/(Expression ^x, float other) { return x->_divide(other); }
		static Expression ^operator+(Expression ^x, float other) { return x->_add(other); }
		static Expression ^operator*(float other, Expression ^x);
		static Expression ^operator-(float other, Expression ^x);
		static Expression ^operator/(float other, Expression ^x);
		static Expression ^operator+(float other, Expression ^x);
		static Expression ^operator-(Expression ^x) { return 0 - x; }

		void SetValue(float newVal);
		void SetValue(array<float> ^newVal);
		void SetValue(array<array<float>^> ^newVal);
		void Forward();
		void IncrementalForward();
		void Backward();
		void Backward(bool full);
		float ScalarValue();
		float ScalarValue(bool fRecalculate);
		array<float> ^VectorValue();
		array<float> ^VectorValue(bool fRecalculate);
		Tensor ^TensorValue();
		Tensor ^TensorValue(bool fRecalculate);
		Tensor ^Gradient();
		Dim ^Shape();

		// Add a brackets lookup
		property Expression ^default[int] {
			Expression ^get(int index);
		};
		property Expression ^default[int,int]{
			Expression ^get(int index, int dim);
		};
		property Expression ^default[array<int>^]{
			Expression ^get(array<int> ^indexes);
		};
		property Expression ^default[array<int>^, int]{
			Expression ^get(array<int> ^indexes, int dim);
		};
		// Equals method
		static bool operator==(Expression^ t1, Expression ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->variableIndex == t2->variableIndex;
		}
	};

	// Some headers that just need to know about Expression class
	std::vector<dynet::Expression> GetDyExpVector(List<Expression ^> ^l);
	std::vector<dynet::Expression> GetDyExpVector(array<Expression ^> ^arr);
	List<Expression ^> ^GetManagedExpList(std::vector<dynet::Expression> vec);
	array<Expression ^> ^GetManagedExpArr(std::vector<dynet::Expression> vec);

	/// <summary>
	/// <para>LookupParameters represents a table of parameters.</para>
	/// <para>They are used to embed a set of discrete objects (e.g. word embeddings). These are sparsely updated.</para>
	/// </summary>
	public ref class LookupParameter {
	internal:
		dynet::LookupParameter *__thisptr;
		LookupParameter(dynet::LookupParameter inp) {
			__thisptr = new dynet::LookupParameter(inp);
		}
		~LookupParameter() {
			this->!LookupParameter();
		}
		!LookupParameter() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	public:
		array<long> ^Shape();
		int Size();
		void InitRow(int iRow, array<float> ^row);
		void InitFromArray(Tensor ^t);
		array<Tensor ^> ^AsTensorArray();
		Tensor ^RowAsTensor(int iRow);
		array<Tensor ^> ^GradientArray();
		Tensor ^RowGradient(int iRow);
		void Zero();
		void Scale(float s);
		void ScaleGradient(float s);

		property Expression ^default[int]{
			Expression ^get(int index);
		};
		property Expression ^default[int,bool]{
			Expression ^get(int index, bool fUpdate);
		};
		property Expression ^default[array<int>^]{
			Expression ^get(array<int>^ indexes);
		};
		property Expression ^default[array<int>^,bool]{
			Expression ^get(array<int>^ indexes, bool fUpdate);
		};
		
		// Equals method
		static bool operator==(LookupParameter^ t1, LookupParameter ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->__thisptr->p == t2->__thisptr->p;
		}
	};

	/// <summary>
	/// <para>Parameters are things that are optimized. in contrast to a system like Torch where computational modules may have their own parameters, in DyNet parameters are just parameters.</para>
	/// </summary>
	public ref class Parameter {
	private:
		dynet::Parameter *__thisptr;
		Expression ^__exp;
		Expression ^__const_exp;
		Expression ^GetExpression() {
			// Default "update" is true, so return the regular experssion
			if (!__exp || __exp->IsStale())
				__exp = gcnew Expression(dynet::parameter(*cg, *__thisptr));
			return __exp;
		}
	internal:
		Parameter(dynet::Parameter inp) {
			__thisptr = new dynet::Parameter(inp);
			__exp = nullptr;
		}
		~Parameter() {
			this->!Parameter();
		}
		!Parameter() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	public:
		array<long> ^Shape();
		void ClipInPlace(float left, float right);
		void SetValue(Tensor ^val);
		void Zero();
		void Scale(float s);
		void ScaleGradient(float s);
		bool IsUpdated();
		void SetUpdated(bool b);
		float ScalarValue();
		float ScalarValue(bool fRecalculate);
		array<float> ^VectorValue();
		array<float> ^VectorValue(bool fRecalculate);
		Tensor ^TensorValue();
		Tensor ^TensorValue(bool fRecalculate);
		Tensor ^AsTensor();
		Tensor ^Gradient();
		Expression ^ToExpression();
		Expression ^ToExpression(bool fUpdate);
		property String ^Name {
			String ^get() {
				return gcnew String(__thisptr->get_fullname().c_str());
			}
		}

		// Equals method
		static bool operator==(Parameter^ t1, Parameter ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->__thisptr->p == t2->__thisptr->p;
		}
		static operator Expression ^(Parameter ^p) {
			return p->ToExpression();
		}
	};

	// Initializer
	/// <summary>
	/// <para>Base class for parameter initializer</para>
	/// </summary>
	public ref class ParamInitializer abstract {
	internal:
		ParamInitializer() { CheckForInitialized(); }
		dynet::ParameterInit *__thisptr;
		~ParamInitializer() {
			this->!ParamInitializer();
		}
		!ParamInitializer() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	};
	// NormalInitializer
	/// <summary>
	/// <para>Initialize the parameters with a gaussian distribution</para>
	/// </summary>
	public ref class NormalInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize the parameters with a gaussian distribution</para>
		/// </summary>
		NormalInitializer() {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitNormal(0, 1);
			)
		}
		/// <summary>
		/// <para>Initialize the parameters with a gaussian distribution</para>
		/// </summary>
		/// <param name='mean'>Mean of the distribution (default: 0)</param>
		/// <param name='var'>Variance of the distribution (default: 1)</param>
		NormalInitializer(float mean, float var) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitNormal(mean, var);
			)
		}
	};
	// UniformInitializer
	/// <summary>
	/// <para>Initialize the parameters with a uniform distribution</para>
	/// </summary>
	public ref class UniformInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize the parameters with a uniform distribution</para>
		/// </summary>
		/// <param name='scale'>Parameters are sampled from U([-scale,scale])</param>
		UniformInitializer(float scale) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitUniform(scale);
			)
		}
	};
	// ConstInitializer
	/// <summary>
	/// <para>Initialize the parameters with a constant value</para>
	/// </summary>
	public ref class ConstInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize the parameters with a constant value</para>
		/// </summary>
		/// <param name='c'>Value to initialize the parameters</param>
		ConstInitializer(float c) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitConst(c);
			)
		}
	};
	// IdentityInitializer
	/// <summary>
	/// <para>Initialize the parameters as the identity</para>
	/// <remarks>Only works with square matrices</remarks>
	/// </summary>
	public ref class IdentityInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize the parameters as the identity</para>
		/// <remarks>Only works with square matrices</remarks>
		/// </summary>
		IdentityInitializer() {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitIdentity();
			)
		}
	};
	// GlorotInitializer
	/// <summary>
	/// <para>Initializes the weights according to 'Glorot &amp; Bengio (2011) (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>)'</para>
	/// <para>If the dimensions of the parameter matrix are [m,n], the weights are sampled from U([-sqrt{6/(m+n)},sqrt{6/(m+n)}])</para>
	/// <para>In the case of 4d tensors (common in convolutional networks) of shape [XH,XW,XC,N] the weights are sampled from U([-sqrt{6/d},\sqrt{6/d}]), where d=XC*(XH*XW)+N*(XH*XW)</para>
	/// <para>The gain `g` depends on the activation function: </para>
	/// <para>'tanh': 1.0, 'ReLU': 0.5, 'sigmoid': 4.0, Any smooth function: 1/f'(0)</para>
	/// <remarks>*Note:* This is also known as **Xavier initialization**</remarks>
	/// </summary>
	public ref class GlorotInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initializes the weights according to 'Glorot &amp; Bengio (2011) (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>)'</para>
		/// <para>If the dimensions of the parameter matrix are [m,n], the weights are sampled from U([-sqrt{6/(m+n)},sqrt{6/(m+n)}])</para>
		/// <para>In the case of 4d tensors (common in convolutional networks) of shape [XH,XW,XC,N] the weights are sampled from U([-sqrt{6/d},\sqrt{6/d}]), where d=XC*(XH*XW)+N*(XH*XW)</para>
		/// <para>The gain `g` depends on the activation function: </para>
		/// <para>'tanh': 1.0, 'ReLU': 0.5, 'sigmoid': 4.0, Any smooth function: 1/f'(0)</para>
		/// <remarks>*Note:* This is also known as **Xavier initialization**</remarks>
		/// </summary>
		GlorotInitializer() {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitGlorot();
			)
		}
		/// <summary>
		/// <para>Initializes the weights according to 'Glorot &amp; Bengio (2011) (http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>)'</para>
		/// <para>If the dimensions of the parameter matrix are [m,n], the weights are sampled from U([-sqrt{6/(m+n)},sqrt{6/(m+n)}])</para>
		/// <para>In the case of 4d tensors (common in convolutional networks) of shape [XH,XW,XC,N] the weights are sampled from U([-sqrt{6/d},\sqrt{6/d}]), where d=XC*(XH*XW)+N*(XH*XW)</para>
		/// <para>The gain `g` depends on the activation function: </para>
		/// <para>'tanh': 1.0, 'ReLU': 0.5, 'sigmoid': 4.0, Any smooth function: 1/f'(0)</para>
		/// <remarks>*Note:* This is also known as **Xavier initialization**</remarks>
		/// </summary>
		/// <param name='isLookup'>Whether the parameter is alookup parameter (default: False)</param>
		/// <param name='gain'>Gain (Depends on the activation function) (default: 1.0)</param>
		GlorotInitializer(bool isLookup, float gain) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitGlorot(isLookup, gain);
			)
		}
	};
	// SaxeInitializer
	/// <summary>
	/// <para>Initializes according to 'Saxe et al. (2014) (https://arxiv.org/abs/1312.6120)'</para>
	/// <para>Initializes as a random orthonormal matrix</para>
	/// </summary>
	public ref class SaxeInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initializes according to 'Saxe et al. (2014) (https://arxiv.org/abs/1312.6120)'</para>
		/// <para>Initializes as a random orthonormal matrix</para>
		/// </summary>
		SaxeInitializer() {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitSaxe();
			)
		}
		/// <summary>
		/// <para>Initializes according to 'Saxe et al. (2014) (https://arxiv.org/abs/1312.6120)'</para>
		/// <para>Initializes as a random orthonormal matrix</para>
		/// </summary>
		/// <param name='scale'>scale to apply to the orthonormal matrix (default: 1.0)</param>
		SaxeInitializer(float scale) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitNormal(scale);
			)
		}
	};
	// FromFileInitializer
	/// <summary>
	/// <para>Initialize parameter from file</para>
	/// </summary>
	public ref class FromFileInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize parameter from file</para>
		/// </summary>
		/// <param name='fname'>Filename to initialize from</param>
		FromFileInitializer(System::String ^fname) {
			ExceptionWrap(
				char *str = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(fname).ToPointer();
				__thisptr = new dynet::ParameterInitFromFile(str);
				free(str);
			)
		}
	};
	// FromVectorInitializer
	/// <summary>
	/// <para>Initialize from float array</para>
	/// <remarks>Alternatively, use ParameterCollection.AddParametersFromTensor()</remarks>
	/// </summary>
	public ref class FromVectorInitializer : ParamInitializer {
	public:
		/// <summary>
		/// <para>Initialize from float array</para>
		/// <remarks>Alternatively, use ParameterCollection.AddParametersFromTensor()</remarks>
		/// </summary>
		/// <param name='arr'>Array to initialize with</param>
		FromVectorInitializer(array<float> ^arr) {
			ExceptionWrap(
				__thisptr = new dynet::ParameterInitFromVector(ConvertArrayToVector<float>(arr));
			)
		}
	};

	/// <summary>
	/// <para>A ParameterCollection holds Parameters. Use it to create, load and save parameters.</para>
	/// <para>(It used to be called Model in previous versions of DyNet)</para>
	/// <para>A ParameterCollection is a container for Parameters and LookupParameters.</para>
	/// <para>dynetsharp.Trainer objects take ParameterCollection objects that define which parameters are being trained.</para>
	/// <para>The values of the parameters in a collection can be persisted to and loaded from files.</para>
	/// <para>Hierarchy: The parameter collections can be nested, where each collection can hold zero or more sub-collection, which are also ParameterCollection objects.Each(sub-)collection contains the parameters in it and in all the(sub-)collections below it.</para>
	/// </summary>
	public ref class ParameterCollection {
	internal:
		dynet::ParameterCollection *__thisptr;
		ParameterCollection(dynet::ParameterCollection origPc) {
			__thisptr = new dynet::ParameterCollection(origPc);
		}
		~ParameterCollection() {
			this->!ParameterCollection();
		}
		!ParameterCollection() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	public:
		/// <summary>
		/// <para>A ParameterCollection holds Parameters. Use it to create, load and save parameters.</para>
		/// <para>(It used to be called Model in previous versions of DyNet)</para>
		/// <para>A ParameterCollection is a container for Parameters and LookupParameters.</para>
		/// <para>dynetsharp.Trainer objects take ParameterCollection objects that define which parameters are being trained.</para>
		/// <para>The values of the parameters in a collection can be persisted to and loaded from files.</para>
		/// <para>Hierarchy: The parameter collections can be nested, where each collection can hold zero or more sub-collection, which are also ParameterCollection objects.Each(sub-)collection contains the parameters in it and in all the(sub-)collections below it.</para>
		/// </summary>
		ParameterCollection() {
			CheckForInitialized();
			ExceptionWrap(
				__thisptr = new dynet::ParameterCollection();
			)
		}
		Parameter ^AddParameters(array<long>^ dim);
		Parameter ^AddParameters(array<long>^ dim, ParamInitializer ^pi);
		Parameter ^AddParametersFromTensor(Tensor ^t);
		Parameter ^AddParameters(array<long>^ dim, String ^device);
		Parameter ^AddParameters(array<long>^ dim, ParamInitializer ^pi, String ^device);
		Parameter ^AddParametersFromTensor(Tensor ^t, String ^device);
		LookupParameter ^AddLookupParameters(int size, array<long>^ dim);
		LookupParameter ^AddLookupParameters(int size, array<long>^ dim, ParamInitializer ^pi);
		LookupParameter ^AddLookupParametersFromTensor(Tensor ^t);
		LookupParameter ^AddLookupParameters(int size, array<long>^ dim, String ^device);
		LookupParameter ^AddLookupParameters(int size, array<long>^ dim, ParamInitializer ^pi, String ^device);
		LookupParameter ^AddLookupParametersFromTensor(Tensor ^t, String ^device);
		List<Parameter ^> ^GetParametersList();
		List<LookupParameter ^> ^GetLookupParametersList();
		ParameterCollection ^AddSubCollection();
		float GetWeightDecay();
		void SetWeightDecay(float lam);
		void SetWeightDecayLambda(float lam);
		int GetParameterCount();
		
		void Save(System::String ^filename);
		void Load(System::String ^filename);
	};
	

	/// <summary>
	/// <para>This is the main class for working with RNNs / LSTMs / GRUs.</para>
	/// <para>Request an RNNState initial_state() from a builder, and then progress from there.</para>
	/// </summary>
	public ref class RNNState {
	private:
		dynet::RNNBuilder *__builderptr;
		dynet::RNNPointer *__stateptr;
		RNNState ^_prev;
		Expression ^_out;
		int self_cg_version;
		void ensure_freshness() {
			if (self_cg_version != _cg_version)
				throw gcnew Exception(gcnew String("Stale State (created before renewing the Computation Graph)."));
		}
	internal:
		RNNState(dynet::RNNBuilder *builder, dynet::RNNPointer state) {
			__builderptr = builder;
			__stateptr = new dynet::RNNPointer(state);
			_prev = nullptr;
			_out = nullptr;
			self_cg_version = _cg_version;
		}
		RNNState(dynet::RNNBuilder *builder, dynet::RNNPointer state, RNNState ^prev, Expression ^out) {
			__builderptr = builder;
			__stateptr = new dynet::RNNPointer(state);
			_prev = prev;
			_out = out;
			self_cg_version = _cg_version;
		}
		~RNNState() {
			this->!RNNState();
		}
		!RNNState() {
			if (__stateptr != NULL)
				delete __stateptr;
			__stateptr = NULL;
		}
	public:
		RNNState ^GetPrev();
		Expression ^Output();
		RNNState ^AddInput(Expression ^e);
		List<RNNState ^> ^AddInputs(List<Expression ^> ^l);
		List<RNNState ^> ^AddInputs(... array<Expression ^> ^l);
		List<Expression ^> ^Transduce(List<Expression ^> ^l);
		List<Expression ^> ^Transduce(... array<Expression ^> ^l);
		RNNState ^SetH(... array<Expression ^> ^vecs);
		RNNState ^SetS(... array<Expression ^> ^vecs);
		List<Expression ^> ^GetH();
		List<Expression ^> ^GetS();
		void SetBuilderDropout(float f);
		void DisableBuilderDropout();

		// Equals method
		static bool operator==(RNNState^ t1, RNNState ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->__builderptr == t2->__builderptr && *t1->__stateptr == *t2->__stateptr;
		}
	};

	/// <summary>
	/// <para>Base class for RNNBuilder initializer</para>
	/// </summary>
	public ref class RNNBuilder abstract {
	internal:
		dynet::RNNBuilder *__thisptr;
		RNNState ^__init_state;
		int self_cg_version;
		~RNNBuilder() {
			this->!RNNBuilder();
		}
		!RNNBuilder() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	public:
		RNNState ^GetInitialState();
		RNNState ^GetInitialState(bool fUpdate);
		RNNState ^GetInitialState(array<Expression ^> ^vecs);
		RNNState ^GetInitialState(array<Expression ^> ^vecs, bool fUpdate);
		RNNState ^GetInitialStateFromRawVectors(array<Tensor ^> ^vecs);
		RNNState ^GetInitialStateFromRawVectors(array<Tensor ^> ^vecs, bool fUpdate);
		void SetDropout(float f);
		void DisableDropout();

		// Equals method
		static bool operator==(RNNBuilder^ t1, RNNBuilder ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->__thisptr == t2->__thisptr;
		}
	};

	/// <summary>
	/// <para>Simple RNNBuilder with tanh as the activation.</para>
	/// <para>This cell runs according to the following dynamics:</para>
	/// <para>h_t = tanh(W_{x}*x_t + W_{h} * h_{t-1} + b)</para>
	/// </summary>
	public ref class SimpleRNNBuilder : RNNBuilder {
	private:
		dynet::SimpleRNNBuilder *__thissimpleptr;
	public:
		/// <summary>
		/// <para>Simple RNNBuilder with tanh as the activation.</para>
		/// <para>This cell runs according to the following dynamics:</para>
		/// <para>h_t = tanh(W_{x}*x_t + W_{h} * h_{t-1} + b)</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		SimpleRNNBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thissimpleptr = new dynet::SimpleRNNBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}
		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
		/// <summary>
		/// <para>Set the dropout rates</para>
		/// <para>The dropout implemented here is the variational dropout introduced in (Gal, 2016 `http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks`)</para>
		/// <para>More specifically, dropout masks `{z_x}\sim{Bernoulli}(1-d)`, `{z_h}\sim{Bernoulli}(1-d_h)` are sampled at the start of each sequence.</para>
		/// <para>The dynamics of the cell are then modified to:</para>
		/// <para>h_t &amp; =\\tanh(W_{x}(\\frac 1 {1-d}\mathbf{z_x} \circ x_t)+W_{h}(\\frac 1 {1-d}\mathbf{z_h} \circ h_{t-1})+b)</para>
		/// <para>For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation</para>
		/// </summary>
		/// <param name='d'>Dropout rate `d` for the input.</param>
		/// <param name='d_h'>Dropout rate `d_h` for the hidden unit `h_t`</param>
		void SetDropout(float d, float d_h) {
			ExceptionWrap(
				__thissimpleptr->set_dropout(d, d_h);
			)
		}
		/// <summary>
		/// <para>Set dropout masks at the beginning of a sequence for a specific batch size</para>
		/// <para>If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element</para>
		/// <remarks>You need to call this __AFTER__ calling `GetInitialState()`</remarks>
		/// </summary>
        /// <param name='batchSize'>Batch size (default: {1})</param>
		void SetDropoutMask(int batchSize) {
			ExceptionWrap(
				__thissimpleptr->set_dropout_masks(batchSize);
			)
		}
	};
	// GRUBuilder
	public ref class GRUBuilder : RNNBuilder {
	private:
		dynet::GRUBuilder *__thisgruptr;
	public:
		/// <summary>
		/// <para>GRUBuilder</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		GRUBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thisgruptr = new dynet::GRUBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}
		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
	};
	// CoupledLSTMBuilder
	/// <summary>
	/// <para>CoupledLSTMBuilder creates an LSTM unit with coupled input and forget gate as well as peepholes connections.</para>
	/// <para>More specifically, here are the equations for the dynamics of this cell:</para>
	/// <para>i_t &amp; =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)</para>
	/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)</para>
	/// <para>c_t &amp; = c_{t-1}\circ (1-i_t) + \\tilde{c_t}\circ i_t</para>
	/// <para>&amp; = c_{t-1} + (\\tilde{c_t}-c_{t-1})\circ i_t</para>
	/// <para>o_t &amp; = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+W_{oc}c_{t}+b_o)</para>
	/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
	/// </summary>
	public ref class CoupledLSTMBuilder : RNNBuilder {
	private:
		dynet::CoupledLSTMBuilder *__thislstmptr;
	public:
		/// <summary>
		/// <para>CoupledLSTMBuilder creates an LSTM unit with coupled input and forget gate as well as peepholes connections.</para>
		/// <para>More specifically, here are the equations for the dynamics of this cell:</para>
		/// <para>i_t &amp; =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+W_{ic}c_{t-1}+b_i)</para>
		/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)</para>
		/// <para>c_t &amp; = c_{t-1}\circ (1-i_t) + \\tilde{c_t}\circ i_t</para>
		/// <para>&amp; = c_{t-1} + (\\tilde{c_t}-c_{t-1})\circ i_t</para>
		/// <para>o_t &amp; = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+W_{oc}c_{t}+b_o)</para>
		/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		CoupledLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thislstmptr = new dynet::CoupledLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}
		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
	};
	// VanillaLSTMBuilder
	/// <summary>
	/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
	/// <para>The parameters are initialized as follow:</para>
	/// <para> - `W_{*x}` (input connections): Sampled from `\mathcal U\left([\sqrt{\\frac{6}{4d_h + d_x}}]\\right)`</para>
	/// <para> - `W_{*h}` (recurrent connections): Sampled from `\mathcal U\left([\sqrt{\frac{6}{4d_h + d_h}}]\right)`</para>
	/// <para> - `b_{h}` (biases): Set to `0` except for `d_f` which is set to `1`</para>
	/// <para>This cell runs according to the following dynamics:</para>
	/// <para>i_t &amp; =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)</para>
	/// <para>f_t &amp; = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)</para>
	/// <para>o_t &amp; = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)</para>
	/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)</para>
	/// <para>c_t &amp; = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t</para>
	/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
	/// </summary>
	public ref class VanillaLSTMBuilder : RNNBuilder {
	private:
		dynet::VanillaLSTMBuilder *__thislstmptr;
	public:
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>For full description, see the docs for VanillaLSTMBuilder object (not for the constructor)</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		VanillaLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thislstmptr = new dynet::VanillaLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>For full description, see the docs for VanillaLSTMBuilder object (not for the constructor)</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		/// <param name='lnLstm'>Whether to use layer normalization(default: false)</param>
		/// <param name='forgetBias'>value to use as forget gate bias(default: 1.0)</param>
		VanillaLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc, bool lnLstm, float forgetBias) {
			ExceptionWrap(
				__thisptr = __thislstmptr = new dynet::VanillaLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr, lnLstm, forgetBias);
			)
		}
		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
		void SetDropout(float d, float d_r);
		void SetDropoutMask(int batchSize);
	};
	/// <summary>
	/// For documentation, see <see cref="VanillaLSTMBuilder" />
	/// </summary>
	public ref class LSTMBuilder : VanillaLSTMBuilder {
	public:
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>For full description, see the docs for VanillaLSTMBuilder object (not for the constructor)</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		LSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) : VanillaLSTMBuilder(layers, inputDim, hiddenDim, pc) {}
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>For full description, see the docs for VanillaLSTMBuilder object (not for the constructor)</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		/// <param name='lnLstm'>Whether to use layer normalization(default: false)</param>
		/// <param name='forgetBias'>value to use as forget gate bias(default: 1.0)</param>
		LSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc, bool lnLstm, float forgetBias) : VanillaLSTMBuilder(layers, inputDim, hiddenDim, pc, lnLstm, forgetBias) {}
	};
	
	// SparseLSTMBuilder
	/// <summary>
	/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
	/// <para>During training the sparsity of the LSTM has to be increased incrementally.</para>
	/// <para>Sparsity is controlled using the set_sparsity method. This works by sorting all the weights based on their magnitude and applying mask on the top x-percent weight with the lowest magnitude.</para>
	/// <para>More details on the process can be found in (Narang et al., 2017 `https://arxiv.org/pdf/1704.05119.pdf`). The rest of the implementation is identical to VanillaLSTM</para>
	/// <para>DISCLAIMER: This is an experimental/untested module.</para>
	/// </summary>
	public ref class SparseLSTMBuilder : RNNBuilder {
	private:
		dynet::SparseLSTMBuilder *__thissparsevanillaptr;
	public:
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>During training the sparsity of the LSTM has to be increased incrementally.</para>
		/// <para>Sparsity is controlled using the set_sparsity method. This works by sorting all the weights based on their magnitude and applying mask on the top x-percent weight with the lowest magnitude.</para>
		/// <para>More details on the process can be found in (Narang et al., 2017 `https://arxiv.org/pdf/1704.05119.pdf`). The rest of the implementation is identical to VanillaLSTM</para>
		/// <para>DISCLAIMER: This is an experimental/untested module.</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		SparseLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thissparsevanillaptr = new dynet::SparseLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>VanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>During training the sparsity of the LSTM has to be increased incrementally.</para>
		/// <para>Sparsity is controlled using the set_sparsity method. This works by sorting all the weights based on their magnitude and applying mask on the top x-percent weight with the lowest magnitude.</para>
		/// <para>More details on the process can be found in (Narang et al., 2017 `https://arxiv.org/pdf/1704.05119.pdf`). The rest of the implementation is identical to VanillaLSTM</para>
		/// <para>DISCLAIMER: This is an experimental/untested module.</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		/// <param name='lnLstm'>Whether to use layer normalization(default: false)</param>
		/// <param name='forgetBias'>value to use as forget gate bias(default: 1.0)</param>
		SparseLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc, bool lnLstm, float forgetBias) {
			ExceptionWrap(
				__thisptr = __thissparsevanillaptr = new dynet::SparseLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr, lnLstm, forgetBias);
			)
		}
		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
		void SetDropout(float d, float d_r);
		void SetDropoutMask(int batchSize);

		/// <summary>
		/// <para>Set the sparsity rate</para>
		/// </summary>
		/// <param name='sparsity'>The relative number of weights that will be pruned</param>
		void SetSparsity(float sparsity) {
			ExceptionWrap(
				__thissparsevanillaptr->set_sparsity(sparsity);
			)
		}
	};

	// CompactVanillaLSTMBuilder
	public ref class CompactVanillaLSTMBuilder : RNNBuilder {
	private:
		dynet::CompactVanillaLSTMBuilder *__thiscompvanillaptr;
	public:
		/// <summary>
		/// <para>CompactVanillaLSTM allows to create an "standard" LSTM, ie with decoupled input and forget gate and no peepholes connections</para>
		/// <para>This cell runs according to the following dynamics:</para>
		/// <para>i_t &amp; =\sigma(W_{ix}x_t+W_{ih}h_{t-1}+b_i)</para>
		/// <para>f_t &amp; = \sigma(W_{fx}x_t+W_{fh}h_{t-1}+b_f+1)</para>
		/// <para>o_t &amp; = \sigma(W_{ox}x_t+W_{oh}h_{t-1}+b_o)</para>
		/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}x_t+W_{ch}h_{t-1}+b_c)</para>
		/// <para>c_t &amp; = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t</para>
		/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		CompactVanillaLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thiscompvanillaptr = new dynet::CompactVanillaLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}

		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
		void SetDropout(float d, float d_r);
		void SetDropoutMask(int batchSize);

		/// <summary>
		/// <para>Set the gaussian weight noise</para>
		/// </summary>
		/// <param name='std'>Standard deviation of weight noise</param>
		void SetWeightNoise(float std) {
			ExceptionWrap(
				__thiscompvanillaptr->set_weightnoise(std);
			)
		}
	};

	// FastLSTMBuilder
	public ref class FastLSTMBuilder : RNNBuilder {
	private:
		dynet::FastLSTMBuilder *__thisfastlstmptr;
	public:
		/// <summary>
		/// <para>FastLSTMBuilder</para>
		/// </summary>
		/// <param name='layers'>Number of layers</param>
		/// <param name='inputDim'>Dimension of the input</param>
		/// <param name='hiddenDim'>Dimension of the recurrent units</param>
		/// <param name='pc'>ParameterCollection to hold the parameters</param>
		FastLSTMBuilder(int layers, int inputDim, int hiddenDim, ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = __thisfastlstmptr = new dynet::FastLSTMBuilder(layers, inputDim, hiddenDim, *pc->__thisptr);
			)
		}

		List<List<Parameter ^> ^> ^GetParameters();
		List<List<Expression ^> ^> ^GetParameterExpressions();
	};

	/// <summary>
	/// <para>Generic trainer</para>
	/// <para>Attributes: learning_rate(number): Global learning rate for all parameters</para>
	/// </summary>
	public ref class Trainer abstract {
	internal:
		dynet::Trainer *__thisptr;
		Trainer() { CheckForInitialized(); }
		~Trainer() {
			this->!Trainer();
		}
		!Trainer() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	public:
		/// <summary>
		/// <para>Update the parameters</para>
		/// <para>The update equation is different for each trainer, check the online c++ documentation for more details on what each trainer does</para>
		/// </summary>
		void Update() {
			ExceptionWrap(
				__thisptr->update();
			)
		}
		/// <summary>
		/// <para>Restarts the optimizer</para>
		/// <para>Clears all momentum values and assimilate (if applicable)</para>
		/// </summary>
		void Restart() {
			ExceptionWrap(
				__thisptr->restart();
			)
		}
		/// <summary>
		/// <para>Restarts the optimizer</para>
		/// <para>Clears all momentum values and assimilate (if applicable)</para>
		/// </summary>
		/// <param name='learningRate'>The new learning rate (optional)</param>
		void Restart(float learningRate) {
			ExceptionWrap(
				__thisptr->restart(learningRate);
			)
		}
		/// <summary>
		/// <para>Outputs information about the trainer in the stderr </para>
		/// <para>(number of updates since last call, number of clipped gradients, learning rate, etc...)</para>
		/// </summary>
		void Status() {
			ExceptionWrap(
				__thisptr->status();
			)
		}
		/// <summary>
		/// <para>Sets updates to sparse updates</para>
		/// <para>DyNet trainers support two types of updates for lookup parameters, sparse and dense. Sparse updates are the default. They have the potential to be faster, as they only touch the parameters that have non-zero gradients. However, they may not always be faster (particulary on GPU with mini-batch training), and are not precisely numerically correct for some update rules such as MomentumTrainer and AdamTrainer. Thus, if you set this variable to false, the trainer will perform dense updates and be precisely correct, and maybe faster sometimes.</para>
		/// <param name='fSu'>Flag to activate/deactivate sparse updates</param>
		/// </summary>
		void SetSparseUpdates(bool fSu) {
			ExceptionWrap(
				__thisptr->sparse_updates_enabled = fSu;
			)
		}
		/// <summary>
		/// <para>Set clipping thershold</para>
		/// <remarks>Gradients are clipped to 5 by default.</remarks>
		/// <remarks>To deactivate clipping, set the threshold to be &lt;=0</remarks>
		/// </summary>
		/// <param name='thresh'>Clipping threshold</param>
		void SetClipThreshold(float thresh) {
			ExceptionWrap(
				if (thresh <= 0) {
				__thisptr->clipping_enabled = false;
				__thisptr->clip_threshold = 0.0;
			}
			else {
				__thisptr->clipping_enabled = true;
				__thisptr->clip_threshold = thresh;
			}
			)
		}
		/// <summary>
		/// <para>Get clipping threshold</para>
		/// </summary>
		float GetClipThreshold() {
			ExceptionWrap(
				return __thisptr->clip_threshold;
			)
		}
		void SetLearningRate(float lr) {
			ExceptionWrap(
				LearningRate = lr;
			)
		}
		property float LearningRate {
			void set(float lr) { __thisptr->learning_rate = lr; }
			float get() { return __thisptr->learning_rate; }
		}

		// Equals method
		static bool operator==(Trainer^ t1, Trainer ^t2) {
			bool t1Null = Object::ReferenceEquals(t1, nullptr);
			bool t2Null = Object::ReferenceEquals(t2, nullptr);
			if (t1Null || t2Null) return t1Null == t2Null;
			return t1->__thisptr == t2->__thisptr;
		}
	};

	/// <summary>
	/// <para>Stochastic gradient descent trainer</para>
	/// <para>This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.</para>
	/// </summary>
	public ref class SimpleSGDTrainer : Trainer {
	public:
		/// <summary>
		/// <para>Stochastic gradient descent trainer</para>
		/// <para>This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		SimpleSGDTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::SimpleSGDTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>Stochastic gradient descent trainer</para>
		/// <para>This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='learningRate'>Initial learning rate (default: 0.1)</param>
		SimpleSGDTrainer(ParameterCollection ^pc, float learningRate) {
			ExceptionWrap(
				__thisptr = new dynet::SimpleSGDTrainer(*pc->__thisptr, learningRate);
			)
		}
	};
	/// <summary>
	/// <para>This trainer performs stochastic gradient descent with a cyclical learning rate as proposed in (Smith, 2015 `https://arxiv.org/abs/1506.01186`).</para>
	/// <para>This uses a triangular function with optional exponential decay.</para>
	/// </summary>
	public ref class CyclicalSGDTrainer : Trainer {
	public:
		/// <summary>
		/// <para>This trainer performs stochastic gradient descent with a cyclical learning rate as proposed in (Smith, 2015 `https://arxiv.org/abs/1506.01186`).</para>
		/// <para>This uses a triangular function with optional exponential decay.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		CyclicalSGDTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::CyclicalSGDTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>This trainer performs stochastic gradient descent with a cyclical learning rate as proposed in (Smith, 2015 `https://arxiv.org/abs/1506.01186`).</para>
		/// <para>This uses a triangular function with optional exponential decay.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='learningRateMin'>Lower learning rate (default: {0.01})</param>
		/// <param name='learningRateMax'>Upper learning rate (default: {0.1})</param>
		/// <param name='stepSize'>Period of the triangular function in number of iterations (__not__ epochs). According to the original paper, this should be set around (2-8) x (training iterations in epoch) (default: {2000})</param>
		/// <param name='gamma'>Learning rate upper bound decay parameter (1.0 = no decay) (default: {1.0})</param>
		CyclicalSGDTrainer(ParameterCollection ^pc, float learningRateMin, float learningRateMax, float stepSize, float gamma) {
			ExceptionWrap(
				__thisptr = new dynet::CyclicalSGDTrainer(*pc->__thisptr);
			)
		}
	};
	/// <summary>
	/// <para>Stochastic gradient descent with momentum</para>
	/// <para>This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.</para>
	/// </summary>
	public ref class MomentumSGDTrainer : Trainer {
	public:
		/// <summary>
		/// <para>Stochastic gradient descent with momentum</para>
		/// <para>This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		MomentumSGDTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::MomentumSGDTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>Stochastic gradient descent with momentum</para>
		/// <para>This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='learningRate'>Initial learning rate (default: 0.1)</param>
		/// <param name='mom'>Momentum (default: 0.9)</param>
		MomentumSGDTrainer(ParameterCollection ^pc, float learningRate, float mom) {
			ExceptionWrap(
				__thisptr = new dynet::MomentumSGDTrainer(*pc->__thisptr, learningRate, mom);
			)
		}
	};
	/// <summary>
	/// <para>Adagrad optimizer</para>
	/// <para>The adagrad algorithm assigns a different learning rate to each parameter.</para>
	/// </summary>
	public ref class AdagradTrainer : Trainer {
	public:
		/// <summary>
		/// <para>Adagrad optimizer</para>
		/// <para>The adagrad algorithm assigns a different learning rate to each parameter.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		AdagradTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::AdagradTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>Adagrad optimizer</para>
		/// <para>The adagrad algorithm assigns a different learning rate to each parameter.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='learningRate'>Initial learning rate (default: 0.1)</param>
		/// <param name='eps'>Epsilon parameter to prevent numerical instability (default: 1e-20)</param>
		AdagradTrainer(ParameterCollection ^pc, float learningRate, float eps) {
			ExceptionWrap(
				__thisptr = new dynet::AdagradTrainer(*pc->__thisptr, learningRate, eps);
			)
		}
	};
	/// <summary>
	/// <para>AdaDelta optimizer</para>
	/// <para>The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.</para>
	/// </summary>
	public ref class AdadeltaTrainer : Trainer {
	public:
		/// <summary>
		/// <para>AdaDelta optimizer</para>
		/// <para>The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		AdadeltaTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::AdadeltaTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>AdaDelta optimizer</para>
		/// <para>The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='eps'>Epsilon parameter to prevent numerical instability (default: 1e-6)</param>
		/// <param name='rho'>Update parameter for the moving average of updates in the numerator(default: 0.95)</param>
		AdadeltaTrainer(ParameterCollection ^pc, float eps, float rho) {
			ExceptionWrap(
				__thisptr = new dynet::AdadeltaTrainer(*pc->__thisptr, eps, rho);
			)
		}
	};
	/// <summary>
	/// <para>RMSProp optimizer</para>
	/// <para>The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter rho.</para>
	/// </summary>	
	public ref class RMSPropTrainer : Trainer {
	public:
		/// <summary>
		/// <para>RMSProp optimizer</para>
		/// <para>The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter rho.</para>
		/// </summary>		
		/// <param name='pc'>ParameterCollection to be trained</param>
		RMSPropTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::RMSPropTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>RMSProp optimizer</para>
		/// <para>The RMSProp optimizer is a variant of Adagrad where the squared sum of previous gradients is replaced with a moving average with parameter rho.</para>
		/// </summary>		
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='learningRate'>Initial learning rate (default: 0.001)</param>
		/// <param name='eps'>Epsilon parameter to prevent numerical instability (default: 1e-8)</param>
		/// <param name='rho'>Update parameter for the moving average (`rho = 0` is equivalent to using Adagrad) (default: 0.9)</param>
		RMSPropTrainer(ParameterCollection ^pc, float learningRate, float eps, float rho) {
			ExceptionWrap(
				__thisptr = new dynet::RMSPropTrainer(*pc->__thisptr, learningRate, eps, rho);
			)
		}
	};
	/// <summary>
	/// <para>Adam optimizer</para>
	/// <para>The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient</para>
	/// </summary>
	public ref class AdamTrainer : Trainer {
	public:
		/// <summary>
		/// <para>Adam optimizer</para>
		/// <para>The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		AdamTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::AdamTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>Adam optimizer</para>
		/// <para>The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='alpha'>Initial learning rate (default: 0.001)</param>
		/// <param name='beta1'>Moving average parameter for the mean (default: 0.9)</param>
		/// <param name='beta2'>Moving average parameter for the variance (default: 0.999)</param>
		/// <param name='eps'>Epsilon parameter to prevent numerical instability (default: 1e-8)</param>
		AdamTrainer(ParameterCollection ^pc, float alpha, float beta1, float beta2, float eps) {
			ExceptionWrap(
				__thisptr = new dynet::AdamTrainer(*pc->__thisptr, alpha, beta1, beta2, eps);
			)
		}
	};
	/// <summary>
	/// <para>AMSGrad optimizer</para>
	/// <para>The AMSGrad optimizer is similar to Adam which uses unbiased estimates of the first and second moments of the gradient, however AMSGrad keeps the maximum of all the second moments and uses that instead</para>
	/// </summary>
	public ref class AmsgradTrainer : Trainer {
	public:
		/// <summary>
		/// <para>AMSGrad optimizer</para>
		/// <para>The AMSGrad optimizer is similar to Adam which uses unbiased estimates of the first and second moments of the gradient, however AMSGrad keeps the maximum of all the second moments and uses that instead</para>
		/// </summary>		
		/// <param name='pc'>ParameterCollection to be trained</param>
		AmsgradTrainer(ParameterCollection ^pc) {
			ExceptionWrap(
				__thisptr = new dynet::AmsgradTrainer(*pc->__thisptr);
			)
		}
		/// <summary>
		/// <para>AMSGrad optimizer</para>
		/// <para>The AMSGrad optimizer is similar to Adam which uses unbiased estimates of the first and second moments of the gradient, however AMSGrad keeps the maximum of all the second moments and uses that instead</para>
		/// </summary>
		/// <param name='pc'>ParameterCollection to be trained</param>
		/// <param name='alpha'>Initial learning rate (default: 0.001)</param>
		/// <param name='beta1'>Moving average parameter for the mean (default: 0.9)</param>
		/// <param name='beta2'>Moving average parameter for the variance (default: 0.999)</param>
		/// <param name='eps'>Epsilon parameter to prevent numerical instability (default: 1e-8)</param>
		AmsgradTrainer(ParameterCollection ^pc, float alpha, float beta1, float beta2, float eps) {
			ExceptionWrap(
				__thisptr = new dynet::AmsgradTrainer(*pc->__thisptr, alpha, beta1, beta2, eps);
			)
		}
	};
		
	public ref class DynetParams {
	private:
		size_t maxMemory = 0;
		DynetParams(dynet::DynetParams ptr) {
			ExceptionWrap(
				__thisptr = new dynet::DynetParams(ptr);
			);
		}
		~DynetParams() {
			this->!DynetParams();
		}
		!DynetParams() {
			if (__thisptr != NULL)
				delete __thisptr;
			__thisptr = NULL;
		}
	internal:
		dynet::DynetParams *__thisptr;
	public:
		DynetParams();
		/// <summary>
		/// <para>Memory allocated to dynet, unit is in MB</para>
		/// </summary>
		property size_t MemDescriptor {
			void set(size_t mem) { __thisptr->mem_descriptor = std::to_string(mem); }
			size_t get() { return stoull(__thisptr->mem_descriptor); }
		}
		/// <summary>
		/// <para>Maximum memory allowed before clearing (memory is cleared at RenewCG(), 0 means no limit)</para>
		/// </summary>
		property size_t MaxMemDescriptor {
			void set(size_t mem) { maxMemory = mem; }
			size_t get() { return maxMemory; }
		}
		/// <summary>
		/// <para>Random seed for dynet</para>
		/// </summary>
		property unsigned int RandomSeed {
			void set(unsigned int seed) { __thisptr->random_seed = seed; }
			unsigned int get() { return __thisptr->random_seed; }
		}
		/// <summary>
		/// <para>Activate autobatching for calculations</para>
		/// </summary>
		property bool AutoBatch {
			void set(bool fBatch) { __thisptr->autobatch = fBatch && 1; }
			bool get() { return __thisptr->autobatch && 1; }
		}
		/// <summary>
		/// <para>Activate autobatching debug</para>
		/// </summary>
		property bool Profiling {
			void set(bool fProfile) { __thisptr->profiling = fProfile && 1; }
			bool get() { return __thisptr->profiling && 1; }
		}
		/// <summary>
		/// <para>Weight decay parameter</para>
		/// </summary>
		property float WeightDecay {
			void set(float weight) { __thisptr->weight_decay = weight; }
			float get() { return __thisptr->weight_decay; }
		}
		/// <summary>
		/// <para>Shared parameters flag</para>
		/// </summary>
		property bool SharedParameters {
			void set(bool fProfile) { __thisptr->shared_parameters = fProfile && 1; }
			bool get() { return __thisptr->shared_parameters && 1; }
		}
		void SetRequestedGPUs(int n);
		property int RequestedGPUs {
			int get() { return __thisptr->ngpus_requested ? __thisptr->requested_gpus : -1; }
		}
		void SetGPUMask(array<bool> ^mask);
		void SetGPUMask(List<bool> ^mask);
		void SetDeviceIDs(String ^devices);
		void SetDeviceIDs(... array<String ^> ^devices);
		void SetDeviceIDs(List<String ^> ^devices);
		bool GetGPUMaskState(int index);
		void Initialize();
		static DynetParams ^FromArgs(array<String ^> ^args);
		void UpdateMemDescriptors();
	};

	public enum class GradientMode {
		ZeroGradient = dynet::zero_gradient,
		StraightThroughGradient = dynet::straight_through_gradient,
	};
	public enum class DeviceType {
		CPU = (int)dynet::DeviceType::CPU,
		GPU = (int)dynet::DeviceType::GPU
	};
	public ref class DeviceInfo {
	private:
		String ^name;
		int id;
		DeviceType dtype;
	internal:
		DeviceInfo(std::string name, int id, dynet::DeviceType dtype) : name(gcnew String(name.c_str())), id(id), dtype((DeviceType)dtype) {}
	public:
		property String ^Name { String ^get() { return name; } }
		property int Id { int get() { return id; } }
		property DeviceType dType { DeviceType get() { return dtype; } }
	};

	public ref class DynetFunctions abstract sealed {
		// Static functions
	public:
		static void RenewCG(bool fImmediateCompute, bool fCheckValidity);
		static void RenewCG();
		static void CheckpointCG();
		static void RevertCG();
		static Expression ^lookup(LookupParameter ^lp, int index);
		static Expression ^lookup(LookupParameter ^lp, int index, bool fUpdate);
		static Expression ^lookup_batch(LookupParameter ^lp, array<int> ^indexes);
		static Expression ^lookup_batch(LookupParameter ^lp, array<int> ^indexes, bool fUpdate);
		static Expression ^parameter(Parameter ^p);
		static Expression ^const_parameter(Parameter ^p);
		static Expression ^pick(Expression ^exp, int index);
		static Expression ^pick(Expression ^exp, int index, int dim);
		static Expression ^pick_batch(Expression ^exp, array<int> ^indexes);
		static Expression ^pick_batch(Expression ^exp, array<int> ^indexes, int dim);
		static Expression ^input(float num);
		static Expression ^input(float num, String ^device);
		static Expression ^input(array<float>^ num);
		static Expression ^input(array<float>^ num, int batchSize);
		static Expression ^input(array<float>^ num, String ^device);
		static Expression ^input(array<float>^ num, int batchSize, String ^device);
		static Expression ^input(array<array<float>^>^ num);
		static Expression ^input(array<array<float>^>^ num, int batchSize);
		static Expression ^input(array<array<float>^>^ num, String ^device);
		static Expression ^input(array<array<float>^>^ num, int batchSize, String ^device);
		static Expression ^inputTensor(Tensor ^tensor);
		static Expression ^inputTensor(Tensor ^tensor, String ^device);
		static Expression ^inputVector(long dim);
		static Expression ^inputVector(long dim, int batchSize);
		static Expression ^inputTensor(array<long> ^dim);
		static Expression ^inputTensor(array<long> ^dim, int batchSize);
		static Expression ^inputVector(long dim, String ^device);
		static Expression ^inputVector(long dim, int batchSize, String ^device);
		static Expression ^inputTensor(array<long> ^dim, String ^device);
		static Expression ^inputTensor(array<long> ^dim, int batchSize, String ^device);
		static Expression ^average(List<Expression^> ^l);
		static Expression ^average(... array<Expression^> ^arr);
		static Expression ^esum(List<Expression^> ^l);
		static Expression ^esum(... array<Expression^> ^arr);
		static Expression ^sum(List<Expression^> ^l);
		static Expression ^sum(... array<Expression^> ^arr);
		static Expression ^zeros(array<long> ^dim);
		static Expression ^zeros(array<long> ^dim, int batchSize);
		static Expression ^zeros(array<long> ^dim, String ^device);
		static Expression ^zeros(array<long> ^dim, int batchSize, String ^device);
		static Expression ^one_hot(int dim, int idx);
		static Expression ^one_hot(int dim, int idx, String ^device);
		static Expression ^one_hot(int dim, List<int> ^idx);
		static Expression ^one_hot(int dim, array<int> ^idx);
		static Expression ^one_hot(int dim, List<int> ^idx, String ^device);
		static Expression ^one_hot(int dim, array<int> ^idx, String ^device);
		static Expression ^ones(array<long> ^dim);
		static Expression ^ones(array<long> ^dim, int batchSize);
		static Expression ^ones(array<long> ^dim, String ^device);
		static Expression ^ones(array<long> ^dim, int batchSize, String ^device);
		static Expression ^constant(array<long> ^dim, float val);
		static Expression ^constant(array<long> ^dim, float val, int batchSize);
		static Expression ^constant(array<long> ^dim, float val, String ^device);
		static Expression ^constant(array<long> ^dim, float val, int batchSize, String ^device);
		static Expression ^random_normal(array<long> ^dim);
		static Expression ^random_normal(array<long> ^dim, int batchSize);
		static Expression ^random_normal(array<long> ^dim, String ^device);
		static Expression ^random_normal(array<long> ^dim, int batchSize, String ^device);
		static Expression ^random_normal(array<long> ^dim, float mean, float stddev);
		static Expression ^random_normal(array<long> ^dim, float mean, float stddev, int batchSize);
		static Expression ^random_normal(array<long> ^dim, float mean, float stddev, String ^device);
		static Expression ^random_normal(array<long> ^dim, float mean, float stddev, int batchSize, String ^device);
		static Expression ^random_bernoulli(array<long> ^dim, float p);
		static Expression ^random_bernoulli(array<long> ^dim, float p, int batchSize);
		static Expression ^random_bernoulli(array<long> ^dim, float p, String ^device);
		static Expression ^random_bernoulli(array<long> ^dim, float p, int batchSize, String ^device);
		static Expression ^random_bernoulli(array<long> ^dim, float p, float scale);
		static Expression ^random_bernoulli(array<long> ^dim, float p, float scale, int batchSize);
		static Expression ^random_bernoulli(array<long> ^dim, float p, float scale, String ^device);
		static Expression ^random_bernoulli(array<long> ^dim, float p, float scale, int batchSize, String ^device);
		static Expression ^random_uniform(array<long> ^dim, float left, float right);
		static Expression ^random_uniform(array<long> ^dim, float left, float right, int batchSize);
		static Expression ^random_uniform(array<long> ^dim, float left, float right, String ^device);
		static Expression ^random_uniform(array<long> ^dim, float left, float right, int batchSize, String ^device);
		static Expression ^random_gumbel(array<long> ^dim);
		static Expression ^random_gumbel(array<long> ^dim, int batchSize);
		static Expression ^random_gumbel(array<long> ^dim, String ^device);
		static Expression ^random_gumbel(array<long> ^dim, int batchSize, String ^device);
		static Expression ^random_gumbel(array<long> ^dim, float mu, float beta);
		static Expression ^random_gumbel(array<long> ^dim, float mu, float beta, int batchSize);
		static Expression ^random_gumbel(array<long> ^dim, float mu, float beta, String ^device);
		static Expression ^random_gumbel(array<long> ^dim, float mu, float beta, int batchSize, String ^device);
		static Expression ^flip_gradient(Expression ^x);
		static Expression ^scale_gradient(Expression ^x);
		static Expression ^scale_gradient(Expression ^x, float lambd);
		static Expression ^argmax(Expression ^x, GradientMode gm);
		static Expression ^cdiv(Expression ^x, Expression ^y);
		static Expression ^cmult(Expression ^x, Expression ^y);
		static Expression ^colwise_add(Expression ^x, Expression ^y);
		static Expression ^inverse(Expression ^x);
		static Expression ^logdet(Expression ^x);
		static Expression ^trace_of_product(Expression ^x, Expression ^y);
		static Expression ^dot_product(Expression ^x, Expression ^y);
		static Expression ^circ_conv(Expression ^x, Expression ^y);
		static Expression ^circ_corr(Expression ^x, Expression ^y);
		static Expression ^squared_norm(Expression ^x);
		static Expression ^l2_norm(Expression ^x);
		static Expression ^squared_distance(Expression ^x, Expression ^y);
		static Expression ^l1_distance(Expression ^x, Expression ^y);
		static Expression ^binary_log_loss(Expression ^x, Expression ^y);
		static Expression ^filter1d_narrow(Expression ^x, Expression ^y);
		static Expression ^conv2d(Expression ^x, Expression ^y, array<int> ^stride);
		static Expression ^conv2d(Expression ^x, Expression ^y, array<int> ^stride, bool is_valid);
		static Expression ^conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride);
		static Expression ^conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride, bool is_valid);
		static Expression ^maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride);
		static Expression ^maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride, bool is_valid);
		static Expression ^sin(Expression ^x);
		static Expression ^cos(Expression ^x);
		static Expression ^tan(Expression ^x);
		static Expression ^asin(Expression ^x);
		static Expression ^acos(Expression ^x);
		static Expression ^atan(Expression ^x);
		static Expression ^sinh(Expression ^x);
		static Expression ^cosh(Expression ^x);
		static Expression ^tanh(Expression ^x);
		static Expression ^asinh(Expression ^x);
		static Expression ^acosh(Expression ^x);
		static Expression ^atanh(Expression ^x);
		static Expression ^exp(Expression ^x);
		static Expression ^square(Expression ^x);
		static Expression ^sqrt(Expression ^x);
		static Expression ^abs(Expression ^x);
		static Expression ^erf(Expression ^x);
		static Expression ^cube(Expression ^x);
		static Expression ^log(Expression ^x);
		static Expression ^log_sigmoid(Expression ^x);
		static Expression ^lgamma(Expression ^x);
		static Expression ^logistic(Expression ^x);
		static Expression ^sigmoid(Expression ^x);
		static Expression ^rectify(Expression ^x);
		static Expression ^relu(Expression ^x);
		static Expression ^elu(Expression ^x);
		static Expression ^elu(Expression ^x, float alpha);
		static Expression ^selu(Expression ^x);
		static Expression ^silu(Expression ^x);
		static Expression ^silu(Expression ^x, float beta);
		static Expression ^log_softmax(Expression ^x);
		static Expression ^softmax(Expression ^x);
		static Expression ^softmax(Expression ^x, int d);
		static Expression ^sparsemax(Expression ^x);
		static Expression ^softsign(Expression ^x);
		static Expression ^constrained_softmax(Expression ^x, Expression ^y);
		static Expression ^pow(Expression ^x, Expression ^y);
		static Expression ^emin(Expression ^x, Expression ^y);
		static Expression ^emax(Expression ^x, Expression ^y);
		static Expression ^min(Expression ^x, Expression ^y);
		static Expression ^max(Expression ^x, Expression ^y);
		static Expression ^transpose(Expression ^x);
		static Expression ^transpose(Expression ^x, array<int> ^dims);
		static Expression ^select_rows(Expression ^x, array<int> ^rs);
		static Expression ^select_cols(Expression ^x, array<int> ^cs);
		static Expression ^sum_elems(Expression ^x);
		static Expression ^sum_dim(Expression ^x, array<int> ^d);
		static Expression ^sum_dim(Expression ^x, array<int> ^d, bool b);
		static Expression ^sum_batches(Expression ^x);
		static Expression ^cumsum(Expression ^x);
		static Expression ^cumsum(Expression ^x, int d);
		static Expression ^mean_elems(Expression ^x);
		static Expression ^mean_dim(Expression ^x, array<int> ^d, bool b);
		static Expression ^mean_batches(Expression ^x);
		static Expression ^std_elems(Expression ^x);
		static Expression ^std_dim(Expression ^x, array<int> ^d, bool b);
		static Expression ^std_batches(Expression ^x);
		static Expression ^moment_elems(Expression ^x, int r);
		static Expression ^moment_dim(Expression ^x, array<int> ^d, int r, bool b);
		static Expression ^moment_batches(Expression ^x, int r);
		static Expression ^fold_rows(Expression ^x);
		static Expression ^fold_rows(Expression ^x, int nrows);
		static Expression ^pairwise_rank_loss(Expression ^x, Expression ^y);
		static Expression ^pairwise_rank_loss(Expression ^x, Expression ^y, float m);
		static Expression ^poisson_loss(Expression ^x, int py);
		static Expression ^huber_distance(Expression ^x, Expression ^y);
		static Expression ^huber_distance(Expression ^x, Expression ^y, float c);
		static Expression ^kmax_pooling(Expression ^x, int k);
		static Expression ^kmax_pooling(Expression ^x, int k, int d);
		static Expression ^pickneglogsoftmax(Expression ^x, int v);
		static Expression ^pickneglogsoftmax_batch(Expression ^x, array<int> ^v);
		static Expression ^hinge(Expression ^x, int v);
		static Expression ^hinge(Expression ^x, int v, float m);
		static Expression ^hinge_batch(Expression ^x, array<int> ^v);
		static Expression ^hinge_batch(Expression ^x, array<int> ^v, float m);
		static Expression ^hinge_dim(Expression ^x, array<int> ^v);
		static Expression ^hinge_dim(Expression ^x, array<int> ^v, int d, float m);
		static Expression ^kmh_ngram(Expression ^x, int v);
		static Expression ^pick_range(Expression ^x, int s, int e);
		static Expression ^pick_range(Expression ^x, int s, int e, int d);
		static Expression ^pickrange(Expression ^x, int s, int e);
		static Expression ^strided_select(Expression ^x, array<int> ^strides, array<int> ^range_from, array<int> ^range_to);
		static Expression ^noise(Expression ^x, float stddev);
		static Expression ^dropout(Expression ^x, float p);
		static Expression ^dropout_batch(Expression ^x, float p);
		static Expression ^dropout_dim(Expression ^x, int d, float p);
		static Expression ^block_dropout(Expression ^x, float p);
		static Expression ^reshape(Expression ^x, array<long> ^d);
		static Expression ^reshape(Expression ^x, array<long> ^d, int batchSize);
		static Expression ^max_dim(Expression ^x);
		static Expression ^max_dim(Expression ^x, int d);
		static Expression ^min_dim(Expression ^x);
		static Expression ^min_dim(Expression ^x, int d);
		static Expression ^contract3d_1d(Expression ^x, Expression ^y);
		static Expression ^logsumexp(List<Expression^> ^l);
		static Expression ^logsumexp(... array<Expression^> ^arr);
		static Expression ^logsumexp_dim(Expression ^x);
		static Expression ^logsumexp_dim(Expression ^x, int d);
		static Expression ^concatenate_cols(List<Expression ^> ^arr);
		static Expression ^concatenate_cols(... array<Expression ^> ^arr);
		static Expression ^concatenate(List<Expression ^> ^arr);
		static Expression ^concatenate(List<Expression ^> ^arr, int d);
		static Expression ^concatenate(... array<Expression ^> ^arr);
		static Expression ^concatenate(array<Expression ^> ^arr, int d);
		static Expression ^concatenate_to_batch(List<Expression ^> ^arr);
		static Expression ^concatenate_to_batch(... array<Expression ^> ^arr);
		static Expression ^affine_transform(List<Expression ^> ^arr);
		static Expression ^affine_transform(... array<Expression ^> ^arr);
		static Expression ^layer_norm(Expression ^x, Expression ^g, Expression ^b);
		static Expression ^weight_norm(Expression ^w, Expression ^g);
		static Expression ^round(Expression ^x, GradientMode gm);
		static Expression ^ceil(Expression ^x, GradientMode gm);
		static Expression ^floor(Expression ^x, GradientMode gm);
		static void ResetRandomSeed(unsigned seed);
		static void ShowPoolMemInfo();
		static size_t HowMuchMemoryDynet();
		static size_t HowMuchUsedMemoryDynet();
		static List<String ^> ^GetListOfAvailableDevices();
		static DeviceInfo ^GetDeviceInfo(String ^name);
		static Expression ^ToDevice(Expression ^e, String ^device);
		static void ResetRandomSeed(int seed);
		static void PrintCGGraphViz() {
			ExceptionWrap(cg->print_graphviz();)
		}
	};
	dynet::Device *str2dev(String ^name) {
		if (name->Equals("") || name->Equals("default"))
			return default_device;
		char *str = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(name).ToPointer();
		dynet::Device *d = get_device_manager()->get_global_device(str);
		free(str);
		return d;
	}
	// Private functions only for use here:
	dynet::Dim ConvertArrToDim(array<long>^ arr) {
		ExceptionWrap(
			return dynet::Dim(ConvertArrayToVector<long>(arr));
		)
	}
	array<long> ^ConvertDimToArr(dynet::Dim d) {
		ExceptionWrap(
			// Return an array of ints
			array<long> ^ret = gcnew array<long>(d.ndims());
		// Add them in
		for (unsigned iDim = 0; iDim < d.ndims(); iDim++)
			ret[iDim] = d[iDim];
		return ret;
		)
	}
	std::vector<unsigned int> VecToUInt(std::vector<int> vec) {
		std::vector<unsigned int> ret(vec.begin(), vec.end());
		return ret;
	}
	template<typename T>
	array<T> ^ConvertVectorToArray(std::vector<T> vec) {
		ExceptionWrap(
			array<T> ^ret = gcnew array<T>((int)vec.size());
		for (int iItem = 0; iItem < vec.size(); iItem++)
			ret[iItem] = vec[iItem];
		return ret;
		)
	}

	template<typename T>
	std::vector<T> ConvertArrayToVector(array<T> ^arr) {
		ExceptionWrap(
			std::vector<T> ret(arr->Length);
		for (int iItem = 0; iItem < arr->Length; iItem++)
			ret[iItem] = arr[iItem];
		return ret;
		)
	}

	std::vector<dynet::Expression> GetDyExpVector(List<Expression ^> ^l) {
		ExceptionWrap(
			return GetDyExpVector(l->ToArray());
		)
	}
	std::vector<dynet::Expression> GetDyExpVector(array<Expression ^> ^arr) {
		ExceptionWrap(
			std::vector<dynet::Expression> vec(arr->Length);
		for (int iItem = 0; iItem < arr->Length; iItem++)
			vec[iItem] = arr[iItem]->__thisptr;
		return vec;
		)
	}
	List<Expression ^> ^GetManagedExpList(std::vector<dynet::Expression> vec) {
		ExceptionWrap(
			List<Expression ^> ^ret = gcnew List<Expression ^>((int)vec.size());
		for (dynet::Expression exp : vec)
			ret->Add(gcnew Expression(exp));
		return ret;
		)
	}
	array<Expression ^> ^GetManagedExpArr(std::vector<dynet::Expression> vec) {
		return GetManagedExpList(vec)->ToArray();
	}
	List<List<Parameter ^> ^> ^GetParameterListOfLists(std::vector<std::vector<dynet::Parameter>> vec) {
		ExceptionWrap(
			// Create the return
			List<List<Parameter ^> ^> ^ret = gcnew List<List<Parameter ^> ^>((int)vec.size());
		// Go through
		for (auto paramL : vec) {
			List<Parameter ^> ^curL = gcnew List<Parameter ^>((int)paramL.size());
			for (auto param : paramL)
				curL->Add(gcnew Parameter(param));
			ret->Add(curL);
		}
		return ret;
		)
	}
	List<List<Expression ^> ^> ^GetParameterExpressionListOfLists(std::vector<std::vector<dynet::Expression>> vec) {
		ExceptionWrap(
			// Create the return
			List<List<Expression ^> ^> ^ret = gcnew List<List<Expression ^> ^>((int)vec.size());
		// Go through
		for (auto paramL : vec) {
			List<Expression ^> ^curL = gcnew List<Expression ^>((int)paramL.size());
			for (auto param : paramL)
				curL->Add(gcnew Expression(param));
			ret->Add(curL);
		}
		return ret;
		)
	}
	void ResetDynetMemory(size_t newMemSize) {
		ExceptionWrap(
			if (default_device->type == dynet::DeviceType::CPU) {
				std::cerr << "[dynet] re-allocating memory: " << newMemSize << "MB\n";
				size_t newSize = newMemSize / 4;

				Device_CPU *cpuD = (Device_CPU *)default_device;
				// Clear the forward & backward & scratch memory (*not* parameter)
				delete cpuD->pools[0];
				delete cpuD->pools[1];
				delete cpuD->pools[3];
				// Set the new ones
				cpuD->pools[0] = new AlignedMemoryPool("CPU forward memory", (newSize << 20), &cpuD->cpu_mem);
				cpuD->pools[1] = new AlignedMemoryPool("CPU backward memory", (newSize << 20), &cpuD->cpu_mem);
				//cpuD->pools[2] = new AlignedMemoryPool("CPU parameter memory", (newSize << 20), cpuD->shmem);
				cpuD->pools[3] = new AlignedMemoryPool("CPU scratch memory", (newSize << 20), &cpuD->cpu_mem);
				std::cerr << "[dynet] memory re-allocation done.\n";
			}// end of reallocating
			else {
				std::cerr << "Default device not CPU, cannot reallocate memory";
			}
		)
	}// end of reset dynet memory
}
