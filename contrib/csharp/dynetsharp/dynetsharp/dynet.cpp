#include "dynet.h"

namespace dynetsharp {
	///////////////////////////////////////////////////////////
	//////////////// Expression Class /////////////////////////
	///////////////////////////////////////////////////////////
	Expression ^Expression::_multiply(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr * other->__thisptr);
		)
	}
	Expression ^Expression::_subtract(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr - other->__thisptr);
		)
	}
	Expression ^Expression::_divide(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr / other->__thisptr);
		)
	}
	Expression ^Expression::_add(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr + other->__thisptr);
		)
	}
	Expression ^Expression::_multiply(float other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr * dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_subtract(float other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr - dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_divide(float other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr / dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_add(float other) {
		ExceptionWrap(
			return gcnew Expression(__thisptr + dynet::input(*cg, other));
		)
	}
	Expression ^Expression::operator*(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) * x->__thisptr);
		)
	}
	Expression ^Expression::operator-(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) - x->__thisptr);
		)
	}
	Expression ^Expression::operator/(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) / x->__thisptr);
		)
	}
	Expression ^Expression::operator+(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) + x->__thisptr);
		)
	}
	// Private function
	void Expression::GetValue() {
		ExceptionWrap(
			val = new dynet::Tensor(cg->get_value(__thisptr));
		)
	}
	/// <summary>
	/// <para>DYNET handles changing inputs keeping pointers to memoty locations. 
	/// If you initialized this object with a float value, you can set a new value</para>
	/// </summary>
	void Expression::SetValue(float newVal) {
		if (!fHasFloatVal)
			throw gcnew Exception("Cannot set float value for Expression not initialized with `input`");
		ExceptionWrap(
			*this->floatVal = newVal;
		)
	}
	/// <summary>
	/// <para>DYNET handles changing inputs keeping pointers to memoty locations. 
	/// If you initialized this object with a float array, you can set a new value</para>
	/// </summary>
	void Expression::SetValue(array<float> ^newVal) {
		if (!fHasFloatArrVal)
			throw gcnew Exception("Cannot set float value for Expression not initialized with `input`");
		ExceptionWrap(
			// Clear
			this->floatArrVal->clear();
		// Update
		int max = newVal->Length;
		for (int iVal = 0; iVal < max; iVal++)
			this->floatArrVal->push_back((real)newVal[iVal]);
		)
	}
	/// <summary>
	/// <para>DYNET handles changing inputs keeping pointers to memoty locations. 
	/// If you initialized this object with a float array, you can set a new value</para>
	/// </summary>
	void Expression::SetValue(array<array<float> ^> ^newVal) {
		if (!fHasFloatArrVal)
			throw gcnew Exception("Cannot set float value for Expression not initialized with `input`");
		ExceptionWrap(
			// Clear
			this->floatArrVal->clear();
			// Update
			int max = newVal->Length;
			for (int iVec = 0; iVec < newVal->Length; iVec++)
				for (int iItem = 0; iItem < newVal[iVec]->Length; iItem++)
					this->floatArrVal->push_back((real)newVal[iVec][iItem]);
		)
	}
	/// <summary>
	/// <para>This runs incremental forward on the entire graph every time it's called.</para>
	/// </summary>
	void Expression::Forward() {
		ExceptionWrap(
			val = new dynet::Tensor(cg->forward(__thisptr));
		)
	}
	/// <summary>
	/// <para>This runs incremental forward on the entire graph every time it's called.</para>
	/// </summary>
	void Expression::IncrementalForward() {
		ExceptionWrap(
			val = new dynet::Tensor(cg->incremental_forward(__thisptr));
		)
	}
	/// <summary>
	/// <para>This runs the backward pass based on this expression</para>
	/// </summary>
	void Expression::Backward() {
		ExceptionWrap(
			cg->backward(__thisptr, false);
		)
	}
	/// <summary>
	/// <para>This runs the backward pass based on this expression</para>
	/// <para>Turn `full` on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.</para>
	/// </summary>
	/// <param name='full'>Flag whether to compute all gradients (including with respect to constant nodes).</param>
	void Expression::Backward(bool full) {
		ExceptionWrap(
			cg->backward(__thisptr, full);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a scalar.</para>
	/// <remarks>This only works if the expression is a scalar</remarks>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	float Expression::ScalarValue(bool fRecalculate) {
		ExceptionWrap(
			if (fRecalculate) Forward();
		if (val == NULL) GetValue();
		return as_scalar(*val);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as an array (vector).</para>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	array<float> ^Expression::VectorValue(bool fRecalculate) {
		ExceptionWrap(
			if (fRecalculate) Forward();
			if (val == NULL) GetValue();
			// Get the vector, convert to array
			std::vector<float> vec = as_vector(*val);
			return ConvertVectorToArray<float>(vec);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	Tensor ^Expression::TensorValue(bool fRecalculate) {
		ExceptionWrap(
			if (fRecalculate) Forward();
			if (val == NULL) GetValue();
			return gcnew Tensor(*val);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a scalar.</para>
	/// <remarks>This only works if the expression is a scalar</remarks>
	/// </summary>
	float Expression::ScalarValue() {
		ExceptionWrap(
			return ScalarValue(false);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as an array (vector).</para>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks>
	/// </summary>
	array<float> ^Expression::VectorValue() {
		ExceptionWrap(
			return VectorValue(false);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// </summary>
	Tensor ^Expression::TensorValue() {
		ExceptionWrap(
			return TensorValue(false);
		)
	}
	/// <summary> 
	/// <para>Returns an array of the dimensions</para>
	/// </summary>
	Dim ^Expression::Shape() {
		ExceptionWrap(
			return gcnew Dim(__thisptr.dim());
		)
	}
	/// <summary>
	/// <para>Returns the value of the gradient as a Tensor object</para>
	/// <remarks>Make sure to call `backward` on a downstream expression before calling this.</remarks><para/>
	/// <remarks>If the Expression is a constant expression(meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument `full: True` to `backward`</remarks>
	/// </summary>
	Tensor ^Expression::Gradient() {
		ExceptionWrap(
			return gcnew Tensor(__thisptr.gradient());
		)
	}
	/// <summary>
	/// <para>Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor. This will result in the dimension of the tensor being reduced by 1.</para>
	/// <remarks>Equivalent to: dy.pick(exp, index);</remarks>
	/// </summary>
	/// <param name='index'>Index to pick</param>
	Expression ^Expression::default::get(int index) {
		ExceptionWrap(
			return DynetFunctions::pick(this, index);
		)
	}
	/// <summary>
	/// <para>Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// <remarks>Equivalent to: dy.pick(exp, index, dim);</remarks>
	/// </summary>
	/// <param name='index'>Index to pick</param>
	/// <param name='dim'>Index of dimension to pick from, default 0</param>
	Expression ^Expression::default::get(int index, int dim) {
		ExceptionWrap(
			return DynetFunctions::pick(this, index, dim);
		)
	}
	/// <summary>
	/// <para>Minibatch - Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// <remarks>Equivalent to: dy.pick(exp, indexes);</remarks>
	/// </summary>
	/// <param name='indexes'>Array of indexes to pick</param>
	Expression ^Expression::default::get(array<int> ^indexes) {
		ExceptionWrap(
			return DynetFunctions::pick_batch(this, indexes);
		)
	}
	/// <summary>
	/// <para>Minibatch - Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// <remarks>Equivalent to: dy.pick(exp, indexes, dim);</remarks>
	/// </summary>
	/// <param name='indexes'>Array of indexes to pick</param>
	/// <param name='dim'>Index of dimension to pick from, default 0</param>
	Expression ^Expression::default::get(array<int> ^indexes, int dim) {
		ExceptionWrap(
			return DynetFunctions::pick_batch(this, indexes, dim);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	////////////////// Tensor Functions ///////////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Initialize the Tensor object with a flattened vector and shape</para>
	/// </summary>
	/// <param name='arr'>Values of the flattened vector</param>
	/// <param name='shape'>Shape of the Tensor</param>
	Tensor::Tensor(array<float> ^arr, array<long> ^shape) {
		ExceptionWrap(
			_vec = new std::vector<float>(ConvertArrayToVector<float>(arr));
		_dim = gcnew Dim(shape);
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 1-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 1-dim vector</param>
	Tensor::Tensor(array<float> ^arr) {
		ExceptionWrap(
		_dim = gcnew Dim(std::vector<long>({ arr->Length }));
		_vec = new std::vector<float>(arr->Length);
		// Populate
		int pos = 0;
		for (int i = 0; i < arr->Length; i++)
			(*_vec)[pos++] = arr[i];
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 2-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 2-dim vector</param>
	Tensor::Tensor(array<array<float> ^> ^arr) {
		ExceptionWrap(
		_dim = gcnew Dim(std::vector<long>({ arr->Length, arr[0]->Length }));
		_vec = new std::vector<float>(arr->Length * arr[0]->Length);
		// Populate
		int pos = 0;
		if (arr->Length > 0) {
			for (int j = 0; j < arr[0]->Length; j++)
				for (int i = 0; i < arr->Length; i++)
					(*_vec)[pos++] = arr[i][j];
		}
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 3-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 3-dim vector</param>
	Tensor::Tensor(array<array<array<float> ^> ^> ^arr) {
		ExceptionWrap(
			_dim = gcnew Dim(std::vector<long>({ arr->Length, arr[0]->Length, arr[0][0]->Length }));
			_vec = new std::vector<float>(arr->Length * arr[0]->Length * arr[0][0]->Length);
			// Populate
			int pos = 0;
			for (int k = 0; k < arr[0][0]->Length; k++) {
				for (int j = 0; j < arr[0]->Length; j++)
					for (int i = 0; i < arr->Length; i++)
						(*_vec)[pos++] = arr[i][j][k];
			}
			__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 4-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 4-dim vector</param>
	Tensor::Tensor(array<array<array<array<float> ^> ^> ^> ^arr) {
		ExceptionWrap(
			_dim = gcnew Dim(std::vector<long>({ arr->Length, arr[0]->Length, arr[0][0]->Length, arr[0][0][0]->Length }));
		_vec = new std::vector<float>(arr->Length * arr[0]->Length * arr[0][0]->Length * arr[0][0][0]->Length);
		// Populate
		int pos = 0;
		for (int l = 0; l < arr[0][0][0]->Length; l++)
			for (int k = 0; k < arr[0][0]->Length; k++)
				for (int j = 0; j < arr[0]->Length; j++)
					for (int i = 0; i < arr->Length; i++)
						(*_vec)[pos++] = arr[i][j][k][l];
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Set whether this tensor is batched, if so it moves the last dim to the batch</para>
	/// <para>Only works if generated from user code, and not from dynet tensor</para>
	/// </summary>
	void Tensor::SetBatched(bool fBatched) {
		ExceptionWrap(
			if (__thisptr != NULL)
				throw gcnew Exception(gcnew String("Cannot call SetBatched function on a dynet::Tensor object."));
		// Already batched?
		if (fBatched && _dim->BatchSize != 1)
			return;
		// Already not batched?
		if (!fBatched && _dim->BatchSize == 1)
			return;
		// Move it over
		dynet::Dim d = _dim->get_cdim();
		if (!fBatched) {
			d.add_dim(d.batch_elems());
			d.bd = 1;
			_dim = gcnew Dim(d);
		}
		if (fBatched) {
			if (d.ndims() == 1)
				_dim = gcnew Dim({ 1 }, d[0]);
			else {
				d.bd = d[d.ndims() - 1];
				d.delete_dim(d.ndims() - 1);
				_dim = gcnew Dim(d);
			}
		}
		)
	}
	/// <summary>
	/// <para>Returns the value of the Tensor as a flattened 1-dimensional array (vector)</para>
	/// <para>In case of a multidimensional expression, the values are flattened according to a column major ordering</para>
	/// </summary>
	array<float> ^Tensor::GetFlatVector() {
		ExceptionWrap(
			return ConvertVectorToArray<float>(*_vec);
		)
	}
	/// <summary>
	/// <para>Returns a 1-dimensional array of the Tensor</para>
	/// <remarks>This only works if the Tensor has 1 dimension. Otherwise throws an exception.</remarks>
	/// </summary>
	array<float> ^Tensor::Get1DVector() {
		ExceptionWrap(
			if (ndims != 1)
				throw gcnew Exception(gcnew String((std::string("Dimension mismatch. Cannot return 1-Dimensional vector for shape with ") + std::to_string(ndims) + " dims").c_str()));
			return ConvertVectorToArray<float>(*_vec);
		)
	}
	/// <summary>
	/// <para>Returns a 2-dimensional array of the Tensor</para>
	/// <remarks>This only works if the Tensor has 2 dimensions. Otherwise throws an exception.</remarks>
	/// </summary>
	array<array<float>^> ^Tensor::Get2DVector() {
		ExceptionWrap(
			if (ndims != 2)
				throw gcnew Exception(gcnew String((std::string("Dimension mismatch. Cannot return 2-Dimensional vector for shape with ") + std::to_string(ndims) + " dims").c_str()));
		
		array<long> ^dims = _dim->Dims;
		// Return a 2-dimensional vector
		array<array<float> ^> ^ret = gcnew array<array<float> ^>(dims[0]);

		// Create the output
		int curInd = 0;
		for (int j = 0; j < dims[1]; j++) {
			for (int i = 0; i < dims[0]; i++) {
				if (j == 0) ret[i] = gcnew array<float>(dims[1]);
				ret[i][j] = (*_vec)[curInd++];
			}
		}//next j

		return ret;
		)
	}
	/// <summary>
	/// <para>Returns a 3-dimensional array of the Tensor</para>
	/// <remarks>This only works if the Tensor has 3 dimensions. Otherwise throws an exception.</remarks>
	/// </summary>
	array<array<array<float>^>^> ^Tensor::Get3DVector() {
		ExceptionWrap(
			if (ndims != 3)
				throw gcnew Exception(gcnew String((std::string("Dimension mismatch. Cannot return 3-Dimensional vector for shape with ") + std::to_string(ndims) + " dims").c_str()));
		
		array<long> ^dims = _dim->Dims;
		// Return a 3-dimensional vector
		array<array<array<float> ^> ^> ^ret = gcnew array<array<array<float> ^> ^>(dims[0]);
		// Create the output
		int curInd = 0;
		for (int k = 0; k < dims[2]; k++) {
			for (int j = 0; j < dims[1]; j++) {
				for (int i = 0; i < dims[0]; i++) {
					if (j == 0 && k == 0) ret[i] = gcnew array<array<float> ^>(dims[1]);
					if (k == 0) ret[i][j] = gcnew array<float>(dims[2]);
					ret[i][j][k] = (*_vec)[curInd++];
				}
			}//next j
		}//next k

		return ret;
		)
	}
	/// <summary>
	/// <para>Returns a 4-dimensional array of the Tensor</para>
	/// <remarks>This only works if the Tensor has 4 dimensions. Otherwise throws an exception.</remarks>
	/// </summary>
	array<array<array<array<float> ^>^>^> ^Tensor::Get4DVector() {
		ExceptionWrap(
			if (ndims != 4)
				throw gcnew Exception(gcnew String((std::string("Dimension mismatch. Cannot return 4-Dimensional vector for shape with ") + std::to_string(ndims) + " dims").c_str()));

		array<long> ^dims = _dim->Dims;
		// Return a 4-dimensional vector
		array<array<array<array<float> ^> ^> ^> ^ret = gcnew array<array<array<array<float> ^> ^> ^>(dims[0]);
		// Create the output
		int curInd = 0;
		for (int l = 0; l < dims[3]; l++) {
			for (int k = 0; k < dims[2]; k++) {
				for (int j = 0; j < dims[1]; j++) {
					for (int i = 0; i < dims[0]; i++) {
						if (j == 0 && k == 0 && l == 0) ret[i] = gcnew array<array<array<float> ^> ^>(dims[1]);
						if (k == 0 && l == 0) ret[i][j] = gcnew array<array<float> ^>(dims[2]);
						if (l == 0) ret[i][j][k] = gcnew array<float>(dims[3]);
						ret[i][j][k][l] = (*_vec)[curInd++];
					}
				}//next j
			}//next k
		}//next l

		return ret;
		)
	}
	// Private functions:
	int Tensor::GetActualPosFromArr(array<int> ^arr) {
		ExceptionWrap(
			return GetActualPosFromArr(ConvertArrayToVector<int>(arr));
		)
	}
	int Tensor::GetActualPosFromArr(std::vector<int> vec) {
		ExceptionWrap(
			return GetActualPosFromArr(vec, false);
		)
	}
	int Tensor::GetActualPosFromArr(std::vector<int> vec, bool fCheckBoundaries) {
		ExceptionWrap(
			dynet::Dim dim = _dim->get_cdim();
			// Find the position - go through each dimension, and multiply by the index
			int actualPos = vec[0];
		for (int iDim = 1; iDim < vec.size(); iDim++) {
			int curPos = vec[iDim];
			// Check if we are past the end
			if (fCheckBoundaries && curPos >= dim[iDim])
				throw gcnew IndexOutOfRangeException();

			// Multiply by all following dimensions
			for (int jDim = iDim - 1; jDim < iDim; jDim++)
				curPos *= dim[jDim];
			// Add it in
			actualPos += curPos;
		}
		return actualPos;
		)
	}
	/// <summary>
	/// <para>Set the scalar value of a single cell</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact cell, each position in the array referring to a dimension</param>
	/// <param name='value'>New value of the cell</param>
	void Tensor::SetValue(array<int> ^pos, float value) {
		ExceptionWrap(
			if (pos->Length != ndims)
				throw gcnew Exception(gcnew String((std::string("Dimensions mismatch. Assumed to have ") + std::to_string(pos->Length) + " dims, when in fact has " + std::to_string(ndims) + ".").c_str()));
		int actualPos = GetActualPosFromArr(pos);
		(*_vec)[actualPos] = value;
		)
	}
	/// <summary>
	/// <para>Get the scalar value of a single cell</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact cell, each position in the array referring to a dimension</param>
	float Tensor::GetValue(array<int> ^pos) {
		ExceptionWrap(
			if (pos->Length != ndims)
				throw gcnew Exception(gcnew String((std::string("Dimensions mismatch. Assumed to have ") + std::to_string(pos->Length) + " dims, when in fact has " + std::to_string(ndims) + ".").c_str()));
		int actualPos = GetActualPosFromArr(pos);
		return (*_vec)[actualPos];
		)
	}
	/// <summary>
	/// <para>Set the vector(array) value of a single row (last dimension)</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact row, each position in the array referring to a dimension</param>
	/// <param name='value'>New value of the row</param>
	void Tensor::SetRowValue(array<int> ^pos, array<float> ^value) {
		ExceptionWrap(
			SetRowValue(pos, value, 0);
		)
	}
	/// <summary>
	/// <para>Set the vector(array) value of a single row (last dimension)</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact row, each position in the array referring to a dimension</param>
	/// <param name='value'>New value of the row</param>
	/// <param name='batchDim'>Batch size (default: 0)</param>
	void Tensor::SetRowValue(array<int> ^pos, array<float> ^value, int batchDim) {
		ExceptionWrap(
			if (batchDim > _dim->BatchSize)
				throw gcnew Exception(gcnew String("Batch size out of range"));
			if (pos->Length != _dim->NDims)// use actual function, to avoid addition for batch
				throw gcnew Exception(gcnew String((std::string("Dimensions mismatch. Assumed to have ") + std::to_string(pos->Length + 1) + " dims, when in fact has " + std::to_string(_dim->NDims) + ".").c_str()));

		dynet::Dim dim = _dim->get_cdim();

		int actualPos = GetActualPosFromArr(pos) + batchDim;
		int jump = 1;
		for (int jdim = 0; jdim < pos->Length; jdim++)
			jump *= dim[jdim];

		// Copy in each value
		for (int iVal = 0; iVal < value->Length; iVal++) {
			(*_vec)[actualPos] = value[iVal];
			actualPos += jump;
		}
		)
	}
	/// <summary>
	/// <para>Get the vector(array) value of a single row (last dimension)</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact row, each position in the array referring to a dimension</param>
	array<float> ^Tensor::GetRowValue(array<int> ^pos) {
		ExceptionWrap(
			if (pos->Length != ndims - 1)
				throw gcnew Exception(gcnew System::String((std::string("Dimensions mismatch. Assumed to have ") + std::to_string(pos->Length + 1) + " dims, when in fact has " + std::to_string(ndims) + ".").c_str()));
		dynet::Dim dim = _dim->get_cdim();

		int actualPos = GetActualPosFromArr(pos);
		int jump = 1;
		for (int jdim = 0; jdim < pos->Length; jdim++)
			jump *= dim[jdim];

		std::vector<float> curVec;
		// Copy in each value
		for (int iVal = 0; iVal < dim[ndims - 1]; iVal++) {
			curVec.push_back((*_vec)[actualPos]);
			actualPos += jump;
		}
		return ConvertVectorToArray<float>(curVec);
		)
	}

	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	////////////////// ParameterCollection ////////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// </summary>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	Parameter ^ParameterCollection::AddParameters(array<long>^ dim) {
		ExceptionWrap(
			return AddParameters(dim, gcnew GlorotInitializer(), "");
		)
	}
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// </summary>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Parameter ^ParameterCollection::AddParameters(array<long>^ dim, String ^device) {
		ExceptionWrap(
			return AddParameters(dim, gcnew GlorotInitializer(), device);
		)
	}
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// <remarks>List of possible initializers: NormalInitializer, UniformInitializer, ConstInitializer, IdentityInitializer, GlorotInitializer, SaxeInitializer, FromFileInitializer, FromVectorInitializer</remarks>
	/// </summary>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='pi'>Initializer to use when initializing the parameter. e.g,. GlorotInitializer</param>
	Parameter ^ParameterCollection::AddParameters(array<long>^ dim, ParamInitializer ^pi) {
		ExceptionWrap(
			return AddParameters(dim, pi, "");
		)
	}
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// <remarks>List of possible initializers: NormalInitializer, UniformInitializer, ConstInitializer, IdentityInitializer, GlorotInitializer, SaxeInitializer, FromFileInitializer, FromVectorInitializer</remarks>
	/// </summary>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='pi'>Initializer to use when initializing the parameter. e.g,. GlorotInitializer</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Parameter ^ParameterCollection::AddParameters(array<long>^ dim, ParamInitializer ^pi, String ^device) {
		ExceptionWrap(
			return gcnew Parameter(__thisptr->add_parameters(ConvertArrToDim(dim), *pi->__thisptr, "",str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Add a lookup parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// </summary>
	/// <param name='size'>Number of rows for the collection</param>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	LookupParameter ^ParameterCollection::AddLookupParameters(int size, array<long>^ dim) {
		ExceptionWrap(
			return AddLookupParameters(size, dim, gcnew GlorotInitializer(), "");
		)
	}
	/// <summary>
	/// <para>Add a lookup parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// </summary>
	/// <param name='size'>Number of rows for the collection</param>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	LookupParameter ^ParameterCollection::AddLookupParameters(int size, array<long>^ dim, String ^device) {
		ExceptionWrap(
			return AddLookupParameters(size, dim, gcnew GlorotInitializer(), device);
		)
	}
	/// <summary>
	/// <para>Add a lookup parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// <remarks>List of possible initializers: NormalInitializer, UniformInitializer, ConstInitializer, IdentityInitializer, GlorotInitializer, SaxeInitializer, FromFileInitializer, FromVectorInitializer</remarks>
	/// </summary>
	/// <param name='size'>Number of rows for the collection</param>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='pi'>Initializer to use when initializing the parameter. e.g,. GlorotInitializer</param>
	LookupParameter ^ParameterCollection::AddLookupParameters(int size, array<long>^ dim, ParamInitializer ^pi) {
		ExceptionWrap(
			return AddLookupParameters(size, dim, pi, "");
		)
	}
	/// <summary>
	/// <para>Add a lookup parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// <remarks>List of possible initializers: NormalInitializer, UniformInitializer, ConstInitializer, IdentityInitializer, GlorotInitializer, SaxeInitializer, FromFileInitializer, FromVectorInitializer</remarks>
	/// </summary>
	/// <param name='size'>Number of rows for the collection</param>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	/// <param name='pi'>Initializer to use when initializing the parameter. e.g,. GlorotInitializer</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	LookupParameter ^ParameterCollection::AddLookupParameters(int size, array<long>^ dim, ParamInitializer ^pi, String ^device) {
		ExceptionWrap(
			return gcnew LookupParameter(__thisptr->add_lookup_parameters(size, ConvertArrToDim(dim), *pi->__thisptr, "", str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Returns a list of all the Parameter objects in this collection</para>
	/// </summary>
	List<Parameter ^> ^ParameterCollection::GetParametersList() {
		ExceptionWrap(
			int totalParams = (int)__thisptr->parameters_list().size();
		List<Parameter ^> ^ret = gcnew List<Parameter ^>(totalParams);
		for (dynet::Parameter param : __thisptr->parameters_list())
			ret->Add(gcnew Parameter(param));
		return ret;
		)
	}
	/// <summary>
	/// <para>Returns a list of all the LookupParameter objects in this collection</para>
	/// </summary>
	List<LookupParameter ^> ^ParameterCollection::GetLookupParametersList() {
		ExceptionWrap(
			int totalParams = (int)__thisptr->lookup_parameters_list().size();
		List<LookupParameter ^> ^ret = gcnew List<LookupParameter ^>(totalParams);
		for (dynet::LookupParameter param : __thisptr->lookup_parameters_list())
			ret->Add(gcnew LookupParameter(param));
		return ret;
		)
	}
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, initializing with defined values</para>
	/// </summary>
	/// <param name='t'>Values to initialize parameter with</param>
	Parameter ^ParameterCollection::AddParametersFromTensor(Tensor ^t) {
		ExceptionWrap(
			return AddParametersFromTensor(t, "");
		)
	}
	/// <summary>
	/// <para>Add a parameter to the ParameterCollection, initializing with defined values</para>
	/// </summary>
	/// <param name='t'>Values to initialize parameter with</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Parameter ^ParameterCollection::AddParametersFromTensor(Tensor ^t, String ^device) {
		ExceptionWrap(
			dynet::Dim dim = t->Shape()->get_cdim();;
			return gcnew Parameter(__thisptr->add_parameters(dim, dynet::ParameterInitFromVector(*t->_vec), "", str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Add a LookupParameter to the ParameterCollection, initializing with defined values</para>
	/// </summary>
	/// <param name='t'>Values to initialize parameter with</param>
	LookupParameter ^ParameterCollection::AddLookupParametersFromTensor(Tensor ^t) {
		ExceptionWrap(
			return AddLookupParametersFromTensor(t, "");
		)
	}
	/// <summary>
	/// <para>Add a LookupParameter to the ParameterCollection, initializing with defined values</para>
	/// </summary>
	/// <param name='t'>Values to initialize parameter with</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	LookupParameter ^ParameterCollection::AddLookupParametersFromTensor(Tensor ^t, String ^device) {
		ExceptionWrap(
			dynet::Dim dim = t->Shape()->get_cdim();
			int lp1 = dim[0];
			dim.delete_dim(0);
			return gcnew LookupParameter(__thisptr->add_lookup_parameters(lp1, dim, dynet::ParameterInitFromVector(*t->_vec), "", str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Creates a sub-collection of the current collection, and returns it.</para>
	/// <para>A sub-collection is simply a ParameterCollection object which is tied to a parent collection.ParameterCollections can be nested to arbitraty depth.</para>
	/// <para>Sub-collections are used for grouping of parameters, for example if one wants to train only a subset of the parameters, one can add them in a subcollection and pass the subcollection to a trainer. Similarly, for saving (or loading) only some of the parameters, one can save/populate a sub-collection.</para>
	/// <para>Sub-collections are used inside builder objects (such as the LSTMBuilder): The builder creates a local sub-collection and adds parameters to it instead of to the global collection that is passed to it in the constructor. This way, the parameters participating in the builder are logically grouped, and can be saved/loaded/trained seperately if needed.</para>
	/// </summary>
	ParameterCollection ^ParameterCollection::AddSubCollection() {
		ExceptionWrap(
			return gcnew ParameterCollection(__thisptr->add_subcollection());
		)
	}
	/// <summary>
	/// <para>Get the weight decay lambda value.</para>
	/// </summary>
	float ParameterCollection::GetWeightDecay() {
		ExceptionWrap(
			return __thisptr->get_weight_decay_lambda();
		)
	}
	/// <summary>
	/// <para>Set the weight decay coefficient.</para>
	/// </summary>
	/// <param name='lam'>Weight decay coefficient</param>
	void ParameterCollection::SetWeightDecay(float lam) {
		ExceptionWrap(
			__thisptr->set_weight_decay_lambda(lam);
		)
	}
	/// <summary>
	/// <para>Set the weight decay coefficient.</para>
	/// <remarks>Alias to SetWeightDecay(lam)</remarks>
	/// </summary>
	/// <param name='lam'>Weight decay coefficient</param>
	void ParameterCollection::SetWeightDecayLambda(float lam) {
		ExceptionWrap(
			__thisptr->set_weight_decay_lambda(lam);
		)
	}
	int ParameterCollection::GetParameterCount() {
		ExceptionWrap(
			return (int)__thisptr->parameter_count();
		)
	}


	/// <summary>
	/// <para>Saves all the parameters in the model to disk</para>
	/// </summary>
	/// <param name='filename'>Path of file to save the parameters</param>
	void ParameterCollection::Save(System::String ^filename) {
		ExceptionWrap(
			char *str = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(filename).ToPointer();
			save_dynet_model(str, __thisptr);
			free(str);
		)
	}
	/// <summary>
	/// <para>Populates all the parameters in the model from disk</para>
	/// </summary>
	/// <param name='filename'>Path of file to load the parameters</param>
	void ParameterCollection::Load(System::String ^filename) {
		ExceptionWrap(
			char *str = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(filename).ToPointer();
			load_dynet_model(str, __thisptr);
			free(str);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	//////////////////// Parameter Functions //////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Returns shape of the parameter</para>
	/// </summary>
	array<long> ^Parameter::Shape() {
		ExceptionWrap(
			return ConvertDimToArr(__thisptr->dim());
		)
	}
	/// <summary>
	/// <para>Clip the values in the parameter to a fixed range [left, right] (in place)</para>
	/// </summary>
	void Parameter::ClipInPlace(float left, float right) {
		ExceptionWrap(
			__thisptr->clip_inplace(left, right);
		)
	}
	/// <summary>
	/// <para>Set the values of the parameter to zero</para>
	/// </summary>
	void Parameter::Zero() {
		ExceptionWrap(
			__thisptr->zero();
		)
	}
	/// <summary>
	/// <para>Scales the values of the parameter by factor</para>
	/// </summary>
	/// <param name='s'>Factor to scale the parameter by</param>
	void Parameter::Scale(float s) {
		ExceptionWrap(
			__thisptr->scale(s);
		)
	}
	/// <summary>
	/// <para>Scales the gradient of the parameter by factor</para>
	/// </summary>
	/// <param name='s'>Factor to scale the gradient by</param>
	void Parameter::ScaleGradient(float s) {
		ExceptionWrap(
			__thisptr->scale_gradient(s);
		)
	}
	/// <summary>
	/// <para>Checks whether the parameter is updated or not</para>
	/// </summary>
	bool Parameter::IsUpdated() {
		ExceptionWrap(
			return __thisptr->is_updated();
		)
	}
	/// <summary>
	/// <para>Set parameter as "updated"</para>
	/// </summary>
	/// <param name='b'>New "updated" value</param>
	void Parameter::SetUpdated(bool b) {
		ExceptionWrap(
			__thisptr->set_updated(b);
		)
	}
	/// <summary>
	/// <para>Get the scalar value of this parameter</para>
	/// </summary>
	float Parameter::ScalarValue() {
		ExceptionWrap(
			return ScalarValue(false);
		)
	}
	/// <summary>
	/// <para>Get the scalar value of this parameter</para>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	float Parameter::ScalarValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->ScalarValue(fRecalculate);
		)
	}
	/// <summary>
	/// <para>Get the vector value of this parameter</para>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks><para/>
	/// </summary>
	array<float> ^Parameter::VectorValue() {
		ExceptionWrap(
			return VectorValue(false);
		)
	}
	/// <summary>
	/// <para>Get the vector value of this parameter</para>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks><para/>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	array<float> ^Parameter::VectorValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->VectorValue(fRecalculate);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// </summary>
	Tensor ^Parameter::TensorValue() {
		ExceptionWrap(
			return TensorValue(false);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	Tensor ^Parameter::TensorValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->TensorValue(fRecalculate);
		)
	}
	/// <summary>
	/// <para>Returns parameter as dynetsharp.Tensor object</para>
	/// </summary>
	Tensor ^Parameter::AsTensor() {
		ExceptionWrap(
			return gcnew Tensor(__thisptr->get_storage().values);
		)
	}
	/// <summary>
	/// <para>Returns gradient as dynetsharp.Tensor object</para>
	/// <remarks>Make sure to call `backward` on a downstream expression before calling this.</remarks>
	/// </summary>
	Tensor ^Parameter::Gradient() {
		ExceptionWrap(
			return gcnew Tensor(__thisptr->get_storage().g);
		)
	}
	/// <summary>
	/// <para>Returns the parameter as an expression</para>
	/// <remarks>Deprecated - an explicit conversion is no longer needed.</remarks>
	/// </summary>
	[Obsolete("Deprecated - an explicit conversion is no longer needed.")]
	Expression ^Parameter::ToExpression() {
		ExceptionWrap(
			return GetExpression();
		)
	}
	/// <summary>
	/// <para>Returns the parameter as an expression</para>
	/// </summary>
	/// <param name='fUpdate'>Default true, if this is set to False, the parameter won't be updated during the backward pass</param>
	Expression ^Parameter::ToExpression(bool fUpdate) {
		ExceptionWrap(
			// If the fUpdate flag is true, just call the regular function. 
			if (fUpdate) return GetExpression();
		// Otherwise, update the constant expression
		if (!__const_exp || __const_exp->IsStale())
			__const_exp = gcnew Expression(dynet::const_parameter(*cg, *__thisptr));
		return __const_exp;
		)
	}
	/// <summary>
	/// <para>Set values of the parameter</para>
	/// </summary>
	/// <param name='val'>Tensor object with the new values (with matching shape)</param>
	void Parameter::SetValue(Tensor ^val) {
		ExceptionWrap(
			// Make sure the dimensions match
			dynet::Dim dim = __thisptr->dim();
		// Check the count
		if (val->NDims() != dim.ndims())
			throw gcnew Exception(gcnew String((std::string("Shape of values and parameter don't match in Parameters.SetValue. Input: ") + std::to_string(val->NDims()) + ", Actual: " + std::to_string(dim.ndims())).c_str()));
		// Check the numbers
		for (int iDim = 0; iDim < val->NDims(); iDim++)
			if (val->Shape()->get_cdim()[iDim] != dim[iDim])
				throw gcnew Exception(gcnew String((std::string("Shape of values and parameter don't match in Parameters.SetValue, Dimension #") + std::to_string(iDim) + ", Input: " + std::to_string(val->Shape()->get_cdim()[iDim]) + ", Actual: " + std::to_string(dim[iDim])).c_str()));
		// Put it in
		__thisptr->set_value(*val->_vec);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	/////////////  LookupParameter Functions //////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// <para>Equivalent to doing dy.lookup(lp, index);</para>
	/// </summary>
	/// <param name='index'>Index of row to lookup</param>
	Expression ^LookupParameter::default::get(int index) {
		ExceptionWrap(
			return DynetFunctions::lookup(this, index);
		)
	}
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// <para>Equivalent to doing dy.lookup(lp, index, fUpdate);</para>
	/// </summary>
	/// <param name='index'>Index of row to lookup</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the returned expression won't be updated during the backward pass</param>
	Expression ^LookupParameter::default::get(int index, bool fUpdate) {
		ExceptionWrap(
			return DynetFunctions::lookup(this, index, fUpdate);
		)
	}
	/// <summary>
	/// <para>Minibatch - Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// <para>Equivalent to doing dy.lookup_batch(lp, indexes);</para>
	/// </summary>
	/// <param name='indexes'>Array of indexes of rows to lookup</param>
	Expression ^LookupParameter::default::get(array<int> ^indexes) {
		ExceptionWrap(
			return DynetFunctions::lookup_batch(this, indexes);
		)
	}
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// <para>Equivalent to doing dy.lookup_batch(lp, indexes, fUpdate);</para>
	/// </summary>
	/// <param name='indexes'>List of indexes of rows to lookup</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the returned expression won't be updated during the backward pass</param>
	Expression ^LookupParameter::default::get(array<int> ^indexes, bool fUpdate) {
		ExceptionWrap(
			return DynetFunctions::lookup_batch(this, indexes, fUpdate);
		)
	}
	/// <summary>
	/// <para>Returns shape of the LookupParameter</para>
	/// </summary>
	array<long> ^LookupParameter::Shape() {
		ExceptionWrap(
			return ConvertDimToArr(__thisptr->dim());
		)
	}
	/// <summary>
	/// <para>Returns number of records(rows) in the LookupParameter</para>
	/// </summary>
	int LookupParameter::Size() {
		ExceptionWrap(
			dynet::Dim d = __thisptr->dim();
		return d[d.ndims() - 1];
		)
	}
	/// <summary>
	/// <para>Initializes the values of a specific row</para>
	/// <remarks>Only works if there is only 1 dimension in the lookup table</remarks>
	/// </summary>
	/// <param name='iRow'>Index of row to update</param>
	/// <param name='row'>Values to initialize with</param>
	void LookupParameter::InitRow(int iRow, array<float> ^row) {
		ExceptionWrap(
			if (__thisptr->dim().ndims() > 1)
				throw gcnew Exception(gcnew String("Tried to initialize row in LookupParameter with more than one dimension"));
		__thisptr->initialize(iRow, ConvertArrayToVector<float>(row));
		)
	}
	/// <summary>
	/// <para>Initializes the values of the parameter</para>
	/// <remarks>Preferably uses ParameterCollection.AddLookupParameterFromTensor when possible</remarks>
	/// </summary>
	/// <param name='t'>Tensor of values to populate the LookupParameter with</param>
	void LookupParameter::InitFromArray(Tensor ^t) {
		ExceptionWrap(
			size_t actualRowCount = __thisptr->get_storage().values.size();
		// Check the row count
		dynet::Dim shape = t->Shape()->get_cdim();
		if (shape[0] != actualRowCount)
			throw gcnew Exception(gcnew String(("Row count mismatch when initializing lookup table from array")));
		// Check the dimensions
		if (shape[1] != __thisptr->get_storage().values[0].d.rows())
			throw gcnew Exception(gcnew String(("Dimension mismatch when initializing lookup table from array")));

		// Put in each row
		for (int iRow = 0; iRow < actualRowCount; iRow++)
			InitRow(iRow, t->GetRowValue(gcnew array<int>{ iRow }));
		)
	}
	/// <summary>
	/// <para>Return all values of the parameter as a Tensor array</para>
	/// </summary>
	array<Tensor ^> ^LookupParameter::AsTensorArray() {
		ExceptionWrap(
			array<Tensor ^> ^ret = gcnew array<Tensor ^>((int)__thisptr->get_storage().values.size());
		for (int iRow = 0; iRow < __thisptr->get_storage().values.size(); iRow++)
			ret[iRow] = RowAsTensor(iRow);
		return ret;
		)
	}
	/// <summary>
	/// <para>Return row as a Tensor</para>
	/// </summary>
	/// <param name='iRow'>Index of row to return</param>
	Tensor ^LookupParameter::RowAsTensor(int iRow) {
		ExceptionWrap(
			return gcnew Tensor(__thisptr->get_storage().values[iRow]);
		)
	}
	/// <summary>
	/// <para>Returns the gradient of a specific row as a Tensor object</para>
	/// <remarks>Make sure to call `backward` on a downstream expression before calling this.</remarks><para/>
	/// <remarks>If the Expression is a constant expression(meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument `full: True` to `backward`</remarks>
	/// </summary>
	Tensor ^LookupParameter::RowGradient(int iRow) {
		ExceptionWrap(
			return gcnew Tensor(__thisptr->get_storage().grads[iRow]);
		)
	}
	/// <summary>
	/// <para>Returns the gradient of all rows as a Tensor array</para>
	/// <remarks>Make sure to call `backward` on a downstream expression before calling this.</remarks><para/>
	/// <remarks>If the Expression is a constant expression(meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument `full: True` to `backward`</remarks>
	/// </summary>
	array<Tensor ^> ^LookupParameter::GradientArray() {
		ExceptionWrap(
			array<Tensor ^> ^ret = gcnew array<Tensor ^>((int)__thisptr->get_storage().values.size());
		for (int iRow = 0; iRow < __thisptr->get_storage().values.size(); iRow++)
			ret[iRow] = RowGradient(iRow);
		return ret;
		)
	}
	/// <summary>
	/// <para>Set the values of the LookupParameter to zero</para>
	/// </summary>
	void LookupParameter::Zero() {
		ExceptionWrap(
			__thisptr->zero();
		)
	}
	/// <summary>
	/// <para>Scales the values of the LookupParameter by factor</para>
	/// </summary>
	/// <param name='s'>Factor to scale the LookupParameter by</param>
	void LookupParameter::Scale(float s) {
		ExceptionWrap(
			__thisptr->scale(s);
		)
	}
	/// <summary>
	/// <para>Scales the gradient of the LookupParameter by factor</para>
	/// </summary>
	/// <param name='s'>Factor to scale the gradient by</param>
	void LookupParameter::ScaleGradient(float s) {
		ExceptionWrap(
			__thisptr->scale_gradient(s);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	//////////////////// RNNState Functions ///////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Gets previous RNNState</para>
	/// <para>In case you need to rewind</para>
	/// </summary>
	RNNState ^RNNState::GetPrev() {
		ExceptionWrap(
			if (!_prev)
				throw gcnew Exception(gcnew String("Cannot return prevoius state, current state is initial."));
		return _prev;
		)
	}
	/// <summary>
	/// <para>Get the current output of the RNN</para>
	/// <remarks>Usually this is the function you want to call after calling AddInput()</remarks>
	/// </summary>
	Expression ^RNNState::Output() {
		ExceptionWrap(
			ensure_freshness();
		return _out;
		)
	}
	/// <summary>
	/// <para>Perform a step within the RNN, giving it as input a given expression</para>
	/// <para>This computes h_t = RNN(x_t)</para>
	/// </summary>
	/// <param name='e'>Expression input for the RNN</param>
	RNNState ^RNNState::AddInput(Expression ^e) {
		ExceptionWrap(
			ensure_freshness();
		// Run through
		dynet::Expression output = __builderptr->add_input(*__stateptr, e->__thisptr);
		// Return a new output
		return gcnew RNNState(__builderptr, __builderptr->state(), this, gcnew Expression(output));
		)
	}
	/// <summary>
	/// <para>Returns the list of states obtained by adding the given inputs to the current state, one by one.</para>
	/// <remarks>This returns a list of a states, whereas transduce returns a list of outputs. This is equivalent to calling "AddInput" on each expression sequentially</remarks>
	/// </summary>
	/// <param name='l'>List of Expression inputs for the RNN</param>
	List<RNNState ^> ^RNNState::AddInputs(List<Expression ^> ^l) {
		ExceptionWrap(
			return AddInputs(l->ToArray());
		)
	}
	/// <summary>
	/// <para>Returns the list of states obtained by adding the given inputs to the current state, one by one.</para>
	/// <remarks>This returns a list of a states, whereas transduce returns a list of outputs. This is equivalent to calling "AddInput" on each expression sequentially</remarks>
	/// </summary>
	/// <param name='l'>List of Expression inputs for the RNN</param>
	List<RNNState ^> ^RNNState::AddInputs(... array<Expression ^> ^l) {
		ExceptionWrap(
			ensure_freshness();
		// Return list
		List<RNNState ^> ^ret = gcnew List<RNNState^>(l->Length);
		// Add them all in
		RNNState ^curS = this;
		for (int iExp = 0; iExp < l->Length; iExp++) {
			curS = curS->AddInput(l[iExp]);
			ret->Add(curS);
		}
		return ret;
		)
	}
	/// <summary>
	/// <para>Returns the list of output Expressions obtained by adding the given inputs to the current state, one by one.</para>
	/// </summary>
	/// <param name='l'>List of Expression inputs for the RNN</param>
	List<Expression ^> ^RNNState::Transduce(List<Expression ^> ^l) {
		ExceptionWrap(
			return Transduce(l->ToArray());
		)
	}
	/// <summary>
	/// <para>Returns the list of output Expressions obtained by adding the given inputs to the current state, one by one.</para>
	/// </summary>
	/// <param name='l'>List of Expression inputs for the RNN</param>
	List<Expression ^> ^RNNState::Transduce(... array<Expression ^> ^l) {
		ExceptionWrap(
			ensure_freshness();
		// Return list
		List<Expression ^> ^ret = gcnew List<Expression^>(l->Length);
		if (l->Length == 0) return ret;
		// Add them all in
		// Put in the first
		__builderptr->add_input(*__stateptr, l[0]->__thisptr);
		for (int iExp = 1; iExp < l->Length; iExp++) {
			ret->Add(gcnew Expression(__builderptr->back()));
			__builderptr->add_input(l[iExp]->__thisptr);
		}
		ret->Add(gcnew Expression(__builderptr->back()));

		return ret;
		)
	}
	/// <summary>
	/// <para>Manually set the output `h_t`</para>
	/// </summary>
	/// <param name='vecs'>List of expressions so set the new value</param>
	RNNState ^RNNState::SetH(... array<Expression ^> ^vecs) {
		ExceptionWrap(
			ensure_freshness();
		// Get the output
		Expression ^res = gcnew Expression(__builderptr->set_h(*__stateptr, GetDyExpVector(vecs)));
		return gcnew RNNState(__builderptr, __builderptr->state(), _prev, res);
		)
	}
	/// <summary>
	/// <para>Manually set the hidden states</para>
	/// <para>This is different from `set_h` because, for LSTMs for instance this also sets the cell state. The format is `[new_c[0],...,new_c[n],new_h[0],...,new_h[n]]`</para>
	/// </summary>
	/// <param name='vecs'>List of expressions so set the new value</param>
	RNNState ^RNNState::SetS(... array<Expression ^> ^vecs) {
		ExceptionWrap(
			ensure_freshness();
		Expression ^res = gcnew Expression(__builderptr->set_s(*__stateptr, GetDyExpVector(vecs)));
		return gcnew RNNState(__builderptr, __builderptr->state(), _prev, res);
		)
	}
	/// <summary>
	/// <para>List of Expressions representing the output of each hidden layer of the current step. The actual output of the network is at GetH()[-1].</para>
	/// </summary>
	List<Expression ^> ^RNNState::GetH() {
		ExceptionWrap(
			std::vector<dynet::Expression> expVecs = __builderptr->get_h(*__stateptr);
		// Convert to list
		return GetManagedExpList(expVecs);
		)
	}
	/// <summary>
	/// <para>List of Expressions representing the hidden state of the current step.</para>
	/// <para>For SimpleRNN, s() is the same as h()</para>
	/// <para>For LSTM, s() is a series of of memory vectors, followed the series followed by the series returned by h():</para>
	/// <para>(c[1],...,c[num_layers], h[1],...,h[num_layers])</para>
	/// </summary>
	List<Expression ^> ^RNNState::GetS() {
		ExceptionWrap(
			std::vector<dynet::Expression> expVecs = __builderptr->get_s(*__stateptr);
		// Convert to list
		return GetManagedExpList(expVecs);
		)
	}
	void RNNState::SetBuilderDropout(float f) {
		ExceptionWrap(
			__builderptr->set_dropout(f);
		)
	}
	void RNNState::DisableBuilderDropout() {
		ExceptionWrap(
			__builderptr->disable_dropout();
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	////////////////// RNNBuilder Functions ///////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// </summary>	
	RNNState ^RNNBuilder::GetInitialState() {
		ExceptionWrap(
			return GetInitialState(nullptr, true);
		)
	}
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// </summary>	
	/// <param name='fUpdate'>Default true, if this is set to False, the parameter won't be updated during the backward pass</param>
	RNNState ^RNNBuilder::GetInitialState(bool fUpdate) {
		ExceptionWrap(
			return GetInitialState(nullptr, fUpdate);
		)
	}
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// </summary>	
	/// <param name='vecs'>Initial hidden state for each layer as a list of `dynetsharp.Expression`s  (default: {None})</param>
	RNNState ^RNNBuilder::GetInitialState(array<Expression ^> ^vecs) {
		ExceptionWrap(
			return GetInitialState(vecs, true);
		)
	}
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// </summary>	
	/// <param name='vecs'>Initial hidden state for each layer as a list of `dynetsharp.Expression`s  (default: {None})</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the parameter won't be updated during the backward pass</param>
	RNNState ^RNNBuilder::GetInitialState(array<Expression ^> ^vecs, bool fUpdate) {
		ExceptionWrap(
			if (!__init_state || self_cg_version != _cg_version) {
				__thisptr->new_graph(*cg);
				__thisptr->start_new_sequence();
				__init_state = gcnew RNNState(__thisptr, __thisptr->state());
			}
		if (vecs) {
			__init_state->SetS(vecs);
		}
		return __init_state;
		)
	}
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// <remarks>Use this if you want to initialize the hidden states with values directly rather than expressions.</remarks>
	/// </summary>	
	/// <param name='vecs'>Initial hidden state for each layer as a list of `dynetsharp.Expression`s  (default: {None})</param>
	RNNState ^RNNBuilder::GetInitialStateFromRawVectors(array<Tensor ^> ^vecs) {
		ExceptionWrap(
			return GetInitialStateFromRawVectors(vecs, true);
		)
	}
	/// <summary>
	/// <para>Get a `dynetsharp.RNNState`</para>
	/// <para>This initializes a `dynet.RNNState` by loading the parameters in the computation graph</para>
	/// <remarks>Use this if you want to initialize the hidden states with values directly rather than expressions.</remarks>
	/// </summary>	
	/// <param name='vecs'>Initial hidden state for each layer as a list of `dynetsharp.Expression`s  (default: {None})</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the parameter won't be updated during the backward pass</param>
	RNNState ^RNNBuilder::GetInitialStateFromRawVectors(array<Tensor ^> ^vecs, bool fUpdate) {
		ExceptionWrap(
			array<Expression ^> ^expVecs = gcnew array<Expression ^>(vecs->Length);
		for (int iItem = 0; iItem < vecs->Length; iItem++)
			expVecs[iItem] = DynetFunctions::inputTensor(vecs[iItem]);
		return GetInitialState(expVecs);
		)
	}
	// TODO: Docs
	void RNNBuilder::SetDropout(float f) {
		ExceptionWrap(
			__thisptr->set_dropout(f);
		)
	}
	void RNNBuilder::DisableDropout() {
		ExceptionWrap(
			__thisptr->disable_dropout();
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	///////////// Custom RNNBuilder Functions /////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Retrieve the internal parameters of the RNN</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{hx},W_{hh},b_h`</para>
	/// </summary>
	List<List<Parameter ^> ^> ^SimpleRNNBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thissimpleptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters	expressions of the RNN</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{hx},W_{hh},b_h`</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^SimpleRNNBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thissimpleptr->param_vars.size() == 0 || __thissimpleptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thissimpleptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters of the GRU</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{zx},W_{zh},b_z,W_{rx},W_{rh},b_r,W_{hx},W_{hh},b_h`</para>
	/// </summary>
	List<List<Parameter ^> ^> ^GRUBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thisgruptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters	expressions of the GRU</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{zx},W_{zh},b_z,W_{rx},W_{rh},b_r,W_{hx},W_{hh},b_h`</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^GRUBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thisgruptr->param_vars.size() == 0 || __thisgruptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thisgruptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters of the LSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`</para>
	/// </summary>
	List<List<Parameter ^> ^> ^CoupledLSTMBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thislstmptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters	expressions of the LSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^CoupledLSTMBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thislstmptr->param_vars.size() == 0 || __thislstmptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thislstmptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters of the VanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// </summary>
	List<List<Parameter ^> ^> ^VanillaLSTMBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thislstmptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters expressions of the VanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^VanillaLSTMBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thislstmptr->param_vars.size() == 0 || __thislstmptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thislstmptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Set the dropout rates</para>
	/// <para>The dropout implemented here is the variational dropout introduced in (Gal, 2016 `http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks`)</para>
	/// <para>More specifically, dropout masks `{z_x}\sim{Bernoulli}(1-d)`, `{z_h}\sim{Bernoulli}(1-d_h)` are sampled at the start of each sequence.</para>
	/// <para>The dynamics of the cell are then modified to:</para>
	/// <para>i_t &amp; =\sigma(W_{ix}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)</para>
	/// <para>f_t &amp; = \sigma(W_{fx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)</para>
	/// <para>o_t &amp; = \sigma(W_{ox}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)</para>
	/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)</para>
	/// <para>c_t &amp; = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t</para>
	/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
	/// <para>For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation</para>
	/// </summary>
	/// <param name='d'>Dropout rate `d_x` for the input `x_t`</param>
	/// <param name='d_r'>Dropout rate `d_x` for the output `h_t`</param>
	void VanillaLSTMBuilder::SetDropout(float d, float d_r) {
		ExceptionWrap(
			__thislstmptr->set_dropout(d, d_r);
		)
	}
	/// <summary>
	/// <para>Set dropout masks at the beginning of a sequence for a specific batch size</para>
	/// <para>If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element</para>
	/// <remarks>You need to call this __AFTER__ calling `GetInitialState()`</remarks>
	/// </summary>
	/// <param name='batchSize'>Batch size (default: {1})</param>
	void VanillaLSTMBuilder::SetDropoutMask(int batchSize) {
		ExceptionWrap(
			__thislstmptr->set_dropout_masks(batchSize);
		)
	}

	/// <summary>
	/// <para>Retrieve the internal parameters of the SparseVanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// </summary>
	List<List<Parameter ^> ^> ^SparseLSTMBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thissparsevanillaptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters expressions of the SparseVanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^SparseLSTMBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thissparsevanillaptr->param_vars.size() == 0 || __thissparsevanillaptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thissparsevanillaptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Set the dropout rates</para>
	/// <para>The dropout implemented here is the variational dropout introduced in (Gal, 2016 `http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks`)</para>
	/// <para>More specifically, dropout masks `{z_x}\sim{Bernoulli}(1-d)`, `{z_h}\sim{Bernoulli}(1-d_h)` are sampled at the start of each sequence.</para>
	/// <para>The dynamics of the cell are then modified to:</para>
	/// <para>i_t &amp; =\sigma(W_{ix}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)</para>
	/// <para>f_t &amp; = \sigma(W_{fx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)</para>
	/// <para>o_t &amp; = \sigma(W_{ox}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)</para>
	/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)</para>
	/// <para>c_t &amp; = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t</para>
	/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
	/// <para>For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation</para>
	/// </summary>
	/// <param name='d'>Dropout rate `d_x` for the input `x_t`</param>
	/// <param name='d_r'>Dropout rate `d_x` for the output `h_t`</param>
	void SparseLSTMBuilder::SetDropout(float d, float d_r) {
		ExceptionWrap(
			__thissparsevanillaptr->set_dropout(d, d_r);
		)
	}
	/// <summary>
	/// <para>Set dropout masks at the beginning of a sequence for a specific batch size</para>
	/// <para>If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element</para>
	/// <remarks>You need to call this __AFTER__ calling `GetInitialState()`</remarks>
	/// </summary>
	/// <param name='batchSize'>Batch size (default: {1})</param>
	void SparseLSTMBuilder::SetDropoutMask(int batchSize) {
		ExceptionWrap(
			__thissparsevanillaptr->set_dropout_masks(batchSize);
		)
	}

	/// <summary>
	/// <para>Retrieve the internal parameters of the CompactVanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// </summary>
	List<List<Parameter ^> ^> ^CompactVanillaLSTMBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thiscompvanillaptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters expressions of the CompactVanillaLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_x,W_h,b` where `W_x,W_h` are stacked version of the individual gates matrices</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^CompactVanillaLSTMBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thiscompvanillaptr->param_vars.size() == 0 || __thiscompvanillaptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thiscompvanillaptr->param_vars);
		)
	}
	/// <summary>
	/// <para>Set the dropout rates</para>
	/// <para>The dropout implemented here is the variational dropout introduced in (Gal, 2016 `http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks`)</para>
	/// <para>More specifically, dropout masks `{z_x}\sim{Bernoulli}(1-d)`, `{z_h}\sim{Bernoulli}(1-d_h)` are sampled at the start of each sequence.</para>
	/// <para>The dynamics of the cell are then modified to:</para>
	/// <para>i_t &amp; =\sigma(W_{ix}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ih}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_i)</para>
	/// <para>f_t &amp; = \sigma(W_{fx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{fh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_f)</para>
	/// <para>o_t &amp; = \sigma(W_{ox}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{oh}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_o)</para>
	/// <para>\\tilde{c_t} &amp; = \\tanh(W_{cx}(\\frac 1 {1-d_x}\mathbf{z_x} \circ x_t)+W_{ch}(\\frac 1 {1-d_h}\mathbf{z_h} \circ h_{t-1})+b_c)</para>
	/// <para>c_t &amp; = c_{t-1}\circ f_t + \\tilde{c_t}\circ i_t</para>
	/// <para>h_t &amp; = \\tanh(c_t)\circ o_t</para>
	/// <para>For more detail as to why scaling is applied, see the "Unorthodox" section of the documentation</para>
	/// </summary>
	/// <param name='d'>Dropout rate `d_x` for the input `x_t`</param>
	/// <param name='d_r'>Dropout rate `d_x` for the output `h_t`</param>
	void CompactVanillaLSTMBuilder::SetDropout(float d, float d_r) {
		ExceptionWrap(
			__thiscompvanillaptr->set_dropout(d, d_r);
		)
	}
	/// <summary>
	/// <para>Set dropout masks at the beginning of a sequence for a specific batch size</para>
	/// <para>If this function is not called on batched input, the same mask will be applied across all batch elements. Use this to apply different masks to each batch element</para>
	/// <remarks>You need to call this __AFTER__ calling `GetInitialState()`</remarks>
	/// </summary>
	/// <param name='batchSize'>Batch size (default: {1})</param>
	void CompactVanillaLSTMBuilder::SetDropoutMask(int batchSize) {
		ExceptionWrap(
			__thiscompvanillaptr->set_dropout_masks(batchSize);
		)
	}


	/// <summary>
	/// <para>Retrieve the internal parameters of the FastLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`</para>
	/// </summary>
	List<List<Parameter ^> ^> ^FastLSTMBuilder::GetParameters() {
		ExceptionWrap(
			return GetParameterListOfLists(__thisfastlstmptr->params);
		)
	}
	/// <summary>
	/// <para>Retrieve the internal parameters expressions of the FastLSTM</para>
	/// <para>The output is a list with one item per layer. Each item is a list containing `W_{ix},W_{ih},W_{ic},b_i,W_{ox},W_{oh},W_{oc},b_o,W_{cx},W_{ch},b_c`</para>
	/// <remarks>This raises an expression if GetInitialState() hasn't been called because it requires thr parameters to be loaded in the computation graph. However it prevents the parameters to be loaded twice in the computation graph (compared to `rnn.GetParameters()[0][0].ToExpression()` for example).</remarks>
	/// </summary>
	List<List<Expression ^> ^> ^FastLSTMBuilder::GetParameterExpressions() {
		ExceptionWrap(
			if (__thisfastlstmptr->param_vars.size() == 0 || __thisfastlstmptr->param_vars[0][0].is_stale())
				throw gcnew Exception("Attempt to use a stale expression, renew CG and/or call initial_state before accessing SimpleRNNBuilder internal parameters expression");
		return GetParameterExpressionListOfLists(__thisfastlstmptr->param_vars);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	//////////////// DynetParams Functions ////////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>This object holds the global parameters of Dynet</para>
	/// <para>This is useful if you want to specify the global dynet parameters(memory, random seed...) programmatically. </para>
	/// <remarks>Don't forget to initialize with <c>dyparams.Initialize()</c>, otherwise dynet will raise an error.</remarks>
	/// </summary>
	/// <example>Example:
	/// <c>
	/// DynetParams dyparams = DynetParams();
	/// dyparams.AutoBatch = true;
	/// dyparams.RandomSeed = 8427628;
	/// dyparams.Initialize();
	/// </c>
	/// </example>
	DynetParams::DynetParams() {
		ExceptionWrap(
			__thisptr = new dynet::DynetParams();
		)
	}
	/// <summary>
	/// <para>Number of requested gpus</para>
	/// <remarks>Cancels out "SetGPUMask" and "SetDeviceIDs"</remarks>
	/// </summary>
	/// <param name='n'>Numbers of GPUs to load</param>
	void DynetParams::SetRequestedGPUs(int n) {
		ExceptionWrap(
			__thisptr->requested_gpus = n;
		__thisptr->ngpus_requested = true;
		__thisptr->ids_requested = false;
		)
	}
	/// <summary>
	/// <para>Set the GPU mask to load, should be a list of booleans</para>
	/// <remarks>Cancels out "SetRequestedGPUs" and "SetDeviceIDs"</remarks>
	/// </summary>
	/// <param name='mask'>List of true/false per GPU</param>
	void DynetParams::SetGPUMask(array<bool> ^mask) {
		if (__thisptr->gpu_mask.size() == 0)
			throw gcnew ArgumentException("The current installation of dynet does not support GPU");
		ExceptionWrap(
			// Reset the whole mask
			int maxMask = mask->Length;
		for (int i = 0; i < __thisptr->gpu_mask.size(); i++)
			__thisptr->gpu_mask[i] = i < maxMask && mask[i] && 1;
		__thisptr->ngpus_requested = false;
		__thisptr->ids_requested = true;
		)
	}
	/// <summary>
	/// <para>Set the GPU mask to load, should be a list of booleans</para>
	/// <remarks>Cancels out "SetRequestedGPUs" and "SetDeviceIDs"</remarks>
	/// </summary>
	/// <param name='mask'>List of true/false per GPU</param>
	void DynetParams::SetGPUMask(List<bool> ^mask) {
		ExceptionWrap(
			SetGPUMask(mask->ToArray());
		)
	}
	/// <summary>
	/// <para>Set the IDs of the devices to load</para>
	/// <remarks>Cancels out "SetGPUMask" and "SetRequestedGPUs"</remarks>
	/// </summary>
	/// <param name='devices'>Comma-separated list of devices to load</param>
	void DynetParams::SetDeviceIDs(String ^devices) {
		ExceptionWrap(
			char **argv = (char **)malloc(sizeof(char *) * 3);
		int argc = 3;
		argv[0] = (char *)"";
		argv[1] = (char *)"--dynet-devices";
		argv[2] = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(devices).ToPointer();
		// Send to dynet to extract the gpu mask (and handle all the exceptions)
		auto dp = dynet::extract_dynet_params(argc, argv);
		// Copy everything in
		__thisptr->ids_requested = true;
		__thisptr->cpu_requested = true;
		__thisptr->requested_gpus = dp.requested_gpus;
		__thisptr->gpu_mask.assign(dp.gpu_mask.begin(), dp.gpu_mask.end());

		// Free
		free(argv[2]);
		free(argv);
		)
	}
	/// <summary>
	/// <para>Set the IDs of the devices to load</para>
	/// <remarks>Cancels out "SetGPUMask" and "SetRequestedGPUs"</remarks>
	/// </summary>
	/// <param name='devices'>List of devices to load</param>
	void DynetParams::SetDeviceIDs(... array<String ^> ^devices) {
		ExceptionWrap(
			SetDeviceIDs(System::String::Join(",", devices));
		)
	}
	/// <summary>
	/// <para>Set the IDs of the devices to load</para>
	/// <remarks>Cancels out "SetGPUMask" and "SetRequestedGPUs"</remarks>
	/// </summary>
	/// <param name='devices'>List of devices to load</param>
	void DynetParams::SetDeviceIDs(List<String ^> ^devices) {
		ExceptionWrap(
			SetDeviceIDs(devices->ToArray());
		)
	}

	/// <summary>
	/// <para>Get the mask state of a given GPU</para>
	/// </summary>
	/// <param name='index'>Index of GPU</param>
	bool DynetParams::GetGPUMaskState(int index) {
		if (!__thisptr->ids_requested)
			throw gcnew Exception("Cannot get mask state of GPU when not in ID mode");
		ExceptionWrap(
			return __thisptr->gpu_mask[index];
		)
	}

	/// <summary>
	/// <para>Initialize dynet with the current dynetparams object.</para>
	/// <remarks>This is one way, you can't uninitialize dynet</remarks>
	/// </summary>
	void DynetParams::Initialize() {
		//std::set_unexpected(unexpectedHandler);
		fInitialized = true;
		ExceptionWrap(
			initialMemorySize = std::stoull(__thisptr->mem_descriptor);
		maxOverallMemory = maxMemory;
		dynet::initialize(*__thisptr);
		)
	}
	/// <summary>
	/// <para>Create the DynetParams object for initializing dynet using the command line arguments.</para>
	/// <remarks>Accepts the standard arg format for dynet</remarks>
	/// <param name='args'>Command line arguments</param>
	/// </summary>
	DynetParams ^DynetParams::FromArgs(array<String ^> ^args) {
		ExceptionWrap(
			// Convert to C array
			int argc = args->Length;
			char **argv = (char **)malloc(argc * sizeof(char *));
			for (int iArg = 0; iArg < argc; iArg++)
				argv[iArg] = (char *)Runtime::InteropServices::Marshal::StringToHGlobalAnsi(args[iArg]).ToPointer();
			// Create the return object
			DynetParams ^ret = gcnew DynetParams(dynet::extract_dynet_params(argc, argv));

			// Free
			for (int iArg = 0; iArg < argc; iArg++) free(argv[iArg]);
			free(argv);

			return ret;
		)
	}
	/// <summary>
	/// <para>Update the default&amp;max mem-descriptor variables, memory is released during RenewCG().</para>
	/// </summary>
	void DynetParams::UpdateMemDescriptors() {
		ExceptionWrap(
			// Set the new memory size
			initialMemorySize = std::stoull(__thisptr->mem_descriptor);
		maxOverallMemory = maxMemory;
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////
	//////////////////// Static Functions /////////////////////
	///////////////////////////////////////////////////////////
	/// <summary>
	/// <para>Renew the computation graph.</para>
	/// <para>Call this before building any new computation graph</para>
	/// </summary>
	void DynetFunctions::RenewCG() {
		ExceptionWrap(
			RenewCG(false, false);
		)
	}
	/// <summary>
	/// <para>Renew the computation graph.</para>
	/// <para>Call this before building any new computation graph</para>
	/// </summary>
	/// <param name='fImmediateCompute'>Compute all expressions immediately, without waiting for a `forward` call (default: False)</param>
	/// <param name='fImmediateCompute'>Check validity of expressions without waiting for a `forward` call (default: False)</param>
	void DynetFunctions::RenewCG(bool fImmediateCompute, bool fCheckValidity) {
		CheckForInitialized();
		ExceptionWrap(
			delete cg;
			for (auto val : _floatInputs) delete val;
			for (auto val : _vecInputs) delete val;

		_cg_version++;
		// New memory?
		if (maxOverallMemory && HowMuchMemoryDynet() > maxOverallMemory)
			ResetDynetMemory(initialMemorySize);

		// Create a new graph
		cg = new ComputationGraph();
		cg->set_immediate_compute(fImmediateCompute);
		cg->set_check_validity(fCheckValidity);
		_floatInputs.clear();
		_vecInputs.clear();
		)
	}
	/// <summary>
	/// <para>Saves the state of the computation graph.</para>
	/// </summary>
	void DynetFunctions::CheckpointCG() {
		CheckForInitialized();
		ExceptionWrap(
			cg->checkpoint();
		)
	}
	/// <summary>
	/// <para>Revert the computation graph state to the previous checkpoint.</para>
	/// </summary>
	void DynetFunctions::RevertCG() {
		CheckForInitialized();
		ExceptionWrap(
			cg->revert();
		)
	}
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// </summary>
	/// <param name='lp'>LookupParameter to retrieve record from</param>
	/// <param name='index'>Index of row to lookup</param>
	Expression ^DynetFunctions::lookup(LookupParameter ^lp, int index) {
		ExceptionWrap(
			return lookup(lp, index, true);
		)
	}
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// </summary>
	/// <param name='lp'>LookupParameter to retrieve record from</param>
	/// <param name='index'>Index of row to lookup</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the returned expression won't be updated during the backward pass</param>
	Expression ^DynetFunctions::lookup(LookupParameter ^lp, int index, bool fUpdate) {
		ExceptionWrap(
			if (fUpdate)
				return gcnew Expression(dynet::lookup(*cg, *lp->__thisptr, index));
			return gcnew Expression(dynet::const_lookup(*cg, *lp->__thisptr, index));
		)
	}
	/// <summary>
	/// <para>Minibatch - Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// </summary>
	/// <param name='lp'>LookupParameter to retrieve record from</param>
	/// <param name='indexes'>Array of indexes of rows to lookup</param>
	Expression ^DynetFunctions::lookup_batch(LookupParameter ^lp, array<int> ^indexes) {
		ExceptionWrap(
			return lookup_batch(lp, indexes, true);
		)
	}
	/// <summary>
	/// <para>Lookup an embedding from a lookup parameter and returns it as a expression</para>
	/// </summary>
	/// <param name='lp'>LookupParameter to retrieve record from</param>
	/// <param name='indexes'>Array of indexes of rows to lookup</param>
	/// <param name='fUpdate'>Default true, if this is set to False, the returned expression won't be updated during the backward pass</param>
	Expression ^DynetFunctions::lookup_batch(LookupParameter ^lp, array<int> ^indexes, bool fUpdate) {
		ExceptionWrap(
			if (fUpdate)
				return gcnew Expression(dynet::lookup(*cg, *lp->__thisptr, VecToUInt(ConvertArrayToVector<int>(indexes))));
			return gcnew Expression(dynet::const_lookup(*cg, *lp->__thisptr, VecToUInt(ConvertArrayToVector<int>(indexes))));
		)
	}
	/// <summary>
	/// <para>Add parameters to the computation graph.</para>
	/// <para>Get the expression objects corresponding to parameters. Gradients for parameters will be computed and used by Optimizers to update.</para>
	/// <remarks>Deprecated - an explicit conversion is no longer needed.</remarks>
	/// </summary>
	/// <param name='p'>Parameter object to add to the computation graph</param>
	[Obsolete("Deprecated - an explicit conversion is no longer needed.")]
	Expression ^DynetFunctions::parameter(Parameter ^p) {
		ExceptionWrap(
			return p;
		)
	}
	/// <summary>
	/// <para>Add parameters to the computation graph.</para>
	/// <para>Get the expression objects corresponding to parameters. Gradients for parameters will NOT be computed and used by Optimizers to update. To access parameters that should be updated (which is usually what you want), use parameter() instead.</para>
	/// <remarks>Equivalent to doing Parameter.ToExpression(false)</remarks>
	/// </summary>
	/// <param name='p'>Parameter object to add to the computation graph</param>
	Expression ^DynetFunctions::const_parameter(Parameter ^p) {
		ExceptionWrap(
			return p->ToExpression(false);
		)
	}
	/// <summary>
	/// <para>Pick element.</para> 
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// </summary>
	/// <param name='exp'>Expression to pick from</param>
	/// <param name='index'>Index to pick</param>
	Expression ^DynetFunctions::pick(Expression ^exp, int index) {
		ExceptionWrap(
			return pick(exp, index, 0);
		)
	}
	/// <summary>
	/// <para>Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// </summary>
	/// <param name='exp'>Expression to pick from</param>
	/// <param name='index'>Index to pick</param>
	/// <param name='dim'>Index of dimension to pick from, default 0</param>
	Expression ^DynetFunctions::pick(Expression ^exp, int index, int dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick(exp->__thisptr, index, dim));
		)
	}
	/// <summary>
	/// <para>Minibatch - Pick element.</para> 
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// </summary>
	/// <param name='exp'>Expression to pick from</param>
	/// <param name='indexes'>Array of indexes to pick</param>
	Expression ^DynetFunctions::pick_batch(Expression ^exp, array<int> ^indexes) {
		ExceptionWrap(
			return pick_batch(exp, indexes, 0);
		)
	}
	/// <summary>
	/// <para>Minibatch - Pick element.</para>
	/// <para>Pick a single element/row/column/sub-tensor from an expression. This will result in the dimension of the tensor being reduced by 1.</para>
	/// </summary>
	/// <param name='exp'>Expression to pick from</param>
	/// <param name='indexes'>Array of indexes to pick</param>
	/// <param name='dim'>Index of dimension to pick from, default 0</param>
	Expression ^DynetFunctions::pick_batch(Expression ^exp, array<int> ^indexes, int dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick(exp->__thisptr, VecToUInt(ConvertArrayToVector<int>(indexes)), dim));
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a scalar value</para>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(float num) {
		ExceptionWrap(
			return input(num, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a scalar value</para>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::input(float num, String ^device) {
		ExceptionWrap(
			float *val = new float(num);
			_floatInputs.push_back(val);
			return gcnew Expression(dynet::input(*cg, val, str2dev(device)), val);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 1-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(array<float>^ num) {
		ExceptionWrap(
			return input(num, 1, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 1-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::input(array<float>^ num, int batchSize) {
		ExceptionWrap(
			return input(num, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 1-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::input(array<float>^ num, String ^device) {
		ExceptionWrap(
			return input(num, 1, device);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 1-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::input(array<float>^ num, int batchSize, String ^device) {
		ExceptionWrap(
			std::vector<float> *val = new std::vector<float>(ConvertArrayToVector<float>(num));
			_vecInputs.push_back(val);
			dynet::Dim d({ (unsigned)num->Length }, batchSize);
			return gcnew Expression(dynet::input(*cg, d, val, str2dev(device)), val);
		) 
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 2-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(array<array<float>^>^ num) {
		ExceptionWrap(
			return input(num, 1, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 2-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::input(array<array<float>^>^ num, int batchSize) {
		ExceptionWrap(
			return input(num, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 2-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::input(array<array<float>^>^ num, String ^device) {
		ExceptionWrap(
			return input(num, 1, device);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 2-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::input(array<array<float>^>^ num, int batchSize, String ^device) {
		ExceptionWrap(
			std::vector<float> vec;
			if (num->Length > 0) {
				for (int iItem = 0; iItem < num[0]->Length; iItem++)
					for (int iVec = 0; iVec < num->Length; iVec++)
						vec.push_back((real)num[iVec][iItem]);
			}
		// Convert to pointer, and send that
		std::vector<float> *val = new std::vector<float>(vec);
		_vecInputs.push_back(val);
		dynet::Dim d({ (unsigned)num->Length, (unsigned)num[0]->Length }, batchSize);
		return gcnew Expression(dynet::input(*cg, d, val, str2dev(device)), val);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a the value of the Tensor</para>
	/// <remarks>Note: The values in the Tensor object are *not* linked to the values in the Expression</remarks>
	/// </summary>
	/// <param name='tensor'>Value of Expression</param>
	Expression ^DynetFunctions::inputTensor(Tensor ^tensor) {
		ExceptionWrap(
			return inputTensor(tensor, "");
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a the value of the Tensor</para>
	/// <remarks>Note: The values in the Tensor object are *not* linked to the values in the Expression</remarks>
	/// </summary>
	/// <param name='tensor'>Value of Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::inputTensor(Tensor ^tensor, String ^device) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, tensor->Shape()->get_cdim(), *tensor->_vec, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression object with certain dimensions</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	Expression ^DynetFunctions::inputTensor(array<long> ^dim) {
		ExceptionWrap(
			return inputTensor(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression object with certain dimensions</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::inputTensor(array<long> ^dim, int batchSize) {
		ExceptionWrap(
			return inputTensor(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression object with certain dimensions</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::inputTensor(array<long> ^dim, String ^device) {
		ExceptionWrap(
			return inputTensor(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression object with certain dimensions</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::inputTensor(array<long> ^dim, int batchSize, String ^device) {
		ExceptionWrap(
			std::vector<real> *val = new std::vector<real>();
			_vecInputs.push_back(val);
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::input(*cg, d, val, str2dev(device)), val);
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression vector of a certain dimension</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	Expression ^DynetFunctions::inputVector(long dim) {
		ExceptionWrap(
			return inputVector(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression vector of a certain dimension</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::inputVector(long dim, int batchSize) {
		ExceptionWrap(
			return inputVector(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression vector of a certain dimension</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::inputVector(long dim, String ^device) {
		ExceptionWrap(
			return inputVector(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a modifiable Expression vector of a certain dimension</para>
	/// <remarks>You must call "SetValue" on the Expression before doing a forward pass</remarks>
	/// </summary>
	/// <param name='dim'>Dimensions of the Expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::inputVector(long dim, int batchSize, String ^device) {
		ExceptionWrap(
			std::vector<real> *val = new std::vector<real>();
			_vecInputs.push_back(val);
			dynet::Dim d({ (unsigned int)dim }, batchSize);
			return gcnew Expression(dynet::input(*cg, d, val, str2dev(device)), val);
		)
	}
	Expression ^DynetFunctions::average(List<Expression^> ^l) {
		ExceptionWrap(
			return average(l->ToArray());
		)
	}
	Expression ^DynetFunctions::average(... array<Expression^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::average(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::esum(List<Expression^> ^l) {
		ExceptionWrap(
			return esum(l->ToArray());
		)
	}
	Expression ^DynetFunctions::esum(... array<Expression^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::sum(List<Expression^> ^l) {
		ExceptionWrap(
			return esum(l->ToArray());
		)
	}
	Expression ^DynetFunctions::sum(... array<Expression^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum(GetDyExpVector(arr)));
		)
	}
	/// <summary>
	/// <para>Create an input full of zeros</para>
	/// <para>Create an input full of zeros, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::zeros(array<long> ^dim) {
		ExceptionWrap(
			return zeros(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of zeros</para>
	/// <para>Create an input full of zeros, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::zeros(array<long> ^dim, int batchSize) {
		ExceptionWrap(
			return zeros(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of zeros</para>
	/// <para>Create an input full of zeros, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::zeros(array<long> ^dim, String ^device) {
		ExceptionWrap(
			return zeros(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create an input full of zeros</para>
	/// <para>Create an input full of zeros, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::zeros(array<long> ^dim, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::zeros(*cg, d, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where one coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>Index of the coordinate that is 1</param>
	Expression ^DynetFunctions::one_hot(int dim, int idx) {
		ExceptionWrap(
			return one_hot(dim, idx, "");
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where one coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>Index of the coordinate that is 1</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::one_hot(int dim, int idx, String ^device) {
		ExceptionWrap(
			return gcnew Expression(dynet::one_hot(*cg, dim, idx, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where every idx coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>Array of indexes where the coordinate will be 1</param>
	Expression ^DynetFunctions::one_hot(int dim, array<int> ^idx) {
		ExceptionWrap(
			return one_hot(dim, idx, "");
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where every idx coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>List of indexes where the coordinate will be 1</param>
	Expression ^DynetFunctions::one_hot(int dim, List<int> ^idx) {
		ExceptionWrap(
			return one_hot(dim, idx, "");
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where every idx coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>List of indexes where the coordinate will be 1</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::one_hot(int dim, List<int> ^idx, String ^device) {
		ExceptionWrap(
			return one_hot(dim, idx->ToArray(), device);
		)
	}
	/// <summary>
	/// <para>Inputs a one hot vector into the graph.</para>
	/// <para>A one hot vector is a vector where every idx coordinate is 1 and everything else is 0</para>
	/// </summary>
	/// <param name='dim'>Dimension of the vector</param>
	/// <param name='idx'>Array of indexes where the coordinate will be 1</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::one_hot(int dim, array<int> ^idx, String ^device) {
		ExceptionWrap(
			return gcnew Expression(dynet::one_hot(*cg, dim, VecToUInt(ConvertArrayToVector<int>(idx)), str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create an input full of ones</para>
	/// <para>Create an input full of ones, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::ones(array<long> ^dim) {
		ExceptionWrap(
			return ones(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of ones</para>
	/// <para>Create an input full of ones, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::ones(array<long> ^dim, int batchSize) {
		ExceptionWrap(
			return ones(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of ones</para>
	/// <para>Create an input full of ones, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::ones(array<long> ^dim, String ^device) {
		ExceptionWrap(
			return ones(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create an input full of ones</para>
	/// <para>Create an input full of ones, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::ones(array<long> ^dim, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::ones(*cg, d, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create an input full of a constant value</para>
	/// <para>Create an input full of a constant value, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='val'>Constant value</param>
	Expression ^DynetFunctions::constant(array<long> ^dim, float val) {
		ExceptionWrap(
			return constant(dim, val, 1, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of a constant value</para>
	/// <para>Create an input full of a constant value, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='val'>Constant value</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::constant(array<long> ^dim, float val, int batchSize) {
		ExceptionWrap(
			return constant(dim, val, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create an input full of a constant value</para>
	/// <para>Create an input full of a constant value, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='val'>Constant value</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::constant(array<long> ^dim, float val, String ^device) {
		ExceptionWrap(
			return constant(dim, val, 1, device);
		)
	}
	/// <summary>
	/// <para>Create an input full of a constant value</para>
	/// <para>Create an input full of a constant value, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='val'>Constant value</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::constant(array<long> ^dim, float val, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::constant(*cg, d, val, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim) {
		ExceptionWrap(
			return random_normal(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, int batchSize) {
		ExceptionWrap(
			return random_normal(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, String ^device) {
		ExceptionWrap(
			return random_normal(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_normal(*cg, d, 0.0f, 1.0f, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mean'>Mean value for distribution (default: 0)</param>
	/// <param name='stddev'>Variance value for distribution (default: 1)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, float mean, float stddev) {
		ExceptionWrap(
			return random_normal(dim, mean, stddev, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mean'>Mean value for distribution (default: 0)</param>
	/// <param name='stddev'>Variance value for distribution (default: 1)</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, float mean, float stddev, int batchSize) {
		ExceptionWrap(
			return random_normal(dim, mean, stddev, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mean'>Mean value for distribution (default: 0)</param>
	/// <param name='stddev'>Variance value for distribution (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, float mean, float stddev, String ^device) {
		ExceptionWrap(
			return random_normal(dim, mean, stddev, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mean'>Mean value for distribution (default: 0)</param>
	/// <param name='stddev'>Variance value for distribution (default: 1)</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim, float mean, float stddev, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_normal(*cg, d, mean, stddev, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p) {
		ExceptionWrap(
			return random_bernoulli(dim, p, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, int batchSize) {
		ExceptionWrap(
			return random_bernoulli(dim, p, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, String ^device) {
		ExceptionWrap(
			return random_bernoulli(dim, p, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_bernoulli(*cg, d, p, 1.0f, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='scale'>Scaling factor to apply to the sampled tensor (default: (1.0))</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, float scale) {
		ExceptionWrap(
			return random_bernoulli(dim, p, scale, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='scale'>Scaling factor to apply to the sampled tensor (default: (1.0))</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, float scale, int batchSize) {
		ExceptionWrap(
			return random_bernoulli(dim, p, scale, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='scale'>Scaling factor to apply to the sampled tensor (default: (1.0))</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, float scale, String ^device) {
		ExceptionWrap(
			return random_bernoulli(dim, p, scale, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random bernoulli tensor</para>
	/// <para>Create a tensor distributed according to bernoulli distribution with parameter P</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='p'>Parameter of the bernoulli distribution</param>
	/// <param name='scale'>Scaling factor to apply to the sampled tensor (default: (1.0))</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_bernoulli(array<long> ^dim, float p, float scale, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_bernoulli(*cg, d, p, scale, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random uniform tensor</para>
	/// <para>Create a tensor distributed according to uniform distribution with boundaries left and right.</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='left'>Lower bound of the uniform distribution</param>
	/// <param name='right'>Upper bound of the uniform distribution</param>
	Expression ^DynetFunctions::random_uniform(array<long> ^dim, float left, float right) {
		ExceptionWrap(
			return random_uniform(dim, left, right, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random uniform tensor</para>
	/// <para>Create a tensor distributed according to uniform distribution with boundaries left and right.</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='left'>Lower bound of the uniform distribution</param>
	/// <param name='right'>Upper bound of the uniform distribution</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_uniform(array<long> ^dim, float left, float right, int batchSize) {
		ExceptionWrap(
			return random_uniform(dim, left, right, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random uniform tensor</para>
	/// <para>Create a tensor distributed according to uniform distribution with boundaries left and right.</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='left'>Lower bound of the uniform distribution</param>
	/// <param name='right'>Upper bound of the uniform distribution</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_uniform(array<long> ^dim, float left, float right, String ^device) {
		ExceptionWrap(
			return random_uniform(dim, left, right, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random uniform tensor</para>
	/// <para>Create a tensor distributed according to uniform distribution with boundaries left and right.</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='left'>Lower bound of the uniform distribution</param>
	/// <param name='right'>Upper bound of the uniform distribution</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_uniform(array<long> ^dim, float left, float right, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_uniform(*cg, d, left, right, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim) {
		ExceptionWrap(
			return random_gumbel(dim, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, int batchSize) {
		ExceptionWrap(
			return random_gumbel(dim, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, String ^device) {
		ExceptionWrap(
			return random_gumbel(dim, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, int batchSize, String ^device) {
		ExceptionWrap(
			dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
			return gcnew Expression(dynet::random_gumbel(*cg, d, 0.0f, 1.0f, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mu'>Parameter mu of the gumbel distribution (default: 0.0, non-default not supported)</param>
	/// <param name='beta'>Parameter beta of the gumbel distribution (default: 1.0, non-default not supported)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, float mu, float beta) {
		ExceptionWrap(
			return random_gumbel(dim, mu, beta, 1, "");
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mu'>Parameter mu of the gumbel distribution (default: 0.0, non-default not supported)</param>
	/// <param name='beta'>Parameter beta of the gumbel distribution (default: 1.0, non-default not supported)</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, float mu, float beta, int batchSize) {
		ExceptionWrap(
			return random_gumbel(dim, mu, beta, batchSize, "");
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mu'>Parameter mu of the gumbel distribution (default: 0.0, non-default not supported)</param>
	/// <param name='beta'>Parameter beta of the gumbel distribution (default: 1.0, non-default not supported)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, float mu, float beta, String ^device) {
		ExceptionWrap(
			return random_gumbel(dim, mu, beta, 1, device);
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	/// <param name='mu'>Parameter mu of the gumbel distribution (default: 0.0, non-default not supported)</param>
	/// <param name='beta'>Parameter beta of the gumbel distribution (default: 1.0, non-default not supported)</param>
	/// <param name='batchSize'>Batch size (default: 1)</param>
	/// <param name='device'> Optional device name for this parameter (default: "", default device)</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim, float mu, float beta, int batchSize, String ^device) {
		ExceptionWrap(
			if (mu != 0.0 || beta != 1.0)
				throw gcnew Exception(gcnew String("Currently only paramters of mu=0.0 and beta=1.0 are supported."));
		dynet::Dim d(ConvertArrayToVector<long>(dim), batchSize);
		return gcnew Expression(dynet::random_gumbel(*cg, d, mu, beta, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Flip gradient</para>
	/// <para>This node has no effect on the forward pass, but takes negative on backprop process. This operation is widely used in adversarial networks.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::flip_gradient(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::flip_gradient(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Scale gradient</para>
	/// <para>This node scales the gradient by a constant on backprop, with no effect on the forward pass.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::scale_gradient(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::scale_gradient(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Scale gradient</para>
	/// <para>This node scales the gradient by a constant on backprop, with no effect on the forward pass.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	/// <param name='lambd'>Lambda value for operation</param>
	Expression ^DynetFunctions::scale_gradient(Expression ^x, float lambd) {
		ExceptionWrap(
			return gcnew Expression(dynet::scale_gradient(x->__thisptr, lambd));
		)
	}
	/// <summary>
	/// <para>Argmax</para>
	/// <para>This node takes an input vector `x` and returns a one hot vector `y` such that `y_argmax{x}=1`. There are two gradient modes for this operation:</para>
	/// <para>1] "zero_gradient": This is the standard argmax operation. Note that this almost everywhere differentiable and its gradient is 0. **It will stop your gradient**</para>
	/// <para>2] "straight_through_gradient": This gradient mode implements the straight-through estimator (Bengio et al., 2013) (https://arxiv.org/abs/1308.3432). Its forward pass is the same as the argmax operation, but its gradient is the same as the identity function.</para>
	/// <remarks>Note that this does not technically correspond to a differentiable function (hence the name "estimator"). Tensors of order `1` are not supported yet. If you really need to use this operation on matrices, tensors, etc... feel free to open an issue on github.</remarks>
	/// </summary>
	/// <param name='x'>The input vector (can be batched)</param>
	/// <param name='gm'>Gradient mode for the backward pass (one of "zero_gradient" or "straight_through_gradient")</param>
	Expression ^DynetFunctions::argmax(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::argmax(x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	/// <summary>
	/// <para>Componentwise division</para>
	/// <para>Divide an expressions component-wise by another, broadcasting dimensions (currently only of the second expression!) if necessary as follows:</para>
	/// <para>- When number of dimensions differ, we add dimensions of size 1 to make the number of dimensions match</para>
	/// <para>- Now, every dimensions is required to have matching size, or the dim size of the right expression must equal 1 (in which case it will be broadcasted)</para>
	/// <para>- In the same way, the batch sizes must match, or the batch size of the right expression must equal 1 in which case it will be broadcasted</para>
	/// <para>- The resulting tensor's dimensionality is thus determined as the max of both inputs at every position</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::cdiv(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::cdiv(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Componentwise multiplication</para>
	/// <para>Multiply two expressions component-wise, broadcasting dimensions if necessary as follows:</para>
	/// <para>- When number of dimensions differ, we add dimensions of size 1 to make the number of dimensions match</para>
	/// <para>- Now, every dimensions is required to have matching size, or one of the dimensions must equal 1 (in which case it will be broadcasted)</para>
	/// <para>- In the same way, the batch dimension must match, or equal 1 in which case it will be broadcasted</para>
	/// <para>- The resulting tensor's dimensionality is thus determined as the max of both inputs at every position</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::cmult(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::cmult(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Columnwise addition</para>
	/// <para>Add vector `y` to each column of matrix `x`</para>
	/// </summary>
	/// <param name='x'>An MxN matrix</param>
	/// <param name='y'>A length M vector</param>
	Expression ^DynetFunctions::colwise_add(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::colwise_add(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Matrix Inverse</para>
	/// <para>Takes the inverse of a matrix. Note that back-propagating through an inverted matrix can also be the source of stability problems sometimes.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::inverse(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::inverse(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Log determinant</para>
	/// <para>Takes the log of the determinant of a matrix.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::logdet(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::logdet(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Trace of Matrix Product</para>
	/// <para>Takes the trace of the product of matrices.</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::trace_of_product(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::trace_of_product(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Dot Product</para>
	/// <para>Calculate the dot product [x*y=sum(x_i * y_i)]</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::dot_product(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::dot_product(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Circular convolution</para>
	/// <para>Calculate the circular convolution [u*v]_k=sum(u_i * v_{(k-i)%d})`</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::circ_conv(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::circ_conv(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Circular correlation</para>
	/// <para>Calculate the circular correlation [u`*`v]_k=sum(u_i * v_{(i+k)%d})`</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::circ_corr(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::circ_corr(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Squared norm</para>
	/// <para>The squared norm of the values of `x`: |x|^2=sum(x_i^2)</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::squared_norm(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::squared_norm(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>L2 norm</para>
	/// <para>The l2 norm of the values of `x`: |x|=sqrt(sum(x_i^2))</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::l2_norm(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::l2_norm(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Squared distance</para>
	/// <para>The squared distance between values of `x` and `y`: |x-y|^2=sum((x_i-y_i)^2)</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::squared_distance(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::squared_distance(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>L1 distance</para>
	/// <para>L1 distance between values of `x` and `y`: |x-y|=sum(|x_i-y_i|)</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::l1_distance(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::l1_distance(x->__thisptr, y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Binary log loss</para>
	/// <para>The log loss of a binary decision according to the sigmoid function -sum(y_i*ln(x_i) + (1-y_i)*ln(1-x_i))</para>
	/// </summary>
	/// <param name='x'>The first input expression</param>
	/// <param name='y'>The second input expression</param>
	Expression ^DynetFunctions::binary_log_loss(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::binary_log_loss(x->__thisptr, y->__thisptr));
		)
	}
	//TODO: Docs
	Expression ^DynetFunctions::filter1d_narrow(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::filter1d_narrow(x->__thisptr, y->__thisptr));
		)
	}
	//TODO: Docs
	Expression ^DynetFunctions::conv2d(Expression ^x, Expression ^y, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(x->__thisptr, y->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::conv2d(Expression ^x, Expression ^y, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(x->__thisptr, y->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(x->__thisptr, y->__thisptr, b->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(x->__thisptr, y->__thisptr, b->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::maxpooling2d(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(ksize)), VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::maxpooling2d(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(ksize)), VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::sin(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sin(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cos(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cos(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::tan(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::tan(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::asin(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::asin(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::acos(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::acos(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::atan(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::atan(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sinh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sinh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cosh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cosh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::tanh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::tanh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::asinh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::asinh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::acosh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::acosh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::atanh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::atanh(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::exp(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::exp(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::square(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::square(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sqrt(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sqrt(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::abs(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::abs(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::erf(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::erf(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cube(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cube(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::log(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::log_sigmoid(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log_sigmoid(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::lgamma(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::lgamma(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::logistic(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::logistic(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Alias to logistic</para>
	/// </summary>
	Expression ^DynetFunctions::sigmoid(Expression ^x) {
		ExceptionWrap(
			return logistic(x);
		)
	}
	Expression ^DynetFunctions::rectify(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::rectify(x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Alias to rectify</para>
	/// </summary>
	Expression ^DynetFunctions::relu(Expression ^x) {
		ExceptionWrap(
			return rectify(x);
		)
	}
	Expression ^DynetFunctions::elu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::elu(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::elu(Expression ^x, float alpha) {
		ExceptionWrap(
			return gcnew Expression(dynet::elu(x->__thisptr, alpha));
		)
	}
	Expression ^DynetFunctions::selu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::selu(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::silu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::silu(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::silu(Expression ^x, float beta) {
		ExceptionWrap(
			return gcnew Expression(dynet::silu(x->__thisptr, beta));
		)
	}
	Expression ^DynetFunctions::log_softmax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log_softmax(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softmax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::softmax(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softmax(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::softmax(x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::sparsemax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sparsemax(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softsign(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::softsign(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::constrained_softmax(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::constrained_softmax(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::pow(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::pow(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::emin(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::min(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::emax(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::max(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::min(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::min(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::max(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::max(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::transpose(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::transpose(x->__thisptr, { 1, 0 }));
		)
	}
	Expression ^DynetFunctions::transpose(Expression ^x, array<int> ^dims) {
		ExceptionWrap(
			return gcnew Expression(dynet::transpose(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(dims))));
		)
	}
	Expression ^DynetFunctions::select_rows(Expression ^x, array<int> ^rs) {
		ExceptionWrap(
			return gcnew Expression(dynet::select_rows(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(rs))));
		)
	}
	Expression ^DynetFunctions::select_cols(Expression ^x, array<int> ^cs) {
		ExceptionWrap(
			return gcnew Expression(dynet::select_cols(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(cs))));
		)
	}
	Expression ^DynetFunctions::sum_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_elems(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sum_dim(Expression ^x, array<int> ^d) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d))));
		)
	}
	Expression ^DynetFunctions::sum_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::sum_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_batches(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cumsum(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cumsum(x->__thisptr, 0));
		)
	}
	Expression ^DynetFunctions::cumsum(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::cumsum(x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::mean_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_elems(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::mean_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::mean_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_batches(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::std_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_elems(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::std_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::std_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_batches(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::moment_elems(Expression ^x, int r) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_elems(x->__thisptr, r));
		)
	}
	Expression ^DynetFunctions::moment_dim(Expression ^x, array<int> ^d, int r, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), r, b));
		)
	}
	Expression ^DynetFunctions::moment_batches(Expression ^x, int r) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_batches(x->__thisptr, r));
		)
	}
	Expression ^DynetFunctions::fold_rows(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::fold_rows(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::fold_rows(Expression ^x, int nrows) {
		ExceptionWrap(
			return gcnew Expression(dynet::fold_rows(x->__thisptr, nrows));
		)
	}
	Expression ^DynetFunctions::pairwise_rank_loss(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::pairwise_rank_loss(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::pairwise_rank_loss(Expression ^x, Expression ^y, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::pairwise_rank_loss(x->__thisptr, y->__thisptr, m));
		)
	}
	Expression ^DynetFunctions::poisson_loss(Expression ^x, int py) {
		ExceptionWrap(
			return gcnew Expression(dynet::poisson_loss(x->__thisptr, py));
		)
	}
	Expression ^DynetFunctions::huber_distance(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::huber_distance(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::huber_distance(Expression ^x, Expression ^y, float c) {
		ExceptionWrap(
			return gcnew Expression(dynet::huber_distance(x->__thisptr, y->__thisptr, c));
		)
	}
	Expression ^DynetFunctions::kmax_pooling(Expression ^x, int k) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmax_pooling(x->__thisptr, k));
		)
	}
	Expression ^DynetFunctions::kmax_pooling(Expression ^x, int k, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmax_pooling(x->__thisptr, k, d));
		)
	}
	Expression ^DynetFunctions::pickneglogsoftmax(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::pickneglogsoftmax(x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::pickneglogsoftmax_batch(Expression ^x, array<int> ^v) {
		ExceptionWrap(
			return gcnew Expression(dynet::pickneglogsoftmax(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v))));
		)
	}
	Expression ^DynetFunctions::hinge(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::hinge(Expression ^x, int v, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(x->__thisptr, v, m));
		)
	}
	Expression ^DynetFunctions::hinge_batch(Expression ^x, array<int> ^v) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v))));
		)
	}
	Expression ^DynetFunctions::hinge_batch(Expression ^x, array<int> ^v, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v)), m));
		)
	}
	Expression ^DynetFunctions::hinge_dim(Expression ^x, array<int> ^v) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v))));
		)
	}
	Expression ^DynetFunctions::hinge_dim(Expression ^x, array<int> ^v, int d, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge_dim(x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v)), d, m));
		)
	}
	Expression ^DynetFunctions::kmh_ngram(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmh_ngram(x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::pick_range(Expression ^x, int s, int e) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick_range(x->__thisptr, s, e));
		)
	}
	Expression ^DynetFunctions::pick_range(Expression ^x, int s, int e, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick_range(x->__thisptr, s, e, d));
		)
	}
	Expression ^DynetFunctions::pickrange(Expression ^x, int s, int e) {
		ExceptionWrap(
			return gcnew Expression(dynet::pickrange(x->__thisptr, s, e));
		)
	}
	Expression ^DynetFunctions::strided_select(Expression ^x, array<int> ^strides, array<int> ^range_from, array<int> ^range_to) {
		ExceptionWrap(
			return gcnew Expression(dynet::strided_select(x->__thisptr, ConvertArrayToVector<int>(strides), ConvertArrayToVector<int>(range_from), ConvertArrayToVector<int>(range_to)));
		)
	}
	Expression ^DynetFunctions::noise(Expression ^x, float stddev) {
		ExceptionWrap(
			return gcnew Expression(dynet::noise(x->__thisptr, stddev));
		)
	}
	Expression ^DynetFunctions::dropout(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout(x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::dropout_batch(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout_batch(x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::dropout_dim(Expression ^x, int d, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout_dim(x->__thisptr, d, p));
		)
	}

	Expression ^DynetFunctions::block_dropout(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::block_dropout(x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::reshape(Expression ^x, array<long> ^d) {
		ExceptionWrap(
			return gcnew Expression(dynet::reshape(x->__thisptr, ConvertArrToDim(d)));
		)
	}
	Expression ^DynetFunctions::reshape(Expression ^x, array<long> ^d, int batchSize) {
		ExceptionWrap(
			dynet::Dim _d(ConvertArrayToVector<long>(d), batchSize);
			return gcnew Expression(dynet::reshape(x->__thisptr, _d));
		)
	}
	Expression ^DynetFunctions::max_dim(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::max_dim(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::max_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::max_dim(x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::min_dim(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::min_dim(x->__thisptr));
		)
	}
	Expression ^DynetFunctions::min_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::min_dim(x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::contract3d_1d(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::contract3d_1d(x->__thisptr, y->__thisptr));
		)
	}
	Expression ^DynetFunctions::logsumexp(List<Expression^> ^l) {
		ExceptionWrap(
			return logsumexp(l->ToArray());
		)
	}
	Expression ^DynetFunctions::logsumexp(... array<Expression^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::logsumexp(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::logsumexp_dim(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::logsumexp_dim(x->__thisptr, 0));
		)
	}
	Expression ^DynetFunctions::logsumexp_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::logsumexp_dim(x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::concatenate_cols(List<Expression ^> ^arr) {
		ExceptionWrap(
			return concatenate_cols(arr->ToArray());
		)
	}
	Expression ^DynetFunctions::concatenate_cols(... array<Expression ^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::concatenate_cols(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::concatenate(List<Expression ^> ^arr) {
		ExceptionWrap(
			return concatenate(arr->ToArray(), 0);
		)
	}
	Expression ^DynetFunctions::concatenate(List<Expression ^> ^arr, int d) {
		ExceptionWrap(
			return concatenate(arr->ToArray(), d);
		)
	}
	Expression ^DynetFunctions::concatenate(... array<Expression ^> ^arr) {
		ExceptionWrap(
			return concatenate(arr, 0);
		)
	}
	Expression ^DynetFunctions::concatenate(array<Expression ^> ^arr, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::concatenate(GetDyExpVector(arr), d));
		)
	}
	Expression ^DynetFunctions::concatenate_to_batch(List<Expression ^> ^arr) {
		ExceptionWrap(
			return concatenate_to_batch(arr->ToArray());
		)
	}
	Expression ^DynetFunctions::concatenate_to_batch(... array<Expression ^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::concatenate_to_batch(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::affine_transform(List<Expression ^> ^arr) {
		ExceptionWrap(
			return affine_transform(arr->ToArray());
		)
	}
	Expression ^DynetFunctions::affine_transform(... array<Expression ^> ^arr) {
		ExceptionWrap(
			return gcnew Expression(dynet::affine_transform(GetDyExpVector(arr)));
		)
	}
	Expression ^DynetFunctions::layer_norm(Expression ^x, Expression ^g, Expression ^b) {
		ExceptionWrap(
			return gcnew Expression(dynet::layer_norm(x->__thisptr, g->__thisptr, b->__thisptr));
		)
	}
	Expression ^DynetFunctions::weight_norm(Expression ^w, Expression ^g) {
		ExceptionWrap(
			return gcnew Expression(dynet::weight_norm(w->__thisptr, g->__thisptr));
		)
	}
	Expression ^DynetFunctions::round(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::round(x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	Expression ^DynetFunctions::ceil(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::ceil(x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	Expression ^DynetFunctions::floor(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::floor(x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	void DynetFunctions::ResetRandomSeed(unsigned newSeed) {
		ExceptionWrap(
			reset_rng(newSeed);
		)
	}
	void DynetFunctions::ShowPoolMemInfo() {
		ExceptionWrap(
			show_pool_mem_info();
		)
	}
	size_t DynetFunctions::HowMuchMemoryDynet() {
		ExceptionWrap(
			size_t totalMem = 0;
		for (Device *d : get_device_manager()->get_devices()) {
			totalMem += (d->pools[0]->get_cap() >> 20);
			totalMem += (d->pools[1]->get_cap() >> 20);
			totalMem += (d->pools[2]->get_cap() >> 20);
			totalMem += (d->pools[3]->get_cap() >> 20);
		}
		return totalMem;
		)
	}
	size_t DynetFunctions::HowMuchUsedMemoryDynet() {
		ExceptionWrap(
			size_t totalMem = 0;
		for (Device *d : get_device_manager()->get_devices()) {
			totalMem += (d->pools[0]->used() >> 20);
			totalMem += (d->pools[1]->used() >> 20);
			totalMem += (d->pools[2]->used() >> 20);
			totalMem += (d->pools[3]->used() >> 20);
		}
		return totalMem;
		)
	}
	/// <summary>
	/// <para>We use the term Device to refer to a Computation Device. A computation device is a piece of hardware performing computation(e.g., CPU, GPU).</para>
	/// <para>Computation devices are identified by string names(e.g., 'CPU', 'GPU:0'). This returns the list of available devices.</para>
	/// <para>Devices have both a processor and an associated memory. Hence, each Parameters, LookupParameters and Expression are tied to devices.</para>
	/// <para>- Parameter and LookupParameters are associated with a device at creation time.</para>
	/// <para>If no device is given at creation time, the default device is assumed.</para>
	/// <para>- Parameter Expressions reside on the same device as their Parameters.</para>
	/// <para>- Other Expressions reside on the same device as the expressions that comprise them.</para>
	/// <para>- An Expression e can be copied across devices using <c>dy.ToDevice(e, name)</c>.</para>
	/// </summary>
	/// <returns>list of available device names (as strings)</returns>
	List<String ^> ^DynetFunctions::GetListOfAvailableDevices() {
		ExceptionWrap(
			List<String ^> ^ret = gcnew List<String ^>((int)get_device_manager()->num_devices());
		for (auto d : get_device_manager()->get_devices())
			ret->Add(gcnew String(d->name.c_str()));
		return ret;
		)
	}
	DeviceInfo ^DynetFunctions::GetDeviceInfo(String ^name) {
		ExceptionWrap(
			Device *d = str2dev(name);
			return gcnew DeviceInfo(d->name, d->device_id, d->type);
		)
	}
	Expression ^DynetFunctions::ToDevice(Expression ^e, String ^device) {
		ExceptionWrap(
			return gcnew Expression(dynet::to_device(e->__thisptr, str2dev(device)));
		)
	}
	/// <summary>
	/// <para>Resets the random seed and the random number generator</para>
	/// </summary>
	/// <param name='seed'>The new random seed</param>
	void DynetFunctions::ResetRandomSeed(int seed) {
		ExceptionWrap(
			reset_rng(seed);
		)
	}
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
}