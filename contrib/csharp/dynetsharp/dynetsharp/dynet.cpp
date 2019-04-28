#include "dynet.h"

namespace dynetsharp {
	///////////////////////////////////////////////////////////
	//////////////// Expression Class /////////////////////////
	///////////////////////////////////////////////////////////
	Expression ^Expression::_multiply(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr * *other->__thisptr);
		)
	}
	Expression ^Expression::_subtract(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr - *other->__thisptr);
		)
	}
	Expression ^Expression::_divide(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr / *other->__thisptr);
		)
	}
	Expression ^Expression::_add(Expression ^other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr + *other->__thisptr);
		)
	}
	Expression ^Expression::_multiply(float other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr * dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_subtract(float other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr - dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_divide(float other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr / dynet::input(*cg, other));
		)
	}
	Expression ^Expression::_add(float other) {
		ExceptionWrap(
			return gcnew Expression(*__thisptr + dynet::input(*cg, other));
		)
	}
	Expression ^Expression::operator*(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) * *x->__thisptr);
		)
	}
	Expression ^Expression::operator-(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) - *x->__thisptr);
		)
	}
	Expression ^Expression::operator/(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) / *x->__thisptr);
		)
	}
	Expression ^Expression::operator+(float other, Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, other) + *x->__thisptr);
		)
	}
	// Private function
	void Expression::GetValue() {
		ExceptionWrap(
			val = new dynet::Tensor(cg->get_value(*__thisptr));
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
			val = new dynet::Tensor(cg->forward(*__thisptr));
		)
	}
	/// <summary>
	/// <para>This runs incremental forward on the entire graph every time it's called.</para>
	/// </summary>
	void Expression::IncrementalForward() {
		ExceptionWrap(
			val = new dynet::Tensor(cg->incremental_forward(*__thisptr));
		)
	}
	/// <summary>
	/// <para>This runs the backward pass based on this expression</para>
	/// </summary>
	void Expression::Backward() {
		ExceptionWrap(
			cg->backward(*__thisptr, false);
		)
	}
	/// <summary>
	/// <para>This runs the backward pass based on this expression</para>
	/// <para>Turn `full` on if you want to retrieve gradients w.r.t. inputs for instance. By default this is turned off, so that the backward pass ignores nodes which have no influence on gradients w.r.t. parameters for efficiency.</para>
	/// </summary>
	/// <param name='full'>Flag whether to compute all gradients (including with respect to constant nodes).</param>
	void Expression::Backward(bool full) {
		ExceptionWrap(
			cg->backward(*__thisptr, full);
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
	array<long> ^Expression::Shape() {
		ExceptionWrap(
			return ConvertDimToArr(__thisptr->dim());
		)
	}
	/// <summary>
	/// <para>Returns the value of the gradient as a Tensor object</para>
	/// <remarks>Make sure to call `backward` on a downstream expression before calling this.</remarks><para/>
	/// <remarks>If the Expression is a constant expression(meaning it's not a function of a parameter), dynet won't compute it's gradient for the sake of efficiency. You need to manually force the gradient computation by adding the agument `full: True` to `backward`</remarks>
	/// </summary>
	Tensor ^Expression::Gradient() {
		ExceptionWrap(
			return gcnew Tensor(__thisptr->gradient());
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
		ndims = shape->Length;
		_dim = new std::vector<long>(ConvertArrayToVector<long>(shape));
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 1-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 1-dim vector</param>
	Tensor::Tensor(array<float> ^arr) {
		ExceptionWrap(
			ndims = 1;
		_dim = new std::vector<long>({ arr->Length });
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
			ndims = 2;
		_dim = new std::vector<long>({ arr->Length, arr[0]->Length });
		_vec = new std::vector<float>(arr->Length * arr[0]->Length);
		// Populate
		int pos = 0;
		for (int i = 0; i < arr->Length; i++)
			for (int j = 0; j < arr[0]->Length; j++)
				(*_vec)[pos++] = arr[i][j];
		__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 3-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 3-dim vector</param>
	Tensor::Tensor(array<array<array<float> ^> ^> ^arr) {
		ExceptionWrap(
			ndims = 3;
			_dim = new std::vector<long>({ arr->Length, arr[0]->Length, arr[0][0]->Length });
			_vec = new std::vector<float>(arr->Length * arr[0]->Length * arr[0][0]->Length);
			// Populate
			int pos = 0;
			for (int i = 0; i < arr->Length; i++)
				for (int j = 0; j < arr[0]->Length; j++)
					for (int k = 0; k < arr[0][0]->Length; k++)
						(*_vec)[pos++] = arr[i][j][k];
			__thisptr = NULL;
		)
	}
	/// <summary>
	/// <para>Initialize the Tensor object with a 4-dim vector</para>
	/// </summary>
	/// <param name='arr'>Values of the 4-dim vector</param>
	Tensor::Tensor(array<array<array<array<float> ^> ^> ^> ^arr) {
		ExceptionWrap(
			ndims = 4;
		_dim = new std::vector<long>({ arr->Length, arr[0]->Length, arr[0][0]->Length, arr[0][0][0]->Length });
		_vec = new std::vector<float>(arr->Length * arr[0]->Length * arr[0][0]->Length * arr[0][0][0]->Length);
		// Populate
		int pos = 0;
		for (int i = 0; i < arr->Length; i++)
			for (int j = 0; j < arr[0]->Length; j++)
				for (int k = 0; k < arr[0][0]->Length; k++)
					for (int l = 0; l < arr[0][0][0]->Length; l++)
						(*_vec)[pos++] = arr[i][j][k][l];
		__thisptr = NULL;
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
		// Return a 2-dimensional vector
		array<array<float> ^> ^ret = gcnew array<array<float> ^>((*_dim)[0]);
		// Create the output
		int curInd = 0;
		for (int i = 0; i < (*_dim)[0]; i++) {
			std::vector<float> curVec(_vec->begin() + curInd, _vec->begin() + curInd + (*_dim)[1]);
			ret[i] = ConvertVectorToArray<float>(curVec);
			// Move over X
			curInd += (*_dim)[1];
		}//n ext i

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
		// Return a 3-dimensional vector
		array<array<array<float> ^> ^> ^ret = gcnew array<array<array<float> ^> ^>((*_dim)[0]);
		// Create the output
		int curInd = 0;
		for (int i = 0; i < (*_dim)[0]; i++) {
			ret[i] = gcnew array<array<float> ^>((*_dim)[1]);
			for (int j = 0; j < (*_dim)[1]; j++) {
				std::vector<float> curVec(_vec->begin() + curInd, _vec->begin() + curInd + (*_dim)[2]);
				ret[i][j] = ConvertVectorToArray<float>(curVec);
				// Move over X
				curInd += (*_dim)[2];
			}// next j
			curInd += (*_dim)[1];
		}// next i

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

		// Return a 4-dimensional vector
		array<array<array<array<float> ^> ^> ^> ^ret = gcnew array<array<array<array<float> ^> ^> ^>((*_dim)[0]);
		// Create the output
		int curInd = 0;
		for (int i = 0; i < (*_dim)[0]; i++) {
			ret[i] = gcnew array<array<array<float> ^> ^>((*_dim)[1]);
			for (int j = 0; j < (*_dim)[1]; j++) {
				ret[i][j] = gcnew array<array<float> ^>((*_dim)[1]);
				for (int k = 0; k < (*_dim)[2]; k++) {
					std::vector<float> curVec(_vec->begin() + curInd, _vec->begin() + curInd + (*_dim)[3]);
					ret[i][j][k] = ConvertVectorToArray<float>(curVec);
					// Move over X
					curInd += (*_dim)[3];
				}// next k
				curInd += (*_dim)[2];
			}// next j
			curInd += (*_dim)[1];
		}// next i

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
			// Find the position - go through each dimension, and multiply by the index
			int actualPos = 0;
		for (int iDim = 0; iDim < vec.size(); iDim++) {
			int curPos = vec[iDim];
			// Check if we are past the end
			if (fCheckBoundaries && curPos >= (*_dim)[iDim])
				throw gcnew IndexOutOfRangeException();

			// Multiply by all following dimensions
			for (int jDim = iDim + 1; jDim < ndims; jDim++)
				curPos *= (*_dim)[jDim];
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
			if (pos->Length != ndims - 1)
				throw gcnew Exception(gcnew String((std::string("Dimensions mismatch. Assumed to have ") + std::to_string(pos->Length + 1) + " dims, when in fact has " + std::to_string(ndims) + ".").c_str()));
		int actualPos = GetActualPosFromArr(pos);
		std::vector<float> vecToCopy = ConvertArrayToVector<float>(value);
		// Copy it in
		std::copy(vecToCopy.begin(), vecToCopy.end(), _vec->begin() + actualPos);
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
		int actualPos = GetActualPosFromArr(pos);
		// Get the subset
		std::vector<float> curVec(_vec->begin() + actualPos, _vec->begin() + actualPos + (*_dim)[ndims - 1]);
		return ConvertVectorToArray<float>(curVec);
		)
	}
	/// <summary>
	/// <para>Get the flattened vector(array) value of a position, using the last input dimension for the column major ordering</para>
	/// </summary>
	/// <param name='pos'>Array pointing to the exact row, each position in the array referring to a dimension</param>
	array<float> ^Tensor::GetFlattenedRowValue(array<int> ^pos) {
		ExceptionWrap(
			return ConvertVectorToArray<float>(_getFlattenedRowValue(pos));
		)
	}
	// Private function which implements the function above
	std::vector<float> Tensor::_getFlattenedRowValue(array<int> ^pos) {
		ExceptionWrap(
			if (pos->Length == 0)
				throw gcnew Exception(gcnew System::String("Position cannot be empty"));
		std::vector<int> vecPos = ConvertArrayToVector<int>(pos);
		// Get the position
		int actualPos = GetActualPosFromArr(vecPos);
		// Get the end (add 1 to the last pos - ignore boundaries, we want the theoretical end)
		vecPos[vecPos.size() - 1]++;
		int endPos = GetActualPosFromArr(vecPos, false);

		// Return that
		std::vector<float> ret(_vec->begin() + actualPos, _vec->begin() + endPos);
		return ret;
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
			return AddParameters(dim, gcnew GlorotInitializer());
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
			return gcnew Parameter(__thisptr->add_parameters(ConvertArrToDim(dim), *pi->__thisptr));
		)
	}
	/// <summary>
	/// <para>Add a lookup parameter to the ParameterCollection, with a given initializer (default is GlorotInitializer)</para>
	/// </summary>
	/// <param name='size'>Number of rows for the collection</param>
	/// <param name='dim'>Shape(dims) of the parameter to add</param>
	LookupParameter ^ParameterCollection::AddLookupParameters(int size, array<long>^ dim) {
		ExceptionWrap(
			return AddLookupParameters(size, dim, gcnew GlorotInitializer());
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
			return gcnew LookupParameter(__thisptr->add_lookup_parameters(size, ConvertArrToDim(dim), *pi->__thisptr));
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
			dynet::Dim dim = ConvertArrToDim(t->Shape());
		return gcnew Parameter(__thisptr->add_parameters(dim, dynet::ParameterInitFromVector(*t->_vec)));
		)
	}
	/// <summary>
	/// <para>Add a LookupParameter to the ParameterCollection, initializing with defined values</para>
	/// </summary>
	/// <param name='t'>Values to initialize parameter with</param>
	LookupParameter ^ParameterCollection::AddLookupParametersFromTensor(Tensor ^t) {
		ExceptionWrap(
			std::vector<long> dimVec = ConvertArrayToVector<long>(t->Shape());
		dynet::Dim dim(std::vector<long>(dimVec.begin() + 1, dimVec.end()));
		return gcnew LookupParameter(__thisptr->add_lookup_parameters(dimVec[0], dim, dynet::ParameterInitFromVector(*t->_vec)));
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
	/// <remarks>Equivalent to doing param.ToExpression().ScalarValue()</remarks>
	/// </summary>
	float Parameter::ScalarValue() {
		ExceptionWrap(
			return ScalarValue(false);
		)
	}
	/// <summary>
	/// <para>Get the scalar value of this parameter</para>
	/// <remarks>Equivalent to doing param.ToExpression().ScalarValue(fRecalculate)</remarks>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	float Parameter::ScalarValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->ScalarValue();
		)
	}
	/// <summary>
	/// <para>Get the vector value of this parameter</para>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks><para/>
	/// <remarks>Equivalent to doing param.ToExpression().VectorValue()</remarks>
	/// </summary>
	array<float> ^Parameter::VectorValue() {
		ExceptionWrap(
			return VectorValue(false);
		)
	}
	/// <summary>
	/// <para>Get the vector value of this parameter</para>
	/// <remarks>Equivalent to doing param.ToExpression().VectorValue(fRecalculate)</remarks><para/>
	/// <remarks>In case of a multidimensional expression, the values are flattened according to a column major ordering</remarks><para/>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	array<float> ^Parameter::VectorValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->VectorValue();
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// <remarks>Equivalent to doing param.ToExpression().TensorValue()</remarks>
	/// </summary>
	Tensor ^Parameter::TensorValue() {
		ExceptionWrap(
			return TensorValue(false);
		)
	}
	/// <summary>
	/// <para>Returns the value of the expression as a Tensor.</para>
	/// <remarks>Equivalent to doing param.ToExpression().TensorValue(fRecalculate)</remarks>
	/// </summary>
	/// <param name='fRecalculate'>Recalculate the computation graph (for static graphs with new inputs) (default: False)</param>
	Tensor ^Parameter::TensorValue(bool fRecalculate) {
		ExceptionWrap(
			return ToExpression()->TensorValue();
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
	/// </summary>
	Expression ^Parameter::ToExpression() {
		ExceptionWrap(
			// Default "update" is true, so return the regular experssion
			if (!__exp || __exp->IsStale())
				__exp = gcnew Expression(dynet::parameter(*cg, *__thisptr));
		return __exp;
		)
	}
	/// <summary>
	/// <para>Returns the parameter as an expression</para>
	/// </summary>
	/// <param name='fUpdate'>Default true, if this is set to False, the parameter won't be updated during the backward pass</param>
	Expression ^Parameter::ToExpression(bool fUpdate) {
		ExceptionWrap(
			// If the fUpdate flag is true, just call the regular function. 
			if (fUpdate) return ToExpression();
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
			if (val->Shape()[iDim] != dim[iDim])
				throw gcnew Exception(gcnew String((std::string("Shape of values and parameter don't match in Parameters.SetValue, Dimension #") + std::to_string(iDim) + ", Input: " + std::to_string(val->Shape()[iDim]) + ", Actual: " + std::to_string(dim[iDim])).c_str()));
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
		array<long> ^shape = t->Shape();
		if (shape[0] != actualRowCount)
			throw gcnew Exception(gcnew String(("Row count mismatch when initializing lookup table from array")));
		// Check the dimensions
		if (shape[1] != __thisptr->get_storage().values[0].d.rows())
			throw gcnew Exception(gcnew String(("Dimension mismatch when initializing lookup table from array")));

		// Put in each row
		for (int iRow = 0; iRow < actualRowCount; iRow++)
			InitRow(iRow, t->GetFlattenedRowValue(gcnew array<int>{ iRow }));
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
		dynet::Expression output = __builderptr->add_input(*__stateptr, *e->__thisptr);
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
		__builderptr->add_input(*__stateptr, *l[0]->__thisptr);
		for (int iExp = 1; iExp < l->Length; iExp++) {
			ret->Add(gcnew Expression(__builderptr->back()));
			__builderptr->add_input(*l[iExp]->__thisptr);
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
		_cg_version++;
		// New memory?
		if (maxOverallMemory && HowMuchMemoryDynet() > maxOverallMemory)
			ResetDynetMemory(initialMemorySize);

		// Create a new graph
		cg = new ComputationGraph();
		cg->set_immediate_compute(fImmediateCompute);
		cg->set_check_validity(fCheckValidity);
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
			return gcnew Expression(dynet::lookup(*cg, *lp->__thisptr, index));
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
			return gcnew Expression(dynet::const_lookup(*cg, *lp->__thisptr, index));
		)
	}
	/// <summary>
	/// <para>Add parameters to the computation graph.</para>
	/// <para>Get the expression objects corresponding to parameters. Gradients for parameters will be computed and used by Optimizers to update.</para>
	/// <remarks>Equivalent to doing Parameter.ToExpression()</remarks>
	/// </summary>
	/// <param name='p'>Parameter object to add to the computation graph</param>
	Expression ^DynetFunctions::parameter(Parameter ^p) {
		ExceptionWrap(
			return p->ToExpression();
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
			return gcnew Expression(dynet::pick(*exp->__thisptr, index));
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
			return gcnew Expression(dynet::pick(*exp->__thisptr, index, dim));
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a scalar value</para>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(float num) {
		ExceptionWrap(
			float *val = new float(num);
			return gcnew Expression(dynet::input(*cg, val), val);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 1-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(array<float>^ num) {
		ExceptionWrap(
			std::vector<float> *vec = new std::vector<float>(ConvertArrayToVector<float>(num));
			return gcnew Expression(dynet::input(*cg, { (unsigned)num->Length }, vec), vec);
		) 
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a 2-dim vector value</para>
	/// <remarks>Better to use inputTensor for initializing vector expressions</remarks>
	/// </summary>
	/// <param name='num'>Value of Expression</param>
	Expression ^DynetFunctions::input(array<array<float>^>^ num) {
		ExceptionWrap(
			std::vector<float> vec;
		for (int iVec = 0; iVec < num->Length; iVec++)
			for (int iItem = 0; iItem < num[iVec]->Length; iItem++)
				vec.push_back((real)num[iVec][iItem]);
		// Convert to pointer, and send that
		std::vector<float> *valVec = new std::vector<float>(vec);
		return gcnew Expression(dynet::input(*cg, { (unsigned)num->Length, (unsigned)num[0]->Length }, valVec), valVec);
		)
	}
	/// <summary>
	/// <para>Create an Expression object on the computation graph with a the value of the Tensor</para>
	/// </summary>
	/// <param name='tensor'>Value of Expression</param>
	Expression ^DynetFunctions::inputTensor(Tensor ^tensor) {
		ExceptionWrap(
			return gcnew Expression(dynet::input(*cg, ConvertArrToDim(tensor->Shape()), tensor->_vec));
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
	Expression ^DynetFunctions::zeroes(array<long> ^dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::zeroes(*cg, ConvertArrToDim(dim)));
		)
	}
	/// <summary>
/// <para>Create an input full of zeros</para>
/// <para>Create an input full of zeros, sized according to dimensions `dim`</para>
/// </summary>
/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::zeros(array<long> ^dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::zeros(*cg, ConvertArrToDim(dim)));
		)
	}
	/// <summary>
	/// <para>Create an input full of ones</para>
	/// <para>Create an input full of ones, sized according to dimensions `dim`</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::ones(array<long> ^dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::ones(*cg, ConvertArrToDim(dim)));
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
			return gcnew Expression(dynet::constant(*cg, ConvertArrToDim(dim), val));
		)
	}
	/// <summary>
	/// <para>Create a random normal vector</para>
	/// <para>Create a vector distributed according to normal distribution with mean (default: 0), variance (default: 1).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::random_normal(array<long> ^dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::random_normal(*cg, ConvertArrToDim(dim)));
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
			return gcnew Expression(dynet::random_normal(*cg, ConvertArrToDim(dim), mean, stddev));
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
			return gcnew Expression(dynet::random_bernoulli(*cg, ConvertArrToDim(dim), p));
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
			return gcnew Expression(dynet::random_bernoulli(*cg, ConvertArrToDim(dim), p, scale));
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
			return gcnew Expression(dynet::random_uniform(*cg, ConvertArrToDim(dim), left, right));
		)
	}
	/// <summary>
	/// <para>Create a random Gumbel sampled vector</para>
	/// <para>Create a vector distributed according to a Gumbel distribution with the specified parameters. (Currently only the defaults of mu=0.0 and beta=1.0 supported).</para>
	/// </summary>
	/// <param name='dim'>Dimensions of the expression</param>
	Expression ^DynetFunctions::random_gumbel(array<long> ^dim) {
		ExceptionWrap(
			return gcnew Expression(dynet::random_gumbel(*cg, ConvertArrToDim(dim)));
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
			if (mu != 0.0 || beta != 1.0)
				throw gcnew Exception(gcnew String("Currently only paramters of mu=0.0 and beta=1.0 are supported."));
		return gcnew Expression(dynet::random_gumbel(*cg, ConvertArrToDim(dim), mu, beta));
		)
	}
	/// <summary>
	/// <para>Flip gradient</para>
	/// <para>This node has no effect on the forward pass, but takes negative on backprop process. This operation is widely used in adversarial networks.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::flip_gradient(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::flip_gradient(*x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Scale gradient</para>
	/// <para>This node scales the gradient by a constant on backprop, with no effect on the forward pass.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::scale_gradient(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::scale_gradient(*x->__thisptr));
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
			return gcnew Expression(dynet::scale_gradient(*x->__thisptr, lambd));
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
			return gcnew Expression(dynet::cdiv(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::cmult(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::colwise_add(*x->__thisptr, *y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Matrix Inverse</para>
	/// <para>Takes the inverse of a matrix. Note that back-propagating through an inverted matrix can also be the source of stability problems sometimes.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::inverse(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::inverse(*x->__thisptr));
		)
	}
	/// <summary>
	/// <para>Log determinant</para>
	/// <para>Takes the log of the determinant of a matrix.</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::logdet(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::logdet(*x->__thisptr));
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
			return gcnew Expression(dynet::trace_of_product(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::dot_product(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::circ_conv(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::circ_corr(*x->__thisptr, *y->__thisptr));
		)
	}
	/// <summary>
	/// <para>Squared norm</para>
	/// <para>The squared norm of the values of `x`: |x|^2=sum(x_i^2)</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::squared_norm(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::squared_norm(*x->__thisptr));
		)
	}
	/// <summary>
	/// <para>L2 norm</para>
	/// <para>The l2 norm of the values of `x`: |x|=sqrt(sum(x_i^2))</para>
	/// </summary>
	/// <param name='x'>Input expression</param>
	Expression ^DynetFunctions::l2_norm(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::l2_norm(*x->__thisptr));
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
			return gcnew Expression(dynet::squared_distance(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::l1_distance(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::binary_log_loss(*x->__thisptr, *y->__thisptr));
		)
	}
	//TODO: Docs
	Expression ^DynetFunctions::filter1d_narrow(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::filter1d_narrow(*x->__thisptr, *y->__thisptr));
		)
	}
	//TODO: Docs
	Expression ^DynetFunctions::conv2d(Expression ^x, Expression ^y, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(*x->__thisptr, *y->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::conv2d(Expression ^x, Expression ^y, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(*x->__thisptr, *y->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(*x->__thisptr, *y->__thisptr, *b->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::conv2d_bias(Expression ^x, Expression ^y, Expression ^b, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::conv2d(*x->__thisptr, *y->__thisptr, *b->__thisptr, VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride) {
		ExceptionWrap(
			return gcnew Expression(dynet::maxpooling2d(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(ksize)), VecToUInt(ConvertArrayToVector<int>(stride))));
		)
	}
	Expression ^DynetFunctions::maxpooling2d(Expression ^x, array<int> ^ksize, array<int> ^stride, bool is_valid) {
		ExceptionWrap(
			return gcnew Expression(dynet::maxpooling2d(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(ksize)), VecToUInt(ConvertArrayToVector<int>(stride)), is_valid));
		)
	}
	Expression ^DynetFunctions::sin(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sin(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cos(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cos(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::tan(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::tan(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::asin(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::asin(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::acos(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::acos(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::atan(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::atan(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sinh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sinh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cosh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cosh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::tanh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::tanh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::asinh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::asinh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::acosh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::acosh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::atanh(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::atanh(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::exp(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::exp(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::square(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::square(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sqrt(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sqrt(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::abs(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::abs(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::erf(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::erf(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cube(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cube(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::log(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::log_sigmoid(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log_sigmoid(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::lgamma(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::lgamma(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::logistic(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::logistic(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::rectify(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::rectify(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::elu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::elu(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::elu(Expression ^x, float alpha) {
		ExceptionWrap(
			return gcnew Expression(dynet::elu(*x->__thisptr, alpha));
		)
	}
	Expression ^DynetFunctions::selu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::selu(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::silu(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::silu(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::silu(Expression ^x, float beta) {
		ExceptionWrap(
			return gcnew Expression(dynet::silu(*x->__thisptr, beta));
		)
	}
	Expression ^DynetFunctions::log_softmax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::log_softmax(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softmax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::softmax(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softmax(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::softmax(*x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::sparsemax(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sparsemax(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::softsign(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::softsign(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::constrained_softmax(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::constrained_softmax(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::pow(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::pow(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::emin(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::min(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::emax(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::max(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::min(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::min(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::max(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::max(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::transpose(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::transpose(*x->__thisptr, { 0,1 }));
		)
	}
	Expression ^DynetFunctions::transpose(Expression ^x, array<int> ^dims) {
		ExceptionWrap(
			return gcnew Expression(dynet::transpose(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(dims))));
		)
	}
	Expression ^DynetFunctions::select_rows(Expression ^x, array<int> ^rs) {
		ExceptionWrap(
			return gcnew Expression(dynet::select_rows(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(rs))));
		)
	}
	Expression ^DynetFunctions::select_cols(Expression ^x, array<int> ^cs) {
		ExceptionWrap(
			return gcnew Expression(dynet::select_cols(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(cs))));
		)
	}
	Expression ^DynetFunctions::sum_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_elems(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::sum_dim(Expression ^x, array<int> ^d) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d))));
		)
	}
	Expression ^DynetFunctions::sum_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::sum_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::sum_batches(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::cumsum(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::cumsum(*x->__thisptr, 0));
		)
	}
	Expression ^DynetFunctions::cumsum(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::cumsum(*x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::mean_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_elems(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::mean_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::mean_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::mean_batches(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::std_elems(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_elems(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::std_dim(Expression ^x, array<int> ^d, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), b));
		)
	}
	Expression ^DynetFunctions::std_batches(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::std_batches(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::moment_elems(Expression ^x, int r) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_elems(*x->__thisptr, r));
		)
	}
	Expression ^DynetFunctions::moment_dim(Expression ^x, array<int> ^d, int r, bool b) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(d)), r, b));
		)
	}
	Expression ^DynetFunctions::moment_batches(Expression ^x, int r) {
		ExceptionWrap(
			return gcnew Expression(dynet::moment_batches(*x->__thisptr, r));
		)
	}
	Expression ^DynetFunctions::fold_rows(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::fold_rows(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::fold_rows(Expression ^x, int nrows) {
		ExceptionWrap(
			return gcnew Expression(dynet::fold_rows(*x->__thisptr, nrows));
		)
	}
	Expression ^DynetFunctions::pairwise_rank_loss(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::pairwise_rank_loss(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::pairwise_rank_loss(Expression ^x, Expression ^y, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::pairwise_rank_loss(*x->__thisptr, *y->__thisptr, m));
		)
	}
	Expression ^DynetFunctions::poisson_loss(Expression ^x, int py) {
		ExceptionWrap(
			return gcnew Expression(dynet::poisson_loss(*x->__thisptr, py));
		)
	}
	Expression ^DynetFunctions::huber_distance(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::huber_distance(*x->__thisptr, *y->__thisptr));
		)
	}
	Expression ^DynetFunctions::huber_distance(Expression ^x, Expression ^y, float c) {
		ExceptionWrap(
			return gcnew Expression(dynet::huber_distance(*x->__thisptr, *y->__thisptr, c));
		)
	}
	Expression ^DynetFunctions::kmax_pooling(Expression ^x, int k) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmax_pooling(*x->__thisptr, k));
		)
	}
	Expression ^DynetFunctions::kmax_pooling(Expression ^x, int k, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmax_pooling(*x->__thisptr, k, d));
		)
	}
	Expression ^DynetFunctions::pickneglogsoftmax(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::pickneglogsoftmax(*x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::hinge(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(*x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::hinge(Expression ^x, int v, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge(*x->__thisptr, v, m));
		)
	}
	Expression ^DynetFunctions::hinge_dim(Expression ^x, array<int> ^v) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v))));
		)
	}
	Expression ^DynetFunctions::hinge_dim(Expression ^x, array<int> ^v, int d, float m) {
		ExceptionWrap(
			return gcnew Expression(dynet::hinge_dim(*x->__thisptr, VecToUInt(ConvertArrayToVector<int>(v)), d, m));
		)
	}
	Expression ^DynetFunctions::kmh_ngram(Expression ^x, int v) {
		ExceptionWrap(
			return gcnew Expression(dynet::kmh_ngram(*x->__thisptr, v));
		)
	}
	Expression ^DynetFunctions::pick_range(Expression ^x, int s, int e) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick_range(*x->__thisptr, s, e));
		)
	}
	Expression ^DynetFunctions::pick_range(Expression ^x, int s, int e, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::pick_range(*x->__thisptr, s, e, d));
		)
	}
	Expression ^DynetFunctions::pickrange(Expression ^x, int s, int e) {
		ExceptionWrap(
			return gcnew Expression(dynet::pickrange(*x->__thisptr, s, e));
		)
	}
	Expression ^DynetFunctions::strided_select(Expression ^x, array<int> ^strides, array<int> ^range_from, array<int> ^range_to) {
		ExceptionWrap(
			return gcnew Expression(dynet::strided_select(*x->__thisptr, ConvertArrayToVector<int>(strides), ConvertArrayToVector<int>(range_from), ConvertArrayToVector<int>(range_to)));
		)
	}
	Expression ^DynetFunctions::noise(Expression ^x, float stddev) {
		ExceptionWrap(
			return gcnew Expression(dynet::noise(*x->__thisptr, stddev));
		)
	}
	Expression ^DynetFunctions::dropout(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout(*x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::dropout_batch(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout_batch(*x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::dropout_dim(Expression ^x, int d, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::dropout_dim(*x->__thisptr, d, p));
		)
	}

	Expression ^DynetFunctions::block_dropout(Expression ^x, float p) {
		ExceptionWrap(
			return gcnew Expression(dynet::block_dropout(*x->__thisptr, p));
		)
	}
	Expression ^DynetFunctions::reshape(Expression ^x, array<long> ^d) {
		ExceptionWrap(
			return gcnew Expression(dynet::reshape(*x->__thisptr, ConvertArrToDim(d)));
		)
	}
	Expression ^DynetFunctions::max_dim(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::max_dim(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::max_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::max_dim(*x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::min_dim(Expression ^x) {
		ExceptionWrap(
			return gcnew Expression(dynet::min_dim(*x->__thisptr));
		)
	}
	Expression ^DynetFunctions::min_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::min_dim(*x->__thisptr, d));
		)
	}
	Expression ^DynetFunctions::contract3d_1d(Expression ^x, Expression ^y) {
		ExceptionWrap(
			return gcnew Expression(dynet::contract3d_1d(*x->__thisptr, *y->__thisptr));
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
			return gcnew Expression(dynet::logsumexp_dim(*x->__thisptr, 0));
		)
	}
	Expression ^DynetFunctions::logsumexp_dim(Expression ^x, int d) {
		ExceptionWrap(
			return gcnew Expression(dynet::logsumexp_dim(*x->__thisptr, d));
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
			return gcnew Expression(dynet::layer_norm(*x->__thisptr, *g->__thisptr, *b->__thisptr));
		)
	}
	Expression ^DynetFunctions::weight_norm(Expression ^w, Expression ^g) {
		ExceptionWrap(
			return gcnew Expression(dynet::weight_norm(*w->__thisptr, *g->__thisptr));
		)
	}
	Expression ^DynetFunctions::round(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::round(*x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	Expression ^DynetFunctions::ceil(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::ceil(*x->__thisptr, (dynet::GradientMode)gm));
		)
	}
	Expression ^DynetFunctions::floor(Expression ^x, GradientMode gm) {
		ExceptionWrap(
			return gcnew Expression(dynet::floor(*x->__thisptr, (dynet::GradientMode)gm));
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
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////
}