#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <mkl.h>
#include <pybind11/stl.h>

using namespace std;
namespace py = pybind11;

class Matrix {
public:
    Matrix(size_t nrow, size_t ncol) : m_nrow(nrow), m_ncol(ncol)
    {
        size_t nelement = nrow * ncol;
        m_buffer = new double[nelement];
    }
    // TODO: copy and move constructors and assignment operators.

    // No bound check.
    double   operator() (size_t row, size_t col) const { return m_buffer[row*m_ncol + col]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[row*m_ncol + col]; }
    double* get_buffer() { return m_buffer; }
    void set_buffer(const std::vector<int> &v)
    {
        for(size_t i=0;i<(m_ncol*m_nrow);i++)
            m_buffer[i] = v[i];
    }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    bool operator==(Matrix &rhs)
    {
        if (m_nrow!=rhs.nrow()&&m_ncol!=rhs.ncol())
            return 0;
        bool flag = 1;
        double* temp_buffer = rhs.get_buffer();
        for (size_t i=0;i<m_nrow*m_ncol;i++)
        {
            if(m_buffer[i]!=temp_buffer[i])
        {
            flag = 0;
            break;
        }
        }
        return flag;
    }

    friend ostream &operator<<(ostream &ostr, Matrix const &mat)
    {
        for (size_t i = 0; i < mat.nrow(); ++i)
        {
            ostr << endl
                << " ";
            for (size_t j = 0; j < mat.ncol(); ++j)
            {
                ostr << mat(i, j);
            }
        }
        return ostr;
    }


private:
    size_t m_nrow;
    size_t m_ncol;
    double* m_buffer;
};

vector<double> vector_linspace(int start, int stop) //create an increment vector
{
	vector<double> vec;
	for (int i=start; i<stop; i++)
	{
		vec.push_back((double)i);
	}
	return vec;
}

Matrix matrix_linspace(int row, int col)	//create an increment matrix with row majoring
{
	Matrix amatrix(row,col);
	for (size_t i=0; i<amatrix.nrow(); i++) // the i-th row
	{
		for (size_t j=0; j<amatrix.ncol(); j++) // the j-th column
		{
			amatrix(i,j) = i*amatrix.ncol() + j;
		}
	}
	return amatrix;
}

vector<double> dot_vector (Matrix const & mat, vector<double> const & vec)
{	//vector and martix multiplication
	if (mat.ncol() != vec.size())
	{
		throw std::out_of_range("matrix column differs from vector size");
	}

	std::vector<double> ret(mat.nrow());

	for (size_t i=0; i<mat.nrow(); ++i)
	{
		double v = 0;
		for (size_t j=0; j<mat.ncol(); ++j)
		{
			v += mat(i,j) * vec[j];
		}
		ret[i] = v;
	}

	return ret;
}


Matrix multiply_naive (Matrix const & mat1, Matrix const & mat2)
{	//matrix and matrix multiplication
	if (mat1.ncol() != mat2.nrow())
	{
		throw std::out_of_range(
			"the number of first matrix column "
			"differs from that of second matrix row");
	}

	Matrix ret(mat1.nrow(), mat2.ncol());

	for (size_t i=0; i<ret.nrow(); ++i)
	{
		for (size_t k=0; k<ret.ncol(); ++k)
		{
			double v = 0;
			for (size_t j=0; j<mat1.ncol(); ++j)
			{
				v += mat1(i,j) * mat2(j,k);
			}
			ret(i,k) = v;
		}
	}

	return ret;
}

Matrix multiply_mkl (Matrix  & mat1, Matrix & mat2)
{	// matrix and matrix multiplication function implement with cblas
    Matrix mat_result(mat1.nrow(), mat2.ncol());

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        mat1.nrow(),
        mat2.ncol(),
        mat1.ncol(),
        1.0,
        mat1.get_buffer(),
        mat1.ncol(), //lda
        mat2.get_buffer(),
        mat2.ncol(), //ldb
        0.0,
        mat_result.get_buffer(),
        mat_result.ncol()
    );

    return mat_result;
}

/*
int main() 
{

}
*/

PYBIND11_MODULE(_matrix, m)
{
    m.doc() = "pybind11 matrix plugin"; // optional module docstring
    m.def("multiply_mkl", &multiply_mkl, "MKL matrix multiplication");
    m.def("multiply_naive", &multiply_naive, "naive matrix multiplication");
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("get_buffer", &Matrix::get_buffer)
        .def("set_buffer", &Matrix::set_buffer)
        .def("__getitem__", [](const Matrix &self, pair<size_t, size_t> idx) { return self(idx.first, idx.second); })
        .def("__setitem__", [](Matrix &self, pair<size_t, size_t> idx, double val) { self(idx.first, idx.second) = val; })
	.def("__eq__",[](Matrix &lhs, Matrix &rhs){return lhs==rhs;});
}
