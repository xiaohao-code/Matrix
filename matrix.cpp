#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include "Matrix.h"
using namespace std;

Matrix::Matrix()
{
	row = 0;
	column = 0;
	data = NULL;
}

Matrix::Matrix(int m, int n)
{
	row = m;
	column = n;
	data = NULL;
	if (!(data = new double* [m]))
	{
		cout << "Error:out of memory" << endl;
		exit(1);
	}
	for (int i = 0; i < m; i++)
	{
		if (!(data[i] = new double[n]()))//分配空间的同时初始化为0
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
	}
}

Matrix::Matrix(int m, int n, double value)
{
	row = m;
	column = n;
	data = NULL;
	if (!(data = new double* [m]))
	{
		cout << "Error:out of memory" << endl;
		exit(1);
	}
	for (int i = 0; i < m; i++)
	{
		data[i] = NULL;
		if (!(data[i] = new double[n]))
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
		for (int j = 0; j < n; j++)
		{
			data[i][j] = value;
		}
	}
}

Matrix::Matrix(int m, int n, double* _data)
{
	row = m;
	column = n;
	data = NULL;
	if (!(data = new double* [m]))
	{
		cout << "Error:out of memory" << endl;
		exit(1);
	}
	for (int i = 0; i < m; i++)
	{
		data[i] = NULL;
		if (!(data[i] = new double[n]))
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
		for (int j = 0; j < n; j++)
		{
			data[i][j] = _data[i * n + j];
		}
	}
}

Matrix::Matrix(const Matrix& mat)
{
	row = mat.row;
	column = mat.column;
	data = NULL;
	if (!(data = new double* [row]))
	{
		cout << "Error:out of memory" << endl;
		exit(1);
	}
	for (int i = 0; i < row; i++)
	{
		data[i] = NULL;
		if (!(data[i] = new double[column]))
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
		for (int j = 0; j < column; j++)
		{
			data[i][j] = mat.data[i][j];
		}
	}
}

Matrix::Matrix(Matrix* mat)
{
	row = mat->row;
	column = mat->column;
	data = NULL;
	if (!(data = new double* [row]))
	{
		cout << "Error:out of memory" << endl;
		exit(1);
	}
	for (int i = 0; i < row; i++)
	{
		data[i] = NULL;
		if (!(data[i] = new double[column]))
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
		for (int j = 0; j < column; j++)
		{
			data[i][j] = mat->data[i][j];
		}
	}
}

Matrix::~Matrix()
{
	for (int i = 0; i < row; i++)
	{
		delete[] data[i];
	}
	delete[] data;
}

void Matrix::print()
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			cout << data[i][j] << "  ,";
		}
		cout << endl;
	}
}

void Matrix::save(ofstream& outfile)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			outfile << data[i][j] << "\t";
		}
		outfile << endl;
	}
}

Matrix* Matrix::operator+(const Matrix& mat2)
{
	if (this->row == mat2.row && this->column == mat2.column)
	{
		Matrix* result = new Matrix(this->row, this->column);
		for (int i = 0; i < this->row; i++)
		{
			for (int j = 0; j < this->column; j++)
			{
				result->data[i][j] = this->data[i][j] + mat2.data[i][j];
			}
		}
		return result;
	}
	else
	{
		cout << "Error:The row(column) of the matiexs are different" << endl;
		exit(1);
	}
}

Matrix* Matrix::operator-(const Matrix& mat2)
{
	if (this->row == mat2.row && this->column == mat2.column)
	{
		Matrix* result = new Matrix(this->row, this->column);
		for (int i = 0; i < this->row; i++)
		{
			for (int j = 0; j < this->column; j++)
			{
				result->data[i][j] = this->data[i][j] - mat2.data[i][j];
			}
		}
		return result;
	}
	else
	{
		cout << "Error:The row(column) of the matiexs are different" << endl;
		exit(1);
	}
}

Matrix* Matrix::operator*(const Matrix& mat2)
{
	if (this->column != mat2.row)
	{
		cout << "Erroe:Matrix cannot be multiplied" << endl;
		exit(1);
	}
	else
	{
		Matrix* result = new Matrix(this->row, mat2.column);
		int mid = this->column;
		for (int i = 0; i < result->row; i++)
		{
			for (int j = 0; j < result->column; j++)
			{
				double midvalue = 0;
				for (int k = 0; k < mid; k++)
				{
					midvalue += this->data[i][k] * mat2.data[k][j];
				}
				result->data[i][j] = midvalue;
			}
		}
		return result;
	}
}

Matrix* Matrix::operator*(double num)
{
	Matrix* result = new Matrix(this->row, this->column);
	for (int i = 0; i < this->row; i++)
	{
		for (int j = 0; j < this->column; j++)
		{
			result->data[i][j] = this->data[i][j] * num;
		}
	}
	return result;
}

Matrix* Matrix::T()
{
	Matrix* result = new Matrix(this->column, this->row);
	for (int i = 0; i < result->row; i++)
	{
		for (int j = 0; j < result->column; j++)
		{
			result->data[i][j] = this->data[j][i];
		}
	}
	return result;
}

Matrix* Matrix::abs()
{
	Matrix* result = new Matrix(this->row, this->column);
	for (int i = 0; i < result->row; i++)
	{
		for (int j = 0; j < result->column; j++)
		{
			result->data[i][j] = fabs(this->data[i][j]);
		}
	}
	return result;
}

Matrix* Matrix::sum(int setting)
{
	if (setting == 1)
	{
		Matrix* result = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			result->data[i][0] = 0;
			for (int j = 0; j < this->column; j++)
			{
				result->data[i][0] += this->data[i][j];
			}
		}
		return result;
	}
	else if (setting == 2)
	{
		Matrix* result = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			result->data[0][j] = 0;
			for (int i = 0; i < this->row; i++)
			{
				result->data[0][j] += this->data[i][j];
			}
		}
		return result;
	}
	else
	{
		cout << "Error:Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::mean(int setting)
{
	if (setting == 1)
	{
		int num = this->column;
		Matrix* mean = new Matrix(this->row, 1);
		Matrix* sum = this->sum(setting);//按行求和
		for (int i = 0; i < this->row; i++)
		{
			mean->data[i][0] = sum->data[i][0] / num;
		}
		delete sum;
		return mean;
	}
	else if (setting == 2)
	{
		int num = this->row;
		Matrix* mean = new Matrix(1, this->column);
		Matrix* sum = this->sum(setting);
		for (int j = 0; j < this->column; j++)
		{
			mean->data[0][j] = sum->data[0][j] / num;
		}
		delete sum;
		return mean;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::std(int setting)
{
	if (setting == 1)
	{
		Matrix* mean = this->mean(setting);//先求均值
		int num = this->column;
		Matrix* std = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			double sum = 0;
			for (int j = 0; j < this->column; j++)
			{
				double diff = this->data[i][j] - mean->data[i][0];
				sum += diff * diff;
			}
			std->data[i][0] = sqrt(sum / (num - 1.0));
		}
		delete mean;
		return std;
	}
	else if (setting == 2)
	{
		Matrix* mean = this->mean(setting);//先求均值
		int num = this->row;
		Matrix* std = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			double sum = 0;
			for (int i = 0; i < this->row; i++)
			{
				double diff = this->data[i][j] - mean->data[0][j];
				sum += diff * diff;
			}
			std->data[0][j] = sqrt(sum / (num - 1.0));
		}
		delete mean;
		return std;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::swap(int line1, int line2, int setting)
{
	line1 = line1 - 1;//由于索引从0开始
	line2 = line2 - 1;
	if (setting == 1)
	{
		if (line1 < this->row && line2 < this->row)
		{
			Matrix* result = new Matrix(*this);//复制矩阵
			for (int j = 0; j < result->column; j++)
			{
				result->data[line1][j] = this->data[line2][j];
				result->data[line2][j] = this->data[line1][j];
			}
			return result;
		}
		else
		{
			cout << "Error:The row exchanged exceed the limit" << endl;
			exit(1);
		}
	}
	else if (setting == 2)
	{
		if (line1 < this->column && line2 < this->column)
		{
			Matrix* result = new Matrix(*this);//复制矩阵
			for (int i = 0; i < this->row; i++)
			{
				result->data[i][line1] = this->data[i][line2];
				result->data[i][line2] = this->data[i][line1];
			}
			return result;
		}
		else
		{
			cout << "Error:The column exchanged exceed the limit" << endl;
			exit(1);
		}
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::min_position(int setting)
{
	if (setting == 1)
	{
		Matrix* result = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			double min = DBL_MAX;//正无穷
			for (int j = 0; j < this->column; j++)
			{
				if (this->data[i][j] < min)
				{
					min = this->data[i][j];
					result->data[i][0] = j;
				}
			}
		}
		return result;
	}
	else if (setting == 2)
	{
		Matrix* result = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			double min = DBL_MAX;//正无穷
			for (int i = 0; i < this->row; i++)
			{
				if (this->data[i][j] < min)
				{
					min = this->data[i][j];
					result->data[0][j] = i;
				}
			}
		}
		return result;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::max_position(int setting)
{
	if (setting == 1)
	{
		Matrix* result = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			double max = -DBL_MAX;//负无穷
			for (int j = 0; j < this->column; j++)
			{
				if (this->data[i][j] > max)
				{
					max = this->data[i][j];
					result->data[i][0] = j;
				}
			}
		}
		return result;
	}
	else if (setting == 2)
	{
		Matrix* result = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			double max = -DBL_MAX;//负无穷
			for (int i = 0; i < this->row; i++)
			{
				if (this->data[i][j] > max)
				{
					max = this->data[i][j];
					result->data[0][j] = i;
				}
			}
		}
		return result;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}

}

Matrix* Matrix::min_value(int setting)
{
	if (setting == 1)
	{
		Matrix* min_p = this->min_position(setting);
		Matrix* result = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			int index = (int)min_p->data[i][0];
			result->data[i][0] = this->data[i][index];
		}
		delete min_p;
		return result;
	}
	else if (setting == 2)
	{
		Matrix* min_p = this->min_position(setting);
		Matrix* result = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			int index = (int)min_p->data[0][j];
			result->data[0][j] = this->data[index][j];
		}
		delete min_p;
		return result;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::max_value(int setting)
{
	if (setting == 1)
	{
		Matrix* max_p = this->max_position(setting);
		Matrix* result = new Matrix(this->row, 1);
		for (int i = 0; i < this->row; i++)
		{
			int index = (int)max_p->data[i][0];
			result->data[i][0] = this->data[i][index];
		}
		delete max_p;
		return result;
	}
	else if (setting == 2)
	{
		Matrix* max_p = this->max_position(setting);
		Matrix* result = new Matrix(1, this->column);
		for (int j = 0; j < this->column; j++)
		{
			int index = (int)max_p->data[0][j];
			result->data[0][j] = this->data[index][j];
		}
		delete max_p;
		return result;
	}
	else
	{
		cout << "Error,Wrong setting(1/2)" << endl;
		exit(1);
	}
}

Matrix* Matrix::cut(int row_head, int row_tail, int column_head, int column_tail)
{
	if (row_tail < 0)
	{
		if (row_tail == -1)
		{
			row_tail = this->row;
		}
		else
		{
			cout << "Error:row_tail exceed the limit" << endl;
			exit(1);
		}
	}
	if (row_head < 0)
	{
		if (row_head == -1)
		{
			row_head = this->row;
		}
		else
		{
			cout << "Error:row_head exceed the limit" << endl;
			exit(1);
		}
	}
	if (column_tail < 0)
	{
		if (column_tail == -1)
		{
			column_tail = this->column;
		}
		else
		{
			cout << "Error:column_tail exceed the limit" << endl;
			exit(1);
		}
	}
	if (column_head < 0)
	{
		if (column_head == -1)
		{
			column_head = this->column;
		}
		else
		{
			cout << "Error:column_head exceed the limit" << endl;
			exit(1);
		}
	}
	if (row_tail > this->row || column_tail > this->column)
	{
		cout << "Error:Exceed the limits" << endl;
		exit(1);
	}
	else
	{
		if (row_head > row_tail || column_head > column_tail)
		{
			cout << "Error:Wrong Parameters" << endl;
			exit(1);
		}
		else
		{
			row_head = row_head - 1;
			column_head = column_head - 1;
			Matrix* result = new Matrix(row_tail - row_head, column_tail - column_head);
			for (int i = 0; i < row_tail - row_head; i++)
			{
				for (int j = 0; j < column_tail - column_head; j++)
				{
					result->data[i][j] = this->data[i + row_head][j + column_head];
				}
			}
			return result;
		}
	}
}

Matrix* Matrix::inverse()
{
	if (this->column != this->row)
	{
		cout << "Error:Matrix must be square" << endl;
		exit(1);
	}
	Matrix* result = NULL;
	Matrix* L = new Matrix(this->row, this->column);//单位下三角阵
	Matrix* U = new Matrix(this->row, this->column);//上三角阵
	Matrix* L_inv = new Matrix(this->row, this->column);//单位下三角阵的逆
	Matrix* U_inv = new Matrix(this->row, this->column);//上三角阵的逆
	for (int i = 0; i < this->row; i++)
	{
		L->data[i][i] = 1;//对角线赋1
	}
	for (int j = 0; j < this->column; j++)
	{
		U->data[0][j] = this->data[0][j];//U第一行与原矩阵相同
	}
	for (int i = 1; i < this->row; i++)
	{
		L->data[i][0] = this->data[i][0] / U->data[0][0];//L第一列元素
	}
	//进行LU三角分解
	for (int i = 1; i < this->row; i++)
	{
		for (int j = i; j < this->column; j++) //求U
		{
			double s = 0;
			for (int k = 0; k < i; k++)
			{
				s += L->data[i][k] * U->data[k][j];
			}
			U->data[i][j] = this->data[i][j] - s;
		}
		for (int d = i; d < this->row; d++) //求L
		{
			double s = 0;
			for (int k = 0; k < i; k++)
			{
				s += L->data[d][k] * U->data[k][i];
			}
			L->data[d][i] = (this->data[d][i] - s) / U->data[i][i];
		}
	}
	//对LU求逆
	for (int j = 0; j < this->column; j++) //求L的逆
	{
		for (int i = j; i < this->row; i++)
		{
			if (i == j)
				L_inv->data[i][j] = 1 / L->data[i][j];
			else if (i < j)
				L_inv->data[i][j] = 0;
			else
			{
				double s = 0;
				for (int k = j; k < i; k++)
				{
					s += L->data[i][k] * L_inv->data[k][j];
				}
				L_inv->data[i][j] = -L_inv->data[j][j] * s;
			}
		}
	}
	for (int i = 0; i < this->row; i++) //求U的逆
	{
		for (int j = i; j >= 0; j--)
		{
			if (i == j)
				U_inv->data[j][i] = 1 / U->data[j][i];
			else if (j > i)
				U_inv->data[j][i] = 0;
			else
			{
				double s = 0;
				for (int k = j + 1; k <= i; k++)
				{
					s += U->data[j][k] * U_inv->data[k][i];
				}
				U_inv->data[j][i] = -1 / U->data[j][j] * s;
			}
		}
	}
	result = (*U_inv) * (*L_inv);//运用了矩阵乘法的重载
	delete L;
	delete U;
	delete L_inv;
	delete U_inv;
	return result;
}

Matrix* Matrix::cholesky()
{
	Matrix* L = new Matrix(this->row, this->column);
	for (int i = 0; i < L->row; i++)
	{
		for (int k = 0; k <= i; k++)
		{
			double sum = 0;
			for (int j = 0; j < k; j++)
			{
				sum += L->data[i][j] * L->data[k][j];
			}
			L->data[i][k] = (i != k) ? (this->data[i][k] - sum) / L->data[k][k] : sqrt(this->data[i][i] - sum);
		}
	}
	return L;
}

Matrix** Matrix::QR()
{
	Matrix** Q_R = new Matrix * [2];
	Q_R[0] = NULL;
	Q_R[1] = NULL;
	if (this->row - 1 > 0)
	{
		Matrix** q = new Matrix * [this->row]();
		if (q == NULL)
		{
			cout << "Error:out of memory" << endl;
			exit(1);
		}
		Matrix* z = this;
		Matrix* z1;
		for (int k = 0; k < this->column && k < this->row - 1; k++)
		{
			Matrix* e = new Matrix(1, this->row);
			Matrix* x = new Matrix(1, this->row);
			double a;
			z1 = new Matrix(z->row, z->column);
			for (int i = 0; i < k; i++)
			{
				z1->data[i][i] = 1;
			}
			for (int i = k; i < z->row; i++)
			{
				for (int j = k; j < z->column; j++)
				{
					z1->data[i][j] = z->data[i][j];
				}
			}
			if (z != this)
			{
				delete z;
			}
			z = z1;
			for (int i = 0; i < z->row; i++)
			{
				x->data[0][i] = z->data[i][k];
			}
			a = x->norm(2);//求二范数
			if (this->data[k][k] > 0)
			{
				a = -a;
			}
			for (int i = 0; i < this->row; i++)
			{
				e->data[0][i] = (i == k) ? 1 : 0;
			}
			for (int i = 0; i < this->row; i++)
			{
				e->data[0][i] = x->data[0][i] + a * e->data[0][i];
			}
			double d = e->norm(2);
			for (int i = 0; i < this->row; i++)
			{
				e->data[0][i] = e->data[0][i] / d;
			}
			q[k] = new Matrix(this->row, this->row);
			if (q[k] == NULL)
			{
				cout << "Error:out of memory" << endl;
				exit(1);
			}
			for (int i = 0; i < this->row; i++)
			{
				for (int j = 0; j < this->row; j++)
				{
					q[k]->data[i][j] = -2 * e->data[0][i] * e->data[0][j];
				}
			}
			for (int i = 0; i < this->row; i++)
			{
				q[k]->data[i][i] += 1;
			}
			z1 = (*q[k]) * (*z);
			if (z != this)
			{
				delete z;
			}
			z = z1;
			delete e;
			delete x;
		}
		delete z;
		Q_R[0] = q[0];
		Q_R[1] = (*q[0]) * (*this);
		for (int i = 1; i < this->row - 1 && i < this->column; i++)
		{
			z1 = (**(q + i)) * (*Q_R[0]);
			if (i > 1)
			{
				delete Q_R[0];
			}
			Q_R[0] = z1;
			delete* (q + i);
		}
		delete q[0];
		z = (*Q_R[0]) * (*this);
		delete Q_R[1];
		Q_R[1] = z;
		Matrix* Q_T = Q_R[0]->T();
		delete Q_R[0];
		Q_R[0] = Q_T;
		delete[] q;
	}
	return Q_R;
}

Matrix* Matrix::eigenvalue()
{
	Matrix** M_array_Q_R = NULL; // 保存Q/R矩阵地址
	enum { q = 0, r = 1 };
	double eps = 1e-5, delta = 1; // 设置计算误差
	int i, dim = this->row, epoch = 0;
	Matrix* Ak0, * Ak, * Qk, * Rk, * M_eigen_val;
	Ak = new Matrix(*this);
	while ((delta > eps) && (epoch < (int)1e+5))
	{
		M_array_Q_R = Ak->QR();
		Qk = M_array_Q_R[q];
		Rk = M_array_Q_R[r];
		Ak0 = Ak;
		Ak = (*Rk) * (*Qk);
		delta = 0;
		for (i = 0; i < dim; i++)
		{
			delta += fabs(Ak->data[i][i] - Ak0->data[i][i]);
		}
		delete Ak0;
		delete Qk;
		delete Rk;
		delete[] M_array_Q_R;
		epoch++;
	}
	if (epoch >= (int)1e+5)
	{
		cout << "\n>>ATTENTION: QR Decomposition end with delta = " << delta << "!(epoch = " << (int)1e5 << "eps = " << eps << ")" << endl;
	}
	M_eigen_val = new Matrix(1, dim);
	for (i = 0; i < dim; i++)
	{
		M_eigen_val->data[0][i] = Ak->data[i][i];
	}
	delete Ak;
	return M_eigen_val;
}

double Matrix::norm(int setting)
{
	double** data = this->data;
	int row = this->row;
	int column = this->column;
	double Val_norm = 0;
	if (row == 1 || column == 1)
	{//向量的范数
		switch (setting)
		{
		case 1:
		{//1范数
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					Val_norm += fabs(data[i][j]);
				}
			}
			break;
		}
		case 2:
		{//2范数
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					Val_norm += data[i][j] * data[i][j];
				}
			}
			Val_norm = pow(Val_norm, 0.5);
			break;
		}
		case INT_MAX:
		{//无穷范数
			Matrix* M_temp_0, * M_temp_1;
			M_temp_0 = this->abs();
			M_temp_1 = (this->column > this->row ? M_temp_0->max_position(1) : M_temp_0->max_position(2));//行向量或者列向量
			int temp_num = (int)M_temp_1->data[0][0];
			if (row > column)
				Val_norm = M_temp_0->data[temp_num][0];//列向量
			else
				Val_norm = M_temp_0->data[0][temp_num];//行向量
			// 释放内存
			delete M_temp_0;
			delete M_temp_1;
			break;
		}
		default:
		{//p范数
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					Val_norm += pow(data[i][j], setting);
				}
			}
			if (Val_norm < 0)
			{
				cout << "Error:For the p norm of a vector, the result cannot be a complex number" << endl;
			}
			Val_norm = pow(Val_norm, 1.0 / setting);
			break;
		}
		}
	}
	else
	{//矩阵范数
		switch (setting)
		{
		case 1:
		{//矩阵的1范数
			Matrix* M_temp_0, * M_temp_1, * M_temp_2;
			M_temp_0 = this->abs();
			M_temp_1 = M_temp_0->sum(2);//按列求和
			M_temp_2 = M_temp_1->max_value(1);
			Val_norm = M_temp_2->data[0][0];
			delete M_temp_0;
			delete M_temp_1;
			delete M_temp_2;
			break;
		}
		case 2:
		{//矩阵的2范数
			Matrix* M_temp_0, * M_temp_1;
			M_temp_0 = this->T();
			M_temp_1 = (*M_temp_0) * (*this);
			Val_norm = M_temp_1->eigen_max();
			Val_norm = pow(Val_norm, 0.5);
			delete M_temp_0;
			delete M_temp_1;
			break;
		}
		case INT_MAX:
		{//矩阵的无穷范数
			Matrix* M_temp_0, * M_temp_1, * M_temp_2;
			M_temp_0 = this->abs();
			M_temp_1 = M_temp_0->sum(1);//按行求和
			M_temp_2 = M_temp_1->max_value(2);
			Val_norm = M_temp_2->data[0][0];
			delete M_temp_0;
			delete M_temp_1;
			delete M_temp_2;
			break;
		}
		case INT_MIN:
		{//矩阵的F范数（Frobenius范数）
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < column; j++)
				{
					Val_norm += data[i][j] * data[i][j];
				}
			}
			Val_norm = pow(Val_norm, 0.5);
			break;
		}
		default:
		{
			cout << "Error:Wrong norm type setting" << endl;
			exit(1);
		}
		}
	}
	return Val_norm;
}

double Matrix::eigen_max()
{
	if (this->column == this->row)
	{
		Matrix* mat_z = new Matrix(this->column, 1, 1.0);
		Matrix* mat_temp_1 = NULL;
		Matrix* mat_temp_2 = NULL;
		Matrix* mat_y = NULL;
		Matrix* mat_z_gap = NULL;
		double m_value = 0, mat_z_gap_norm = 1;
		double deta = 1e-7; //精度设置
		while (mat_z_gap_norm > deta)
		{
			mat_y = (*this) * (*mat_z);//列向量
			mat_temp_1 = mat_y->max_value(2);//需要释放结果空间
			m_value = mat_temp_1->data[0][0];
			mat_temp_2 = mat_z;//需要释放结果空间
			mat_z = (*mat_y) * (1 / m_value);
			mat_z_gap = (*mat_z) - (*mat_temp_2);//需要释放结果空间
			mat_z_gap_norm = mat_z_gap->norm(2);
			delete mat_y;
			delete mat_temp_1;
			delete mat_temp_2;
			delete mat_z_gap;
		}
		delete mat_z;
		return m_value;
	}
	else
	{
		cout << "Error:the matrix must be square" << endl;
		exit(1);
	}
}

double Matrix::cond(int setting)
{
	double matrix_cond = 0;
	if (this->column == this->row)
	{
		if (setting == 1 || setting == 2 || setting == INT_MAX || setting == INT_MIN)
		{
			Matrix* mat_inv = this->inverse();
			matrix_cond = this->norm(setting) * mat_inv->norm(setting);
			delete mat_inv;
		}
		else
		{
			cout << "Error:the type should be set" << endl;
			exit(1);
		}
	}
	else
	{
		cout << "Error:the matrix should be square" << endl;
		exit(1);
	}
	return matrix_cond;
}

double Matrix::cond2_nsquare()
{
	Matrix* mat_T = this->T();
	Matrix* ATA = (*mat_T) * (*this);
	Matrix* eigen = ATA->eigenvalue();//返回行向量
	Matrix* eigen_max = eigen->max_value(1);//返回单个值
	Matrix* eigen_min = eigen->min_value(1);//返回单个值
	double result = sqrt(eigen_max->data[0][0] / eigen_min->data[0][0]);//最大特征值比最小特征值再开根号就是条件数
	delete mat_T;
	delete ATA;
	delete eigen;
	delete eigen_max;
	delete eigen_min;
	return result;
}

Matrix* Matrix::merge(Matrix* mat2, int setting)
{
	if (setting == 1)
	{//纵向合并
		if (this->column != mat2->column)
		{
			cout << "Error:The columns are different" << endl;
			exit(1);
		}
		else
		{
			int row = this->row + mat2->row;
			Matrix* result = new Matrix(row, this->column);
			for (int j = 0; j < this->column; j++)
			{
				for (int i = 0; i < row; i++)
				{
					if (i < this->row)
					{
						result->data[i][j] = this->data[i][j];
					}
					else
					{
						result->data[i][j] = mat2->data[i - this->row][j];
					}
				}
			}
			return result;
		}
	}
	else if (setting == 2)
	{
		if (this->row != mat2->row)
		{
			cout << "Error:The rows are different" << endl;
			exit(1);
		}
		int col = this->column + mat2->column;
		Matrix* result = new Matrix(this->row, col);
		for (int i = 0; i < result->row; i++)
		{
			for (int j = 0; j < result->column; j++)
			{
				if (j < this->column)
				{
					result->data[i][j] = this->data[i][j];
				}
				else
				{
					result->data[i][j] = mat2->data[i][j - this->column];
				}
			}
		}
		return result;
	}
	else
	{
		cout << "Error:Wrong setting" << endl;
		exit(1);
	}

}

Matrix* Matrix::mldivide(Matrix* mat2)
{
	Matrix* A = new Matrix(this);
	Matrix* A_T = A->T();
	Matrix* ATA = (*A_T) * (*A);
	Matrix* ATB = (*A_T) * (*mat2);
	Matrix* ATA_inv = ATA->inverse();
	Matrix* result = (*ATA_inv) * (*ATB);
	delete A;
	delete A_T;
	delete ATA;
	delete ATB;
	delete ATA_inv;
	return result;
}



