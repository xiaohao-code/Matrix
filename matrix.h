#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

template <typename T>
class Matrix {
public:
    template <typename U>
	friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);

	Matrix() = default;

	Matrix(int m, int n); // 构造零矩阵

	Matrix(int m, int n, T value); //构造所有元素都是一个值的矩阵

	explicit Matrix(std::vector<std::vector<T>> matrix);

	Matrix(const Matrix<T> &) = default;

	Matrix(Matrix<T> &&) = default;

	~Matrix() = default;

	Matrix<T> operator+(const Matrix<T> &matrix) const;

	Matrix<T> operator-() const;

	Matrix<T> operator-(const Matrix<T> &matrix) const;

	Matrix<T> operator*(const Matrix<T> &matrix) const;

	Matrix<T> operator*(double num) const;//矩阵数乘

	// void print();//显示矩阵
	// void save(ofstream& outfile);//保存矩阵至文件中
	
	Matrix<T> transpose() const;//矩阵转置

	Matrix<T> abs() const;//矩阵所有元素求绝对值

	Matrix<T> sum(int setting) const;//矩阵按行(setting=1)或列(setting=2)求和

	Matrix<T> mean(int setting) const;//矩阵按行列求均值

	Matrix<T> std(int setting) const;//矩阵按行列求标准差

	Matrix<T> swap(int line1, int line2, int setting) const;//矩阵按行列交换位置

	Matrix<int> min_position(int setting) const;//矩阵按行列找到元素最小值位置

	Matrix<int> max_position(int setting) const;//矩阵按行列找到元素最大值位置

	Matrix<T> min_value(int setting) const; //矩阵按行列找到元素最小值

	Matrix<T> max_value(int setting) const;//矩阵按行列找到元素最大值

	Matrix<T> cut(int row_head, int row_tail, int column_head, int column_tail) const;//切取部分矩阵,-1代表末尾

	Matrix<T> inverse() const;//矩阵求逆(LU分解法）

	Matrix<T> cholesky();//矩阵cholesky分解

	// std::pair<Matrix<T>, Matrix<T>> Matrix<T>::QR(); // 矩阵QR分解
	// Matrix<T> eigenvalue();//矩阵特征值
	// double norm(int setting);//矩阵范数(setting=1/2/INT_MAX/INT_MIN:1范数/2范数/无穷范数/F范数，其他值则为向量p范数)
	// double eigen_max();//矩阵最大特征值（幂法）
	// double cond(int setting);//矩阵条件数
	// double cond2_nsquare();//非方阵二范数条件数
	// Matrix<T> merge(Matrix<T> mat2, int setting);//在纵向和横向合并矩阵
	// Matrix<T> mldivide(Matrix<T> mat2);//采用最小二乘法左除


private:
	int row;
	int column;
	std::vector<std::vector<T>> data;
};

template <typename U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix) {
	for (const auto &row : matrix.data) {
        for (const U &element : row) {
            os << element << " ";
        }
        os << std::endl;
        }
    return os;
}


template <typename T>
Matrix<T>::Matrix(int m, int n) 
	: row(m),
	  column(n),
	  data(m, std::vector<T>(n, 0)) {}

template <typename T>
Matrix<T>::Matrix(int m, int n, T value)
	: row(m),
	  column(n),
	  data(m, std::vector<T>(n, value)) {}

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> matrix)
	: row(matrix.size()),
	  column(matrix[0].size()),
	  data(std::move(matrix)) {}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &matrix) const {
	if (this->row == matrix.row && this->column == matrix.column) {
		Matrix<T> result(this->row, this->column);
		for (int i = 0; i < result.row; i++) {
			for (int j = 0; j < result.column; j++) {
				result.data[i][j] = this->data[i][j] + matrix.data[i][j];
			}
		}
		return result;
	} else {
		std::cout << "Error:The row(column) of the matiexs are different" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::operator-() const {
	Matrix<T> result(*this);
	for (int i = 0; i < result.row; i++) {
		for (int j = 0; j < result.column; j++) {
				result.data[i][j] = -result.data[i][j];
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &matrix) const {
	if (this->row == matrix.row && this->column == matrix.column) {
		return *this + (-matrix);
	} else {
		std::cout << "Error:The row(column) of the matiexs are different" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &matrix) const {
	if (this->column == matrix.row) {
		Matrix<T> result(this->row, matrix.column);
		for (int i = 0; i < result.row; i++) {
			for (int j = 0; j < result.column; j++) {
				for (int k = 0; k < this->column; k++) {
					result.data[i][k] += this->data[i][k] * matrix.data[k][j];
				}
			}
		}
		return result;
	} else {
		std::cout << "Erroe:Matrix cannot be multiplied" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::operator*(double num) const {
	Matrix<T> result(this->row, this->column);
	for (int i = 0; i < result.row; i++) {
		for (int j = 0; j < result.column; j++) {
			result.data[i][j] = this->data[i][j] * num;
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
	Matrix<T> result(this->column, this->row);
	for (int i = 0; i < result.row; i++) {
		for (int j = 0; j < result.column; j++) {
			result.data[i][j] = this->data[j][i];
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::abs() const {
	Matrix<T> result(this->row, this->column);
	for (int i = 0; i < result.row; i++) {
		for (int j = 0; j < result.column; j++) {
			result.data[i][j] = fabs(this->data[i][j]);
		}
	}
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::sum(int setting) const {
	if (setting == 1) {
		Matrix<T> result(this->row, 1);
		for (int i = 0; i < this->row; i++){
			for (int j = 0; j < this->column; j++){
				result.data[i][0] += this->data[i][j];
			}
		}
		return result;
	} else if (setting == 2) {
		Matrix<T> result(1, this->column);
		for (int j = 0; j < this->column; j++) {
			for (int i = 0; i < this->row; i++) {
				result.data[0][j] += this->data[i][j];
			}
		}
		return result;
	} else {
		std::cout << "Error:Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::mean(int setting) const {
	if (setting == 1) {
		Matrix<T> mean(this->row, 1);
		auto sum = this->sum(setting);
		for (int i = 0; i < this->row; i++) {
			mean.data[i][0] = sum.data[i][0] / this->column;
		}
		return mean;
	} else if (setting == 2) {
		Matrix<T> mean(1, this->column);
		auto sum = this->sum(setting);
		for (int j = 0; j < this->column; j++) {
			mean.data[0][j] = sum.data[0][j] / this->row;
		}
		return mean;
	} else {
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::std(int setting) const {
	if (setting == 1) {
		auto mean = this->mean(setting);
		Matrix<T> result(this->row, 1);
		for (int i = 0; i < this->row; i++){
			double sum = 0;
			for (int j = 0; j < this->column; j++) {
				double diff = this->data[i][j] - mean.data[i][0];
				sum += diff * diff;
			}
			result.data[i][0] = sqrt(sum / (this->column - 1.0));
		}
		return result;
	} else if (setting == 2) {
		auto mean = this->mean(setting);
		Matrix<T> result(1, this->column);
		for (int j = 0; j < this->column; j++) {
			double sum = 0;
			for (int i = 0; i < this->row; i++) {
				double diff = this->data[i][j] - mean.data[0][j];
				sum += diff * diff;
			}
			result.data[0][j] = sqrt(sum / (this->row - 1.0));
		}
		return result;
	} else{
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::swap(int line1, int line2, int setting) const {
	--line1;
	--line2;
	if (setting == 1) {
		if (line1 < this->row && line2 < this->row) {
			Matrix<T> result(*this);
			for (int j = 0; j < result.column; j++) {
				result.data[line1][j] = this->data[line2][j];
				result.data[line2][j] = this->data[line1][j];
			}
			return result;
		} else{
			std::cout << "Error:The row exchanged exceed the limit" << std::endl;
			exit(1);
		}
	} else if (setting == 2) {
		if (line1 < this->column && line2 < this->column){
			Matrix<T> result(*this);
			for (int i = 0; i < result.row; i++){
				result.data[i][line1] = this->data[i][line2];
				result.data[i][line2] = this->data[i][line1];
			}
			return result;
		} else {
			std::cout << "Error:The column exchanged exceed the limit" << std::endl;
			exit(1);
		}
	} else {
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<int> Matrix<T>::min_position(int setting)  const{
	if (setting == 1) {
		Matrix<int> result(this->row, 1);
		for (int i = 0; i < this->row; i++) {
			auto min = std::numeric_limits<T>::min();
			for (int j = 0; j < this->column; j++) {
				if (this->data[i][j] < min) {
					min = this->data[i][j];
					result.data[i][0] = j;
				}
			}
		}
		return result;
	} else if (setting == 2) {
		Matrix<int> result(1, this->column);
		for (int j = 0; j < this->column; j++) {
			auto min = std::numeric_limits<T>::min();
			for (int i = 0; i < this->row; i++) {
				if (this->data[i][j] < min) {
					min = this->data[i][j];
					result.data[0][j] = i;
				}
			}
		}
		return result;
	} else {
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<int> Matrix<T>::max_position(int setting) const {
	return (-(*this)).min_position(setting);
}

template <typename T>
Matrix<T> Matrix<T>::min_value(int setting) const {
	if (setting == 1) {
		auto min_p = this->min_position(setting);
		Matrix<T> result(this->row, 1);
		for (int i = 0; i < this->row; i++) {
			auto index = min_p.data[i][0];
			result.data[i][0] = this->data[i][index];
		}
		return result;
	} else if (setting == 2) {
		Matrix<T> min_p = this->min_position(setting);
		Matrix<T> result(1, this->column);
		for (int j = 0; j < this->column; j++) {
			auto index = min_p.data[0][j];
			result.data[0][j] = this->data[index][j];
		}
		return result;
	} else {
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<T> Matrix<T>::max_value(int setting) const {
	return -((-(*this)).min_value(setting));
}

template <typename T>
Matrix<T> Matrix<T>::cut(int row_head, int row_tail, int column_head, int column_tail) const {
	if (row_tail < 0) {
		if (row_tail == -1) {
			row_tail = this->row;
		} else {
			std::cout << "Error:row_tail exceed the limit" << std::endl;
			exit(1);
		}
	}
	if (row_head < 0) {
		if (row_head == -1) {
			row_head = this->row;
		} else {
			std::cout << "Error:row_head exceed the limit" << std::endl;
			exit(1);
		}
	}
	if (column_tail < 0) {
		if (column_tail == -1) {
			column_tail = this->column;
		} else {
			std::cout << "Error:column_tail exceed the limit" << std::endl;
			exit(1);
		}
	}
	if (column_head < 0) {
		if (column_head == -1) {
			column_head = this->column;
		} else {
			std::cout << "Error:column_head exceed the limit" << std::endl;
			exit(1);
		}
	}
	if (row_tail > this->row || column_tail > this->column) {
		std::cout << "Error:Exceed the limits" << std::endl;
		exit(1);
	} else {
		if (row_head > row_tail || column_head > column_tail) {
			std::cout << "Error:Wrong Parameters" << std::endl;
			exit(1);
		} else {
			row_head = row_head - 1;
			column_head = column_head - 1;
			Matrix<T> result(row_tail - row_head, column_tail - column_head);
			for (int i = 0; i < result.row; i++) {
				for (int j = 0; j < result.column; j++) {
					result.data[i][j] = this->data[i + row_head][j + column_head];
				}
			}
			return result;
		}
	}
}

template <typename T>
Matrix<T> Matrix<T>::inverse() const {
	if (this->column != this->row) {
		std::cout << "Error:Matrix must be square" << std::endl;
		exit(1);
	}
	Matrix<T> L(this->row, this->column);
	Matrix<T> U(this->row, this->column);
	Matrix<T> L_inv(this->row, this->column);
	Matrix<T> U_inv(this->row, this->column);
	for (int i = 0; i < this->row; i++) {
		L.data[i][i] = 1;
	}
	for (int j = 0; j < this->column; j++) {
		U.data[0][j] = this->data[0][j];
	}
	for (int i = 1; i < this->row; i++)
	{
		L.data[i][0] = this->data[i][0] / U.data[0][0];
	}
	for (int i = 1; i < this->row; i++) {
		for (int j = i; j < this->column; j++) {
			double s = 0;
			for (int k = 0; k < i; k++) {
				s += L.data[i][k] * U.data[k][j];
			}
			U.data[i][j] = this->data[i][j] - s;
		}
		for (int d = i; d < this->row; d++) {
			double s = 0;
			for (int k = 0; k < i; k++) {
				s += L.data[d][k] * U.data[k][i];
			}
			L.data[d][i] = (this->data[d][i] - s) / U.data[i][i];
		}
	}
	for (int j = 0; j < this->column; j++) {
		for (int i = j; i < this->row; i++) {
			if (i == j)
				L_inv.data[i][j] = 1 / L.data[i][j];
			else if (i < j)
				L_inv.data[i][j] = 0;
			else {
				double s = 0;
				for (int k = j; k < i; k++) {
					s += L.data[i][k] * L_inv.data[k][j];
				}
				L_inv.data[i][j] = -L_inv.data[j][j] * s;
			}
		}
	}
	for (int i = 0; i < this->row; i++) {
		for (int j = i; j >= 0; j--) {
			if (i == j)
				U_inv.data[j][i] = 1 / U.data[j][i];
			else if (j > i)
				U_inv.data[j][i] = 0;
			else {
				double s = 0;
				for (int k = j + 1; k <= i; k++) {
					s += U.data[j][k] * U_inv.data[k][i];
				}
				U_inv.data[j][i] = -1 / U.data[j][j] * s;
			}
		}
	}
	auto result = (U_inv) * (L_inv);
	return result;
}

template <typename T>
Matrix<T> Matrix<T>::cholesky() {
	Matrix<T> L(this->row, this->column);
	for (int i = 0; i < L.row; i++) {
		for (int k = 0; k <= i; k++) {
			double sum = 0;
			for (int j = 0; j < k; j++) {
				sum += L.data[i][j] * L.data[k][j];
			}
			L.data[i][k] = (i != k) ? (this->data[i][k] - sum) / L.data[k][k] : sqrt(this->data[i][i] - sum);
		}
	}
	return L;
}

// template <typename T>
// std::pair<Matrix<T>, Matrix<T>> Matrix<T>::QR() {
// 	std::pair<Matrix<T>, Matrix<T>> Q_R;
// 	if (this->row - 1 > 0) {
// 		Matrix<T>* q = new Matrix * [this->row]();
// 		if (q == NULL)
// 		{
// 			std::cout << "Error:out of memory" << std::endl;
// 			exit(1);
// 		}
// 		Matrix<T> z = this;
// 		Matrix<T> z1;
// 		for (int k = 0; k < this->column && k < this->row - 1; k++)
// 		{
// 			Matrix<T> e = new Matrix(1, this->row);
// 			Matrix<T> x = new Matrix(1, this->row);
// 			double a;
// 			z1 = new Matrix(z->row, z->column);
// 			for (int i = 0; i < k; i++)
// 			{
// 				z1->data[i][i] = 1;
// 			}
// 			for (int i = k; i < z->row; i++)
// 			{
// 				for (int j = k; j < z->column; j++)
// 				{
// 					z1->data[i][j] = z->data[i][j];
// 				}
// 			}
// 			if (z != this)
// 			{
// 				delete z;
// 			}
// 			z = z1;
// 			for (int i = 0; i < z->row; i++)
// 			{
// 				x->data[0][i] = z->data[i][k];
// 			}
// 			a = x->norm(2);//�������
// 			if (this->data[k][k] > 0)
// 			{
// 				a = -a;
// 			}
// 			for (int i = 0; i < this->row; i++)
// 			{
// 				e->data[0][i] = (i == k) ? 1 : 0;
// 			}
// 			for (int i = 0; i < this->row; i++)
// 			{
// 				e->data[0][i] = x->data[0][i] + a * e->data[0][i];
// 			}
// 			double d = e->norm(2);
// 			for (int i = 0; i < this->row; i++)
// 			{
// 				e->data[0][i] = e->data[0][i] / d;
// 			}
// 			q[k] = new Matrix(this->row, this->row);
// 			if (q[k] == NULL)
// 			{
// 				std::cout << "Error:out of memory" << std::endl;
// 				exit(1);
// 			}
// 			for (int i = 0; i < this->row; i++)
// 			{
// 				for (int j = 0; j < this->row; j++)
// 				{
// 					q[k]->data[i][j] = -2 * e->data[0][i] * e->data[0][j];
// 				}
// 			}
// 			for (int i = 0; i < this->row; i++)
// 			{
// 				q[k]->data[i][i] += 1;
// 			}
// 			z1 = (*q[k]) * (*z);
// 			if (z != this)
// 			{
// 				delete z;
// 			}
// 			z = z1;
// 			delete e;
// 			delete x;
// 		}
// 		delete z;
// 		Q_R[0] = q[0];
// 		Q_R[1] = (*q[0]) * (*this);
// 		for (int i = 1; i < this->row - 1 && i < this->column; i++)
// 		{
// 			z1 = (**(q + i)) * (*Q_R[0]);
// 			if (i > 1)
// 			{
// 				delete Q_R[0];
// 			}
// 			Q_R[0] = z1;
// 			delete* (q + i);
// 		}
// 		delete q[0];
// 		z = (*Q_R[0]) * (*this);
// 		delete Q_R[1];
// 		Q_R[1] = z;
// 		Matrix<T> Q_T = Q_R[0]->T();
// 		delete Q_R[0];
// 		Q_R[0] = Q_T;
// 		delete[] q;
// 	}
// 	return Q_R;
// }

// template <typename T>
// Matrix<T> Matrix<T>::eigenvalue() {
// 	Matrix<T>* M_array_Q_R = NULL; 
// 	enum { q = 0, r = 1 };
// 	double eps = 1e-5, delta = 1; // ���ü������
// 	int i, dim = this->row, epoch = 0;
// 	Matrix<T> Ak0, * Ak, * Qk, * Rk, * M_eigen_val;
// 	Ak = new Matrix(*this);
// 	while ((delta > eps) && (epoch < (int)1e+5))
// 	{
// 		M_array_Q_R = Ak->QR();
// 		Qk = M_array_Q_R[q];
// 		Rk = M_array_Q_R[r];
// 		Ak0 = Ak;
// 		Ak = (*Rk) * (*Qk);
// 		delta = 0;
// 		for (i = 0; i < dim; i++)
// 		{
// 			delta += fabs(Ak->data[i][i] - Ak0->data[i][i]);
// 		}
// 		delete Ak0;
// 		delete Qk;
// 		delete Rk;
// 		delete[] M_array_Q_R;
// 		epoch++;
// 	}
// 	if (epoch >= (int)1e+5)
// 	{
// 		std::cout << "\n>>ATTENTION: QR Decomposition end with delta = " << delta << "!(epoch = " << (int)1e5 << "eps = " << eps << ")" << std::endl;
// 	}
// 	M_eigen_val = new Matrix(1, dim);
// 	for (i = 0; i < dim; i++)
// 	{
// 		M_eigen_val->data[0][i] = Ak->data[i][i];
// 	}
// 	delete Ak;
// 	return M_eigen_val;
// }

// template <typename T>
// double Matrix<T>::norm(int setting) {
// 	double** data = this->data;
// 	int row = this->row;
// 	int column = this->column;
// 	double Val_norm = 0;
// 	if (row == 1 || column == 1)
// 	{
// 		switch (setting)
// 		{
// 		case 1:
// 		{//1����
// 			for (int i = 0; i < row; i++)
// 			{
// 				for (int j = 0; j < column; j++)
// 				{
// 					Val_norm += fabs(data[i][j]);
// 				}
// 			}
// 			break;
// 		}
// 		case 2:
// 		{//2����
// 			for (int i = 0; i < row; i++)
// 			{
// 				for (int j = 0; j < column; j++)
// 				{
// 					Val_norm += data[i][j] * data[i][j];
// 				}
// 			}
// 			Val_norm = pow(Val_norm, 0.5);
// 			break;
// 		}
// 		case INT_MAX:
// 		{//�����
// 			Matrix<T> M_temp_0, * M_temp_1;
// 			M_temp_0 = this->abs();
// 			M_temp_1 = (this->column > this->row ? M_temp_0->max_position(1) : M_temp_0->max_position(2));//����������������
// 			int temp_num = (int)M_temp_1->data[0][0];
// 			if (row > column)
// 				Val_norm = M_temp_0->data[temp_num][0];//������
// 			else
// 				Val_norm = M_temp_0->data[0][temp_num];//������
// 			// �ͷ��ڴ�
// 			delete M_temp_0;
// 			delete M_temp_1;
// 			break;
// 		}
// 		default:
// 		{//p����
// 			for (int i = 0; i < row; i++)
// 			{
// 				for (int j = 0; j < column; j++)
// 				{
// 					Val_norm += pow(data[i][j], setting);
// 				}
// 			}
// 			if (Val_norm < 0)
// 			{
// 				std::cout << "Error:For the p norm of a vector, the result cannot be a complex number" << std::endl;
// 			}
// 			Val_norm = pow(Val_norm, 1.0 / setting);
// 			break;
// 		}
// 		}
// 	}
// 	else
// 	{//������
// 		switch (setting)
// 		{
// 		case 1:
// 		{//�����1����
// 			Matrix<T> M_temp_0, * M_temp_1, * M_temp_2;
// 			M_temp_0 = this->abs();
// 			M_temp_1 = M_temp_0->sum(2);//�������
// 			M_temp_2 = M_temp_1->max_value(1);
// 			Val_norm = M_temp_2->data[0][0];
// 			delete M_temp_0;
// 			delete M_temp_1;
// 			delete M_temp_2;
// 			break;
// 		}
// 		case 2:
// 		{//�����2����
// 			Matrix<T> M_temp_0, * M_temp_1;
// 			M_temp_0 = this->T();
// 			M_temp_1 = (*M_temp_0) * (*this);
// 			Val_norm = M_temp_1->eigen_max();
// 			Val_norm = pow(Val_norm, 0.5);
// 			delete M_temp_0;
// 			delete M_temp_1;
// 			break;
// 		}
// 		case INT_MAX:
// 		{//����������
// 			Matrix<T> M_temp_0, * M_temp_1, * M_temp_2;
// 			M_temp_0 = this->abs();
// 			M_temp_1 = M_temp_0->sum(1);//�������
// 			M_temp_2 = M_temp_1->max_value(2);
// 			Val_norm = M_temp_2->data[0][0];
// 			delete M_temp_0;
// 			delete M_temp_1;
// 			delete M_temp_2;
// 			break;
// 		}
// 		case INT_MIN:
// 		{//�����F������Frobenius������
// 			for (int i = 0; i < row; i++)
// 			{
// 				for (int j = 0; j < column; j++)
// 				{
// 					Val_norm += data[i][j] * data[i][j];
// 				}
// 			}
// 			Val_norm = pow(Val_norm, 0.5);
// 			break;
// 		}
// 		default:
// 		{
// 			std::cout << "Error:Wrong norm type setting" << std::endl;
// 			exit(1);
// 		}
// 		}
// 	}
// 	return Val_norm;
// }

// template <typename T>
// double Matrix<T>::eigen_max()
// {
// 	if (this->column == this->row)
// 	{
// 		Matrix<T> mat_z = new Matrix(this->column, 1, 1.0);
// 		Matrix<T> mat_temp_1 = NULL;
// 		Matrix<T> mat_temp_2 = NULL;
// 		Matrix<T> mat_y = NULL;
// 		Matrix<T> mat_z_gap = NULL;
// 		double m_value = 0, mat_z_gap_norm = 1;
// 		double deta = 1e-7; //��������
// 		while (mat_z_gap_norm > deta)
// 		{
// 			mat_y = (*this) * (*mat_z);//������
// 			mat_temp_1 = mat_y->max_value(2);//��Ҫ�ͷŽ���ռ�
// 			m_value = mat_temp_1->data[0][0];
// 			mat_temp_2 = mat_z;//��Ҫ�ͷŽ���ռ�
// 			mat_z = (*mat_y) * (1 / m_value);
// 			mat_z_gap = (*mat_z) - (*mat_temp_2);//��Ҫ�ͷŽ���ռ�
// 			mat_z_gap_norm = mat_z_gap->norm(2);
// 			delete mat_y;
// 			delete mat_temp_1;
// 			delete mat_temp_2;
// 			delete mat_z_gap;
// 		}
// 		delete mat_z;
// 		return m_value;
// 	}
// 	else
// 	{
// 		std::cout << "Error:the matrix must be square" << std::endl;
// 		exit(1);
// 	}
// }

// template <typename T>
// double Matrix<T>::cond(int setting)
// {
// 	double matrix_cond = 0;
// 	if (this->column == this->row)
// 	{
// 		if (setting == 1 || setting == 2 || setting == INT_MAX || setting == INT_MIN)
// 		{
// 			Matrix<T> mat_inv = this->inverse();
// 			matrix_cond = this->norm(setting) * mat_inv->norm(setting);
// 			delete mat_inv;
// 		}
// 		else
// 		{
// 			std::cout << "Error:the type should be set" << std::endl;
// 			exit(1);
// 		}
// 	}
// 	else
// 	{
// 		std::cout << "Error:the matrix should be square" << std::endl;
// 		exit(1);
// 	}
// 	return matrix_cond;
// }

// template <typename T>
// double Matrix<T>::cond2_nsquare()
// {
// 	Matrix<T> mat_T = this->T();
// 	Matrix<T> ATA = (*mat_T) * (*this);
// 	Matrix<T> eigen = ATA->eigenvalue();//����������
// 	Matrix<T> eigen_max = eigen->max_value(1);//���ص���ֵ
// 	Matrix<T> eigen_min = eigen->min_value(1);//���ص���ֵ
// 	double result = sqrt(eigen_max->data[0][0] / eigen_min->data[0][0]);//�������ֵ����С����ֵ�ٿ����ž���������
// 	delete mat_T;
// 	delete ATA;
// 	delete eigen;
// 	delete eigen_max;
// 	delete eigen_min;
// 	return result;
// }

// template <typename T>
// Matrix<T> Matrix<T>::merge(Matrix<T> mat2, int setting) {
// 	if (setting == 1) {
// 		if (this->column != mat2.column) {
// 			std::cout << "Error:The columns are different" << std::endl;
// 			exit(1);
// 		} else {
// 			int row = this->row + mat2.row;
// 			Matrix<T> result(row, this->column);
// 			for (int j = 0; j < this->column; j++) {
// 				for (int i = 0; i < row; i++) {
// 					if (i < this->row) {
// 						result.data[i][j] = this->data[i][j];
// 					} else {
// 						result.data[i][j] = mat2.data[i - this->row][j];
// 					}
// 				}
// 			}
// 			return result;
// 		}
// 	} else if (setting == 2) {
// 		if (this->row != mat2.row) {
// 			std::cout << "Error:The rows are different" << std::endl;
// 			exit(1);
// 		}
// 		int col = this->column + mat2.column;
// 		Matrix<T> result(this->row, col);
// 		for (int i = 0; i < result.row; i++) {
// 			for (int j = 0; j < result.column; j++) {
// 				if (j < this->column) {
// 					result.data[i][j] = this->data[i][j];
// 				} else {
// 					result.data[i][j] = mat2.data[i][j - this->column];
// 				}
// 			}
// 		}
// 		return result;
// 	}
// 	else {
// 		std::cout << "Error:Wrong setting" << std::endl;
// 		exit(1);
// 	}
// }

// template <typename T>
// Matrix<T> Matrix<T>::mldivide(Matrix<T> mat2) {
// 	Matrix<T> A(*this);
// 	auto A_T = A.transpose();
// 	auto ATA = (A_T) * (A);
// 	auto ATB = (A_T) * (mat2);
// 	auto ATA_inv = ATA.inverse();
// 	auto result = (ATA_inv) * (ATB);
// 	return result;
// }

