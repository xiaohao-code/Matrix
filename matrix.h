#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>


template <typename T>
class Matrix {
public:
    std::vector<std::vector<T>> data;

	template <typename U>
	friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix);

	Matrix() = default;

	Matrix(int m, int n); // 构造零矩阵

	Matrix(int m, int n, T value); //构造所有元素都是一个值的矩阵

	explicit Matrix(std::vector<std::vector<T>> matrix);

	Matrix(const Matrix<T> &) = default;

	Matrix(Matrix<T> &&) = default;

	~Matrix() = default;

    Matrix<T> &operator=(const Matrix<T> &) = default;

    Matrix<T> &operator=(Matrix<T> &&) = default;

    Matrix<T> operator+(const Matrix<T> &matrix) const;

	Matrix<T> operator-() const;

	Matrix<T> operator-(const Matrix<T> &matrix) const;

	Matrix<T> operator*(const Matrix<T> &matrix) const;

	Matrix<T> operator*(double num) const; //矩阵数乘
	
	Matrix<T> transpose() const; //矩阵转置

	Matrix<T> abs() const; //矩阵所有元素求绝对值

	Matrix<T> sum(int setting) const; //矩阵按行(setting=1)或列(setting=2)求和

	Matrix<double> mean(int setting) const; //矩阵按行列求均值

	Matrix<double> std(int setting) const; //矩阵按行列求标准差

	Matrix<T> swap(int line1, int line2, int setting) const; //矩阵按行列交换位置

	Matrix<int> min_position(int setting) const; //矩阵按行列找到元素最小值位置

	Matrix<int> max_position(int setting) const; //矩阵按行列找到元素最大值位置

	Matrix<T> min_value(int setting) const; //矩阵按行列找到元素最小值

	Matrix<T> max_value(int setting) const; //矩阵按行列找到元素最大值

	Matrix<T> cut(int row_head, int row_tail, int column_head, int column_tail) const; //切取部分矩阵,-1代表末尾

	std::pair<Matrix<double>, Matrix<double>> lu() const;

	Matrix<double> cholesky() const; //矩阵cholesky分解

	Matrix<double> inverse() const; //矩阵求逆(LU分解法）

	//......

private:
	int row;
	int column;
};

template <typename U>
std::ostream &operator<<(std::ostream &os, const Matrix<U> &matrix) {
	for (const auto &row : matrix.data) {
        for (const auto &element : row) {
            os << element << " ";
        }
        os << std::endl;
    }
    return os;
}

template <typename T>
Matrix<T>::Matrix(int m, int n) 
	: data(m, std::vector<T>(n, 0)),
	  row(m),
	  column(n) {}

template <typename T>
Matrix<T>::Matrix(int m, int n, T value)
	: data(m, std::vector<T>(n, value)),
	  row(m),
	  column(n) {}

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> matrix)
	: data(std::move(matrix)),
	  row(data.size()),
	  column(row > 0 ? data[0].size() : 0) {}

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
    return *this + (-matrix);
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &matrix) const {
	if (this->column == matrix.row) {
		Matrix<T> result(this->row, matrix.column);
		for (int i = 0; i < result.row; i++) {
			for (int j = 0; j < result.column; j++) {
				for (int k = 0; k < this->column; k++) {
					result.data[i][j] += this->data[i][k] * matrix.data[k][j];
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
			result.data[i][j] = std::abs(this->data[i][j]);
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
Matrix<double> Matrix<T>::mean(int setting) const {
	if (setting == 1) {
		Matrix<double> mean(this->row, 1);
		auto sum = this->sum(setting);
        double nums = static_cast<double>(this->column);
        for (int i = 0; i < this->row; i++) {
            mean.data[i][0] = sum.data[i][0] / nums;
        }
        return mean;
	} else if (setting == 2) {
		Matrix<double> mean(1, this->column);
		auto sum = this->sum(setting);
        double nums = static_cast<double>(this->row);
		for (int j = 0; j < this->column; j++) {
			mean.data[0][j] = sum.data[0][j] / nums;
		}
		return mean;
	} else {
		std::cout << "Error,Wrong setting(1/2)" << std::endl;
		exit(1);
	}
}

template <typename T>
Matrix<double> Matrix<T>::std(int setting) const {
	if (setting == 1) {
		auto mean = this->mean(setting);
		Matrix<double> result(this->row, 1);
		for (int i = 0; i < this->row; i++){
			double sum = 0;
			for (int j = 0; j < this->column; j++) {
				double diff = this->data[i][j] - mean.data[i][0];
				sum += diff * diff;
			}
			result.data[i][0] = std::sqrt(sum / (this->column - 1.0));
		}
		return result;
	} else if (setting == 2) {
		auto mean = this->mean(setting);
		Matrix<double> result(1, this->column);
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
			result.data[line1] = this->data[line2];
			result.data[line2] = this->data[line1];
			return result;
		} else {
			std::cout << "Error:The row exchanged exceed the limit" << std::endl;
			exit(1);
		}
	} else if (setting == 2) {
		if (line1 < this->column && line2 < this->column){
			Matrix<T> result(*this);
			for (int i = 0; i < result.row; i++) {
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
			auto min = std::numeric_limits<T>::max();
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
			auto min = std::numeric_limits<T>::max();
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
		auto min_p = this->min_position(setting);
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
std::pair<Matrix<double>, Matrix<double>> Matrix<T>::lu() const {
	Matrix<double> l(this->row, this->column);
	for (int i = 0; i < this->row; i++) {
		l.data[i][i] = 1;
	}
	Matrix<double> u(this->row, this->column);
	for (int k = 0; k < this->row; k++) {
		for (int j = k; j < this->column; j++) {
			double sum = 0;
			for (int r = 0; r <= k - 1; r++) {
				sum += l.data[k][r] * u.data[r][j];
			}
			u.data[k][j] = this->data[k][j] - sum;
		}
		for (int i = k + 1; i < this->row; i++) {
			double sum = 0;
			for (int r = 0; r <= k - 1; r++) {
				sum += l.data[i][r] * u.data[r][k];
			}
			l.data[i][k] = (this->data[i][k] - sum) / u.data[k][k];
		}
	}
	return {l, u};
}

template <typename T>
Matrix<double> Matrix<T>::cholesky() const {
	Matrix<double> L(this->row, this->column);
	for (int j = 0; j < L.column; j++) {
		for (int i = j; i < L.row; i++) {
			double sum = 0;
			for (int k = 0; k <= j - 1; k++) {
				sum += L.data[i][k] * L.data[j][k];
			}
			L.data[i][j] = (i == j) ? std::sqrt(this->data[j][j] - sum) : (this->data[i][j] - sum) / L.data[j][j];
		}
	}
	return L;
}

template <typename T>
Matrix<double> Matrix<T>::inverse() const {
	if (this->column != this->row) {
		std::cout << "Error:Matrix must be square" << std::endl;
		exit(1);
	}
	auto lu_res = this->lu();
	auto l = lu_res.first;
	auto u = lu_res.second;
	Matrix<double> l_inv(this->row, this->column);
	Matrix<double> u_inv(this->row, this->column);
	for (int j = 0; j < this->column; j++) {
		for (int i = j; i < this->row; i++) {
			if (i == j)
				l_inv.data[i][j] = 1 / l.data[i][j];
			else {
				double s = 0;
				for (int k = j; k < i; k++) {
					s += l.data[i][k] * l_inv.data[k][j];
				}
				l_inv.data[i][j] = -l_inv.data[j][j] * s;
			}
		}
	}
	for (int i = 0; i < this->row; i++) {
		for (int j = i; j >= 0; j--) {
			if (i == j)
				u_inv.data[j][i] = 1 / u.data[j][i];
			else {
				double s = 0;
				for (int k = j + 1; k <= i; k++) {
					s += u.data[j][k] * u_inv.data[k][i];
				}
				u_inv.data[j][i] = -1 / u.data[j][j] * s;
			}
		}
	}
	return (u_inv * l_inv);
}


