#pragma once
#ifndef _MATRIX_H
#define _MATRIX_H
#include <string>
using namespace std;

class Matrix
{
public:
	int row;
	int column;
	double** data;
	Matrix();
	Matrix(int m, int n);//构造零矩阵
	Matrix(int m, int n, double value);//构造所有元素都是一个值的矩阵
	Matrix(int m, int n, double* _data);//从一维数组空间构造矩阵
	Matrix(const Matrix& mat);//拷贝构造函数
	Matrix(Matrix* mat);
	~Matrix();//析构函数
	void print();//显示矩阵
	void save(ofstream& outfile);//保存矩阵至文件中
	Matrix* operator+(const Matrix& mat2);//矩阵相加
	Matrix* operator-(const Matrix& mat2);//矩阵相减
	Matrix* operator*(const Matrix& mat2);//矩阵相乘
	Matrix* operator*(double num);//矩阵数乘
	Matrix* T();//矩阵转置
	Matrix* abs();//矩阵所有元素求绝对值
	Matrix* sum(int setting);//矩阵按行(setting=1)或列(setting=2)求和
	Matrix* mean(int setting);//矩阵按行列求均值
	Matrix* std(int setting);//矩阵按行列求标准差
	Matrix* swap(int line1, int line2, int setting);//矩阵按行列交换位置
	Matrix* min_position(int setting);//矩阵按行列找到元素最小值位置
	Matrix* max_position(int setting);//矩阵按行列找到元素最大值位置
	Matrix* min_value(int setting); //矩阵按行列找到元素最小值
	Matrix* max_value(int setting);//矩阵按行列找到元素最大值
	Matrix* cut(int row_head, int row_tail, int column_head, int column_tail);//切取部分矩阵,-1代表末尾
	Matrix* inverse();//矩阵求逆(LU分解法）
	Matrix* cholesky();//矩阵cholesky分解
	Matrix** QR();//矩阵QR分解
	Matrix* eigenvalue();//矩阵特征值
	double norm(int setting);//矩阵范数(setting=1/2/INT_MAX/INT_MIN:1范数/2范数/无穷范数/F范数，其他值则为向量p范数)
	double eigen_max();//矩阵最大特征值（幂法）
	double cond(int setting);//矩阵条件数
	double cond2_nsquare();//非方阵二范数条件数
	Matrix* merge(Matrix* mat2, int setting);//在纵向和横向合并矩阵
	Matrix* mldivide(Matrix* mat2);//采用最小二乘法左除
};

#endif
