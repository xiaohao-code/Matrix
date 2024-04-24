/*用于测试矩阵类的使用*/

#include <iostream>
#include <vector>
#include "matrix.h"

using namespace std;

int main() {
    Matrix a;
    Matrix a1(2, 3);
    Matrix a2(4, 5, 3);
    cout << a << a1 << a2;
    vector<vector<double>> vv(4, vector<double>(3, 4));
    Matrix a3(vv);
    cout << a3;
    Matrix a4(3, 3, 5);
    Matrix a5 = a4;
    Matrix a6 = a5 + a4;
    Matrix a7 = a5 - a4;
    Matrix a8 = a5 * a4;
    Matrix a9 = a5 * 10;
    cout << a6 << a7 << a8 << a9;
}