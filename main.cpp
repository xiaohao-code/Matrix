#include "matrix.h"

using namespace std;

int main() {
    // Matrix<int> a1;
    // Matrix<int> a2(3, 3);
    // Matrix<int> a3(3, 3, 3);
    // Matrix<int> a4({3, vector<int>(3, 4)});
    // cout << a1 << a2 << a3 << a4;
    // auto a5 = a3 + a4;
    // auto a6 = a4 - a3;
    // auto a7 = a3 * a4;
    // auto a8 = -a7;
    // auto a9 = a8 * 3;
    // cout << a5 << a6 << a7 << a8 << a9;
    // Matrix<int> a10({{1, 2, 3}, {4, 5, 6}});
    // cout << a10 << a10.transpose() << a9.abs();
    // cout << a10.sum(1) << a10.sum(2);
    // cout << a10.mean(1) << a10.mean(2);
    // cout << a10.std(1) << a10.std(2);
    // cout << a10.swap(1, 2, 1) << a10.swap(1, 2, 2);
    // cout << a10.min_position(1) << a10.max_position(1) << a10.min_position(2) << a10.max_position(2);
    // cout << a10.min_value(1) << a10.min_value(2) << a10.max_value(1) << a10.max_value(2);
    // cout << a10.cut(1, 2, 1, 2);

    // Matrix<double> cho({{4, -1, 1}, {-1, 4.25, 2.75}, {1, 2.75, 3.5}});
    // cout << cho.cholesky();

    // Matrix<double> alu({{2, -3, -2}, {-1, 2, -2}, {3, -1, 4}});
    // auto lu_sol = alu.lu();
    // cout << lu_sol.first << lu_sol.second;

    Matrix<double> ainv({{3, -2, 5}, {2, 3, 4}, {3, -1, 3}});
    cout << ainv.inverse();
}