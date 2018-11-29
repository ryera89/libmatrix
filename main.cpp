#include <iostream>
#include "ndimmatrix/matrix.h"

using namespace std;
using namespace matrix_impl;

constexpr int increment(int x){return ++x;}

int main(){

    Matrix<double,2> M = {{1,2,3,4},
                          {5,6,7,8},
                          {9,10,11,12}};

    M(0,0) = 343;

    cout << M << endl;

    //FIXME: Matrix Slicing must has in range values
    Matrix<double,2> M_slice = M(1,Slice(0,2,2));

    cout << M_slice << endl;

    Matrix<double,1> col2 = M.column(2);

    cout << M.column(3)(2) << endl;

    cout << col2 << endl;

//    char answer = 'y';
//    int x;
//    while (answer== 'y' || answer == 'Y'){
//        cout << ">> x = ";
//        cin >> x;
//        cout << increment(x) << endl;

//        cout << "Desea Continuar: (Si = y | No = n): ";
//        cin >> answer;
//    }


    cout << "Hello World...!!!" << endl;

    return 0;
}
