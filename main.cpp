#include <iostream>
#include "ndimmatrix/matrix.h"

using namespace std;

constexpr int increment(int x){return ++x;}

int main(){

    Matrix<double,2> M = {{1,2,3,4},
                          {5,6,7,8}};

    cout << M << endl;

    char answer = 'y';
    int x;
    while (answer== 'y' || answer == 'Y'){
        cout << ">> x = ";
        cin >> x;
        cout << increment(x) << endl;

        cout << "Desea Continuar: (Si = y | No = n): ";
        cin >> answer;
    }


    cout << "Hello World...!!!" << endl;

    return 0;
}
