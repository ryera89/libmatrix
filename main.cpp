#include <iostream>
#include "ndimmatrix/matrix.h"
#include "mkl.h"
#include <chrono>

using namespace std;

constexpr int increment(int x){return ++x;}


int main(){

//    Matrix<double,2> M1 = {{1,2,3,4},
//                          {5,6,7,8},
//                          {9,10,11,12}};


    Matrix<double,2> M1 = {{1,2,3,4},
                           {2,5,6,7},
                           {3,6,8,9},
                           {4,7,9,10}};

    Matrix<double,2> M2 = {{0,0,0,0},
                           {3,0,0,3},
                           {3,3,0,0},
                           {3,0,3,0}};

    Matrix<double,2> M3 = M1 + M2 + M1;

    cout << setprecision(3) << M3 << endl;

    int val(3);

    M3*=val;

    cout << setprecision(3) << M3 << endl;

    Matrix<double,2> M4 = M1*M2;

    Matrix<complexd,2> CM(4,4,complexd(1,2));

    Matrix<complexd,2> CM1 = CM+M1;
    Matrix<complexd,2> CM2 = CM-M1;
    Matrix<complexd,2> CM3 = M1+CM;
    Matrix<complexd,2> CM4 = M1-CM;

    cout << setprecision(3) << CM1 << endl;
    cout << setprecision(3) << CM2 << endl;
    cout << setprecision(3) << CM3 << endl;
    cout << setprecision(3) << CM4 << endl;

    cout << 1.0-complexd(1,2) << endl;

    Matrix<int,2> MI1 = {{1,2,3,4},
                            {2,5,6,7},
                            {3,6,8,9},
                            {4,7,9,10}};

    cout << (M1+=MI1) << endl;

    cout << setprecision(3) << M4 << endl;

    Matrix<double,2,MATRIX_TYPE::SYMM> SM1(M1);

    cout << SM1 << endl;
    valarray<uint32_t> indx1 = {0,1,2};
    Matrix<double,2> indM1 = SM1(indx1,indx1);
    if (indM1.matrix_type == MATRIX_TYPE::GEN) cout << "GEN" << endl;

    cout << indM1 << endl;

    cout << setprecision(3) << M1 << endl;

    cout << setprecision(3) << SM1 << endl;

    cout << setprecision(3) << (SM1+=M1) << endl;
    cout << setprecision(3) << (SM1+=SM1) << endl;

    cout << setprecision(3) << SM1 << endl;


    Matrix<std::complex<double>,2,MATRIX_TYPE::HER> HM1(4);
    for (size_t i = 0; i < HM1.rows(); ++i){
        for (size_t j = i; j < HM1.cols(); ++j){
            HM1(i,j) = std::complex<double>(j,i+j);
        }
    }
    cout << setprecision(3) << HM1 << endl;
    cout << (HM1+=HM1) << endl;
    cout << (HM1+=SM1) << endl;

    HM1 += 10;

    cout << setprecision(3) << HM1 << endl;


    HM1*=10;

    cout << setprecision(3) << HM1 << endl;

    Matrix<double,2,MATRIX_TYPE::CSR> SpM1(M2);

    cout << M2 << endl;

    cout << SpM1 << endl;
    SpM1.printData();

    Matrix<double,2,MATRIX_TYPE::CSR3> SpM2(M2);

    cout << SpM2 << endl;
    SpM2.printData();

    vector<uint32_t> iindx = {1,2,3,0};
    vector<uint32_t> jindx = {0,1,2,3};

    Matrix<double,2,MATRIX_TYPE::CSR> SpM11 = SpM1(iindx,jindx);

    Matrix<double,2,MATRIX_TYPE::CSR3> SpM22 = SpM2(iindx,jindx);

    cout << "new data here" << endl;

    SpM11.printData();
    SpM22.printData();

    cout << SpM11 << endl;
    cout << SpM22 << endl;

    Matrix<complexd,2> CM5 = {{complexd(0,0),complexd(0,1),complexd(0,0),complexd(1,2)},
                             {complexd(1,0),complexd(0,0),complexd(1,2),complexd(0,0)},
                             {complexd(2,0),complexd(2,1),complexd(2,2),complexd(2,3)}};

    cout << CM5 << endl;
    Matrix<complexd,2,MATRIX_TYPE::CSR> SCM1(CM5);
    Matrix<complexd,2,MATRIX_TYPE::CSR3> SCM2(CM5);

    cout << SCM1 << endl;
    cout << SCM2 << endl;

    SCM1.printData();
    SCM2.printData();

    cout << SpM11 << endl;
    cout << SpM1 << endl;


    Matrix<double,2,MATRIX_TYPE::CSR> SpM3 = SpM11 - SpM1;

    cout <<  SpM3 <<  endl;

    SpM3.printData();

    //SM.printData();


//    M(0,0) = 343;

//    cout << M << endl;

//    Matrix<double,2> M_slice = M(1,Slice(0,4,1));

//    cout << M_slice << endl;

//    Matrix<double,1> col2 = M.column(2);

//    cout << M.column(3)(2) << endl;

//    cout << col2 << endl;

//    cout << col2*5 << endl;

//    double val = 0.5;

//    Matrix<double,2> Mhalf = 1 - M*val;

//    //Mhalf = 1;
//    cout << Mhalf << endl;

//    Matrix<double,1> row2 = M.row(2);

//    cout << row2 << endl;

//    cout << Mhalf*row2 << endl;

//    Matrix<double,1> VecX(1000);
//    Matrix<double,1> VecY(1000);
//    Matrix<double,2> MM(2000,2000);
//    double value = 3.5;
//    for (size_t i = 0; i < VecX.size(); ++i){
//        VecX(i) = ++value;
//        VecY(i) = ++value;
//   }

//   for (size_t i = 0; i < MM.rows(); ++i){
//       for (size_t j = 0; j < MM.cols(); ++j){
//           MM(i,j) = i+j;
//       }
//   }

//    auto start = std::chrono::high_resolution_clock::now();
//    double res = accumulate(VecX.begin(),VecX.end(),0.0);
//    auto end = std::chrono::high_resolution_clock::now();
//    auto accumulate_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

//    start = std::chrono::high_resolution_clock::now();
//    double res_blas = cblas_dasum(int(VecX.size()),VecX.data(),1);
//    end = std::chrono::high_resolution_clock::now();
//    auto dasum_time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

//    double aval = 2.3;
//    start = std::chrono::high_resolution_clock::now();
//    //VecY += aval*VecX;
//    //transform(VecX.begin(),VecX.end(),VecY.begin(),VecY.begin(),[&aval](auto &elem1,auto &elem2){return aval*elem1 + elem2;});
//    //double rdot_nomkl = VecX*VecY;
//    //Matrix<double,2> VecYY = MM*VecX;
//    Matrix<double,2,Matrix_Type::GEN> C = MM*MM;
//    //cblas_dgemv(CBLAS_LAYOUT::CblasColMajor,CBLAS_TRANSPOSE::CblasNoTrans,MM.rows(),MM.cols(),aval,MM.data(),MM.cols(),
//                //VecX.data(),1,aval,VecY.data(),1);
//    end = std::chrono::high_resolution_clock::now();
//    auto ops_no_mkl = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

//    start = std::chrono::high_resolution_clock::now();
//    //cblas_daxpy(int(VecX.size()),aval,VecX.data(),1,VecY.data(),1);
//    //double rdot_mkl = cblas_ddot(VecX.size(),VecX.data(),1,VecY.data(),1);
//    //cblas_dgemv(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,MM.rows(),MM.cols(),1.0,MM.data(),MM.cols(),
//                //VecX.data(),1,0.0,VecY.data(),1);
//    end = std::chrono::high_resolution_clock::now();
//    auto ops_mkl = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();


//    cout << "vector_size = " << VecX.size() << " STD_LIB ACCUMULATE:" << "sum = " << res << "elapsed_time = " << accumulate_time << "\n";
//    cout << "vector_size = " << VecX.size() << " MKL_CBLAS dasum   :" << "sum = " <<res_blas << "elapsed_time = " << dasum_time << "\n";
//    cout << "vector_size = " << VecX.size() << " " << " NO_MKL daspy :" << "elapsed_time = " << ops_no_mkl << "\n";
//    cout << "vector_size = " << VecX.size() << " "  <<" MKL daspy    :" << "elapsed_time = " << ops_mkl << "\n";

//    Matrix<double,2,Matrix_Type::SYMM> sM(4);

//    sM = 4;

//    for (size_t i = 0; i < 4; ++i){
//        for (size_t j = 0; j < 4; ++j){
//            cout << sM(i,j) << "\t";
//        }
//        cout << "\n";
//    }

//    for (size_t i = 0; i < 4; ++i){
//        for (size_t j = 0; j < 4; ++j){
//            sM(i,j) = i+j;
//        }
//    }


//    for (size_t i = 0; i < 4; ++i){
//        for (size_t j = 0; j < 4; ++j){
//            cout << sM(i,j) << "\t";
//        }
//        cout << "\n";
//    }

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
