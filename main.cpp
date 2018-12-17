#include <iostream>
#include "ndimmatrix/matrix.h"
#include "mkl.h"
#include <chrono>

using namespace std;
using namespace matrix_impl;

constexpr int increment(int x){return ++x;}

typedef Matrix<double,2,Matrix_Type::GEN,Matrix_Storage_Scheme::FULL> Gen_Matrix;
typedef Matrix<double,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::FULL> SymFull_Matrix;
typedef Matrix<double,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::UPP> SymUpp_Matrix;
typedef Matrix<double,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::LOW> SymLow_Matrix;
typedef Matrix<double,2,Matrix_Type::UTR,Matrix_Storage_Scheme::UPP> UTR_Matrix;
typedef Matrix<double,2,Matrix_Type::LTR,Matrix_Storage_Scheme::LOW> LTR_Matrix;
typedef Matrix<std::complex<double>,2,Matrix_Type::HER,Matrix_Storage_Scheme::UPP> HUpp_Matrix;
typedef Matrix<std::complex<double>,2,Matrix_Type::HER,Matrix_Storage_Scheme::LOW> HLow_Matrix;
typedef Matrix<double,2,Matrix_Type::GEN,Matrix_Storage_Scheme::CSR3> Sparse_Matrix;

int main(){

    Gen_Matrix M = {{1,2,3,4},
                       {5,6,7,8},
                       {9,10,11,12}};


    SymFull_Matrix SMF = {{1,2,3,4},
                          {2,5,6,7},
                          {3,6,8,9},
                          {4,7,9,10}};



    SymLow_Matrix SML(SMF);

    SymUpp_Matrix SMU(SML);

    cout << M << endl;

    cout << SMF << endl;

    cout << (SMU+=SMF) << endl;
    cout << SML << endl;

    UTR_Matrix UTRM(3);
    UTRM = 10.0;

    LTR_Matrix LTRM(3);
    LTRM = -10.0;

    UTRM(2,2) = 5;

    cout << UTRM << endl;
    cout << LTRM << endl;

    HUpp_Matrix HUM(4);
    for (size_t i = 0; i < HUM.rows(); ++i){
        for (size_t j = i; j < HUM.cols(); ++j){
            HUM(i,j) = std::complex<double>(j,i+j);
        }
    }

    HLow_Matrix HLM(4);
    for (size_t i = 0; i < HUM.rows(); ++i){
        for (size_t j = 0; j <= i; ++j){
            HLM(i,j) = std::complex<double>(i+j,j);
        }
    }

    cout << HUM << endl;
    cout <<  HLM << endl;

    cout << (HUM+=HLM) << endl;
    cout << (HLM+=HUM) << endl;

    cout << (HUM+=HUM) << endl;
    cout << (HLM+=HLM) << endl;

    HUM+=10;
    HLM+=5.0;

    cout << HUM << endl;
    cout <<  HLM << endl;

    HUM*=10;
    HLM*=5.0;

    cout << HUM << endl;
    cout <<  HLM << endl;

    Sparse_Matrix SM;

    SM.setVals(0,0,1);
    SM.setVals(0,1,-1);
    SM.setVals(0,3,-3);
    SM.setVals(1,0,-2);
    SM.setVals(1,1,5);
    SM.setVals(2,2,4);
    SM.setVals(2,3,6);
    SM.setVals(2,4,4);
    SM.setVals(3,0,-4);
    SM.setVals(3,2,2);
    SM.setVals(3,3,7);
    SM.setVals(4,1,2);
    SM.setVals(4,4,-5);
    SM.setValsFinished();

    SM.printData();


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
