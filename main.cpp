#include <iostream>
#include "ndimmatrix/ndmatrix.h"
#include "ndimmatrix/symm_matrix.h"
#include "ndimmatrix/herm_matrix.h"
#include "ndimmatrix/sparse_matrix.h"
#include "ndimmatrix/matrix_arithmetic_ops.h"
//#include "mkl.h"
#include <chrono>

using namespace std;

constexpr int increment(int x){return ++x;}

Matrix<double,2,MATRIX_TYPE::CSR> sparseMatrix(){
    vector<double> values = { 1,-1,-3,-2, 5, 4, 6, 4,-4, 2, 7, 8,-5};
    vector<int_t> columns = { 0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
    vector<int_t> rows_start = {0,3,5,8,11};
    vector<int_t> rows_end = {3,5,8,11,13};
    return Matrix<double,2,MATRIX_TYPE::CSR>(5,5,rows_start,rows_end,columns,values);
}
//Matrix<double,2,MATRIX_TYPE::CSR> sparseMatrix1(){
//    vector<double> values = { 1, 1, 3, 2, 5, 4, 6, 4, 4, 2, 7, 8, 5};
//    vector<int_t> columns = { 0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 1, 4};
//    vector<int_t> rows_start = {0,3,5,8,11,13};
//    vector<int_t> rows_end = {3,5,8,11,13,13};
//    return Matrix<double,2,MATRIX_TYPE::CSR>(6,6,rows_start,rows_end,columns,values);
//}
Matrix<double,2,MATRIX_TYPE::SCSR> symmetricSparseMatrix(){
//    vector<double> values = { 1,-1,-3, 5, 4, 6, 4, 7,-5};
//    vector<int_t> columns = { 0, 1, 3, 1, 2, 3, 4, 3, 4};
//    vector<int_t> rows_start = {0,3,4,7,8};
//    vector<int_t> rows_end =   {3,4,7,8,9};
//    return Matrix<double,2,MATRIX_TYPE::SCSR>(5,rows_start,rows_end,columns,values);
    vector<double> values = { 1,-1,-3,-1, 5, 4, 6, 4,-3, 6, 7, 4,-5};
    vector<int_t> columns = { 0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 2, 4};
    vector<int_t> rows_start = {0,3,5,8,11};
    vector<int_t> rows_end =   {3,5,8,11,13};
    return Matrix<double,2,MATRIX_TYPE::SCSR>(5,rows_start,rows_end,columns,values);
}
Matrix<double,2,MATRIX_TYPE::SCSR> symmetricSparseMatrix_v1(){
    vector<double> values = { 1,-1,-3, 5, 4, 6, 4, 7,-5};
    vector<int_t> columns = { 0, 1, 3, 1, 2, 3, 4, 3, 4};
    vector<int_t> rows_start = {0,3,4,7,8};
    vector<int_t> rows_end =   {3,4,7,8,9};
    return Matrix<double,2,MATRIX_TYPE::SCSR>(5,rows_start,rows_end,columns,values);
//    vector<double> values = { 1,-1,-3,-1, 5, 4, 6, 4,-3, 6, 7, 4,-5};
//    vector<int_t> columns = { 0, 1, 3, 0, 1, 2, 3, 4, 0, 2, 3, 2, 4};
//    vector<int_t> rows_start = {0,3,5,8,11};
//    vector<int_t> rows_end =   {3,5,8,11,13};
//    return Matrix<double,2,MATRIX_TYPE::SCSR>(5,rows_start,rows_end,columns,values);
}
int main(){

    cout << "TESTING SPARSE MATRIX FACILITIES \n";
    Matrix<double,2,MATRIX_TYPE::CSR> SM(sparseMatrix());
    cout << SM << endl;
    Matrix<double,2,MATRIX_TYPE::CSR> SM2(SM);
    cout << SM2 << endl;
    Matrix<double,2,MATRIX_TYPE::CSR> SM3 = SM+SM2;
    cout <<  SM3 << endl;
    Matrix<double,2,MATRIX_TYPE::CSR> SM4 = SM2-SM3;
    cout <<  SM4 << endl;

    cout << SM2 << endl;
    cout << SM3 << endl;

    Matrix<double,2> CHECK(5,5,0.0);

    for (int i = 0; i < 5; ++i){
        for (int j = 0; j < 5; ++j){
            for (int k = 0; k < 5; ++k){
                CHECK(i,j) += SM2(i,k)*SM3(k,j);
            }

        }
    }

    Matrix<double,2> SM5 = SM2*SM3;
    cout << "********************sparse multiplation***************************************** \n";
    cout << "****Factors******* \n";
    cout << SM2 << endl;
    cout << SM3 << endl;
    cout << "****Product******* \n";
    cout <<  SM5 << endl;
    cout << CHECK << endl;



    Matrix<double,1> vec = {1,2,3,4,5};

    Matrix<double,1> rvec1 = SM5*vec;
    Matrix<double,1> rvec2 = CHECK*vec;


    Matrix<double,1> rvec3 = vec*SM5;
    Matrix<double,1> rvec4 = vec*CHECK;

    cout << rvec1 << endl;
    cout << rvec2 << endl;
    cout << rvec3 << endl;
    cout << rvec4 << endl;

    Matrix<double,2> DM1 = {{1,2,3,4,5},
                             {6,7,8,9,10}};


    Matrix<double,2> DM2 = { {1,2},
                             {3,4},
                             {5,6},
                             {7,8},
                             {9,10}};

    Matrix<double,2> DSR1 = DM1*SM5;
    Matrix<double,2> DSR2 = SM5*DM2;

    Matrix<double,2> DDR1 = DM1*CHECK;
    Matrix<double,2> DDR2 = CHECK*DM2;

    cout << "dense matrix - sparse matrix product \n ";
    cout << DSR1 << endl;
    cout << DDR1 << endl;

    cout << " sparse matrix - dense matrix product \n ";
    cout << DSR2 << endl;
    cout << DDR2 << endl;

    cout << "Rolando Yera Moreno" << "\n";


    cout << "************************testing symmetric sparse matrix**************************************\n";
    Matrix<double,2,MATRIX_TYPE::SCSR> SSM1 = symmetricSparseMatrix();
    SSM1.printData();
    cout << SSM1 << endl;

    Matrix<double,2,MATRIX_TYPE::SCSR> SSM2 = SSM1+SSM1;
    SSM2.printData();
    cout << SSM2 << endl;

    //Matrix<double,2,MATRIX_TYPE::CSR> SSM3 = SSM1*SSM1;
    //SSM2.printData();
    //cout << SSM3 << endl;

    Matrix<double,1> vvv1 = SSM1*vec;
    cout << vvv1 << endl;
    Matrix<double,2,MATRIX_TYPE::SCSR> SSM1_v1 = symmetricSparseMatrix_v1();
    Matrix<double,1> vvv1_v1 = SSM1_v1*vec;
    cout << vvv1_v1 << endl;

    Matrix<double,1> vvv2 = vec*SSM1;
    cout << vvv2 << endl;
    Matrix<double,1> vvv2_v1 = vec*SSM1_v1;
    cout << vvv2_v1 << endl;


//    sparse_matrix_t handlerC; //result sparse matrix handler;
//    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
//    status = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE,SSM1.descr(),SSM1.sparse_matrix_handler(),
//                             SPARSE_OPERATION_NON_TRANSPOSE,SSM1.descr(),SSM1.sparse_matrix_handler(),
//                             SPARSE_STAGE_FULL_MULT,&handlerC);

    Matrix<double,2> SSM3 = CHECK*SSM1;
    cout << SSM3 << endl;
    Matrix<double,2> SSM3_v1 = CHECK*SSM1_v1;
    cout << SSM3_v1 << endl;

    Matrix<double,2> SSM3_ = SSM1*CHECK;
    cout << SSM3_ << endl;
    Matrix<double,2> SSM3_v1_ = SSM1_v1*CHECK;
    cout << SSM3_v1_ << endl;

    //vec = vec*SM5;

    //cout << vec << endl;


    //SM = sparseMatrix1();

    //cout << SM << endl;

    //cout << SM2 << endl;


//    Matrix<double,2> M1 = {{1,2,3,4},
//                          {5,6,7,8},
//                          {9,10,11,12}};


//Matrix<double,3> M3D(4,4,4);
//M3D= 5;

//Matrix<double,3> M3D1 = M3D.apply(std::cos);
////Matrix<double,3> M3D1 = 3
////M3D= 5;

////cout << M3D1.extent(0) << " " << M3D1.extent(1) << " " << M3D1.extent(2) << " ";// << M3D1(0,0,0) << " " << M3D1(0,0,0) << "\n";

//for (int i = 0; i < 4; ++i) cout << "cos(" << M3D(i,i,i) << ") = " << M3D1(i,i,i) << "\n";

//Matrix<double,2> M1 = {{1,2,3,4},
//                        {2,5,6,7},
//                        {3,6,8,9},
//                        {4,7,9,10}};

//Matrix<double,2> M2 = {{0,0,0,0},
//                        {3,0,0,3},
//                        {3,3,0,0},
//                        {3,0,3,0}};

//    Matrix<double,2> M3 = M1 + M2 + M1;

//    cout << setprecision(3) << M3 << endl;

//    int val(3);

//    M3*=val;

//    cout << setprecision(3) << M3 << endl;

//    Matrix<double,2> M4 = M1*M2;

//    Matrix<complexd,2> CM(4,4,complexd(1,2));

//    Matrix<complexd,2> CM1 = CM+M1;
//    Matrix<complexd,2> CM2 = CM-M1;
//    Matrix<complexd,2> CM3 = M1+CM;
//    Matrix<complexd,2> CM4 = M1-CM;

//    cout << setprecision(3) << CM1 << endl;
//    cout << setprecision(3) << CM2 << endl;
//    cout << setprecision(3) << CM3 << endl;
//    cout << setprecision(3) << CM4 << endl;

//    cout << 1.0-complexd(1,2) << endl;

//    Matrix<int,2> MI1 = {{1,2,3,4},
//                            {2,5,6,7},
//                            {3,6,8,9},
//                            {4,7,9,10}};

//    cout << (M1+=MI1) << endl;

//    cout << setprecision(3) << M4 << endl;

//    Matrix<double,2,MATRIX_TYPE::SYMM> SM1(M1);

//    cout << SM1 << endl;
//    valarray<uint32_t> indx1 = {0,1,2};
//    Matrix<double,2> indM1 = SM1(indx1,indx1);
//    if (indM1.matrix_type == MATRIX_TYPE::GEN) cout << "GEN" << endl;

//    cout << indM1 << endl;

//    cout << setprecision(3) << M1 << endl;

//    cout << setprecision(3) << SM1 << endl;

//    cout << setprecision(3) << (SM1+=M1) << endl;
//    cout << setprecision(3) << (SM1+=SM1) << endl;

//    cout << setprecision(3) << SM1 << endl;


//    Matrix<std::complex<double>,2,MATRIX_TYPE::HER> HM1(4);
//    for (size_t i = 0; i < HM1.rows(); ++i){
//        for (size_t j = i; j < HM1.cols(); ++j){
//            HM1(i,j) = std::complex<double>(j,i+j);
//        }
//    }
//    cout << setprecision(3) << HM1 << endl;
//    cout << (HM1+=HM1) << endl;
//    cout << (HM1+=SM1) << endl;

//    HM1 += 10;

//    cout << setprecision(3) << HM1 << endl;


//    HM1*=10;

//    cout << setprecision(3) << HM1 << endl;

//    Matrix<double,2,MATRIX_TYPE::CSR> SpM1(M2);

//    cout << M2 << endl;

//    cout << SpM1 << endl;
//    SpM1.printData();

//    Matrix<double,2,MATRIX_TYPE::CSR3> SpM2(M2);

//    cout << SpM2 << endl;
//    SpM2.printData();

//    vector<uint32_t> iindx = {1,2,3,0};
//    vector<uint32_t> jindx = {0,1,2,3};

//    Matrix<double,2,MATRIX_TYPE::CSR> SpM11 = SpM1(iindx,jindx);

//    Matrix<double,2,MATRIX_TYPE::CSR3> SpM22 = SpM2(iindx,jindx);

//    cout << "new data here" << endl;

//    SpM11.printData();
//    SpM22.printData();

//    cout << SpM11 << endl;
//    cout << SpM22 << endl;

//    Matrix<complexd,2> CM5 = {{complexd(0,0),complexd(0,1),complexd(0,0),complexd(1,2)},
//                             {complexd(1,0),complexd(0,0),complexd(1,2),complexd(0,0)},
//                             {complexd(2,0),complexd(2,1),complexd(2,2),complexd(2,3)}};

//    cout << CM5 << endl;
//    Matrix<complexd,2,MATRIX_TYPE::CSR> SCM1(CM5);
//    Matrix<complexd,2,MATRIX_TYPE::CSR3> SCM2(CM5);

//    cout << SCM1 << endl;
//    cout << SCM2 << endl;

//    SCM1.printData();
//    SCM2.printData();

//    cout << SpM11 << endl;
//    cout << SpM1 << endl;


//    Matrix<double,2,MATRIX_TYPE::CSR> SpM3 = SpM11 - SpM1;

//    cout <<  SpM3 <<  endl;

//    SpM3.printData();

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


//Matrix<double,3,MATRIX_TYPE::SYMM> HM;

//Matrix<double,2,MATRIX_TYPE::SYMM> SM1(4);
//Matrix<double,2,MATRIX_TYPE::SYMM> SM2(4);
//Matrix<complex<double>,2,MATRIX_TYPE::GEN> GMC(4,4);
//Matrix<double,2,MATRIX_TYPE::GEN> GM(4,4);
//Matrix<complex<double>,2,MATRIX_TYPE::HER> HM1(4);
//Matrix<complex<double>,2,MATRIX_TYPE::HER> HM2(4);


//complex<double> tmp = complex<double>(1,1);
//for (int i = 0; i < SM1.rows(); ++i)
//    for (int j = i; j < SM1.cols(); ++j){
//        SM1(i,j) = i*i;
//        SM2(i,j) = i+j;
//        GMC(i,j) = complex<double>(i*j,i+j);
//        HM1(i,j) = complex<double>(i*j,i+j);
//        HM2(i,j) = complex<double>(i+j,i*j);
//    }

//cout << HM1 << "\n" << HM2 << endl;
//auto R1 = HM1 + HM2;
//cout << "suma de dos matrices hermiticas: \n" << R1 << endl;
//auto R2 = SM1 + SM2;
//cout << "suma de dos matrices simetricas: \n" << R2 << endl;
//auto R3 = SM1+HM1;
//auto R4 = HM2-SM2;
//auto R5 = HM1+=5;
//cout << "suma de matriz simetrica y hermitica \n" << R3 << "\n" << R4 << "\n" << R5 << endl;
//auto R6 = GMC + SM1;
//auto R7 = HM2 + GMC;
//cout << R6 << "\n" << R7 << endl;


//Matrix<complex<double>,1> vec = {complex<double>(1,0),complex<double>(1,0),complex<double>(1,0),complex<double>(1,0)};
//Matrix<double,1> vec1 = {1,1,1,1};

//auto R8 = HM1*vec1;
//auto R9 = SM1*vec1;
//cout << "mulstiplicacion Matriz vector \n" << R9 << "\n" << R8;
//cout << SM1+SM2 << endl;
//cout  << GMC + GM << endl;
//cout << SM2-SM1 << endl;
//cout << GMC+SM1 << endl;
//auto A =  SM1 + TESTHER;

//complex<double> tmp(1,2);
//auto tmp2 = 4.0/tmp;
//cout << tmp2 << endl;
//cout << SM1 << endl;

//cout << SM2 << endl;
//cout << SM2/tmp << endl;
//cout << tmp2*SM2 << endl;

//Matrix<complex<double>,1,MATRIX_TYPE::GEN> cvec = {complex<double>(1,2),complex<double>(2,2)};
//cout << "norma de vector complejo = " << cvec.norm2() << endl;
//cout << "norma cuadrada de vector complejo = " << cvec.norm2sqr() << endl;

//Matrix<double,1,MATRIX_TYPE::GEN> rvec = {1,2};

//Matrix<complex<double>,1> cvecres = -cvec-rvec;



//cout << cvecres << endl;


//Matrix<double,2,MATRIX_TYPE::GEN> M = {{2,4,6,8},
//                      {1,3,5,7}};

//Matrix<double,2,MATRIX_TYPE::GEN> R = M(0,slice(0,3,1));

//cout << "complejo + Matrix Real \n";
//cout << R << endl;
//cout << tmp+R << endl;

//M += M;
//Matrix<double,2,MATRIX_TYPE::GEN> T = R;
//M = R - 2*T;
//cout << M << "\n";
//cout << R << endl;

//T*=30;
//cout << T << endl;
//T/=15;
//cout << T << endl;

//Matrix<double,1,MATRIX_TYPE::GEN> vec1 = M.row(0);

//cout << vec1 << "\n";

//Matrix<double,1,MATRIX_TYPE::GEN> v3 = {4,5,6,7};

//cout << v3 << endl;

//v3 = {1,2,3,4,5,6,7,8,9};

//cout << v3 <<  endl;

//v3 = -vec1;

//cout << v3 << endl;

////cout << "Norma M = " << M.norm2() << " ---" << "vector v = " << v3.norm2() << endl;


//v3 = vec1 + 4;
//cout << v3 << endl;

//v3 = v3 - vec1;
//cout << v3 << endl;


    return 0;
}
