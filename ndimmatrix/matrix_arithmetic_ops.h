#ifndef MATRIX_ARITHMETIC_OPS_H
#define MATRIX_ARITHMETIC_OPS_H

#include "herm_matrix.h"
#include "symm_matrix.h"
#include "sparse_matrix.h"


template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR> sparse_add(const Matrix<T,2,MATRIX_TYPE::CSR> &sm1,const Matrix<T,2,MATRIX_TYPE::CSR> &sm2){


    if (sm1.nnz() == 0) return sm2;
    if (sm2.nnz() == 0) return sm1;

    sparse_matrix_t handlerC; //result sparse matrix handler;
    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    sparse_status_t status1 = SPARSE_STATUS_NOT_SUPPORTED;

    sparse_index_base_t index;
    int rows;
    int cols;
    int *rowStart;
    int *rowEnd;
    int *columns;
    T *values;
    if constexpr (is_same_v<T,float>){
        status = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE,sm1.sparse_matrix_handler()
                                                                      ,1,sm2.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_s_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }else if constexpr (is_same_v<T,double>) {
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,sm1.sparse_matrix_handler()
                                                                      ,1,sm2.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_d_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);

    }else if  constexpr (is_same_v<T,complex<float>>){
        complex<float> alpha(1,0);
        status = mkl_sparse_c_add(SPARSE_OPERATION_NON_TRANSPOSE,sm1.sparse_matrix_handler()
                                                                      ,alpha,sm2.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_c_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);

    }else if constexpr (is_same_v<T,complex<double>>){
        complex<double> alpha(1,0);
        status = mkl_sparse_z_add(SPARSE_OPERATION_NON_TRANSPOSE,sm1.sparse_matrix_handler()
                                                                      ,alpha,sm2.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_z_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }
    check_sparse_operation_status(status1);

    int nnz = rowEnd[rows-1]-rowStart[0];

    vector<int_t> rows_start(rowStart,rowStart+rows);
    vector<int_t> rows_end(rowEnd,rowEnd+rows);
    vector<int_t> columns_index(columns,columns+nnz);
    vector<T> vals(values,values+nnz);

    mkl_sparse_destroy(handlerC); //destroy handler

    return Matrix<T,2,MATRIX_TYPE::CSR>(rows,cols,move(rows_start),move(rows_end),move(columns_index),move(vals));

}
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR> sparse_sub(const Matrix<T,2,MATRIX_TYPE::CSR> &sm1,const Matrix<T,2,MATRIX_TYPE::CSR> &sm2){

    if (sm1.nnz() == 0) return -sm2;
    if (sm2.nnz() == 0) return sm1;

    sparse_matrix_t handlerC; //result sparse matrix handler;
    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    sparse_status_t status1 = SPARSE_STATUS_NOT_SUPPORTED;

    sparse_index_base_t index;
    int rows;
    int cols;
    int *rowStart;
    int *rowEnd;
    int *columns;
    T *values;
    if constexpr (is_same_v<T,float>){
        status = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE,sm2.sparse_matrix_handler()
                                                                      ,-1,sm1.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_s_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }else if constexpr (is_same_v<T,double>) {
        status = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE,sm2.sparse_matrix_handler()
                                                                      ,-1,sm1.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_d_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);

    }else if  constexpr (is_same_v<T,complex<float>>){
        complex<float> alpha(-1,0);
        status = mkl_sparse_c_add(SPARSE_OPERATION_NON_TRANSPOSE,sm2.sparse_matrix_handler()
                                                                      ,alpha,sm1.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_c_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);

    }else if constexpr (is_same_v<T,complex<double>>){
        complex<double> alpha(-1,0);
        status = mkl_sparse_z_add(SPARSE_OPERATION_NON_TRANSPOSE,sm2.sparse_matrix_handler()
                                                                      ,alpha,sm1.sparse_matrix_handler(),&handlerC);
        check_sparse_operation_status(status); //checking for operation success
        status = mkl_sparse_order(handlerC);
        check_sparse_operation_status(status);
        status1 = mkl_sparse_z_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }
    check_sparse_operation_status(status1);

    int nnz = rowEnd[rows-1]-rowStart[0];

    vector<int_t> rows_start(rowStart,rowStart+rows);
    vector<int_t> rows_end(rowEnd,rowEnd+rows);
    vector<int_t> columns_index(columns,columns+nnz);
    vector<T> vals(values,values+nnz);

    mkl_sparse_destroy(handlerC); //destroy handler

    return Matrix<T,2,MATRIX_TYPE::CSR>(rows,cols,move(rows_start),move(rows_end),move(columns_index),move(vals));
}
template<typename T>
inline Matrix<T,1> sparse_matrix_vector_prod(const Matrix<double,2,MATRIX_TYPE::CSR> &sm,const Matrix<T,1> &v){
    assert(sm.cols() == v.size());
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    Matrix<T,1> res(sm.rows());
    T d;
    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;;
    if constexpr (is_same_v<T,float>){
        float alpha = 1;
        float beta = 0;
        status = mkl_sparse_s_dotmv(SPARSE_OPERATION_NON_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                               ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);
    }else if constexpr (is_same_v<T,double>) {
        double alpha = 1;
        double beta = 0;
        status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                               ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);

    }else if constexpr (is_same_v<T,float>) {
        complex<float> alpha(1,0);
        complex<float> beta(0,0);
        status = mkl_sparse_c_dotmv(SPARSE_OPERATION_NON_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                               ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);

    }else if constexpr (is_same_v<T,float>){
        complex<double> alpha(1,0);
        complex<double> beta(0,0);
        status = mkl_sparse_z_dotmv(SPARSE_OPERATION_NON_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                               ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);
    }

    return res;
}
template<typename T>
inline Matrix<T,1> vector_sparse_matrix_prod(const Matrix<T,1> &v,const Matrix<double,2,MATRIX_TYPE::CSR> &sm){
    assert(sm.cols() == v.size());
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    Matrix<T,1> res(sm.rows());
    T d;
    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;;
    if constexpr (is_same_v<T,float>){
        float alpha = 1;
        float beta = 0;
        status = mkl_sparse_s_dotmv(SPARSE_OPERATION_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                           ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);
    }else if constexpr (is_same_v<T,double>) {
        double alpha = 1;
        double beta = 0;
        status = mkl_sparse_d_dotmv(SPARSE_OPERATION_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                           ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);

    }else if constexpr (is_same_v<T,float>) {
        complex<float> alpha(1,0);
        complex<float> beta(0,0);
        status = mkl_sparse_c_dotmv(SPARSE_OPERATION_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                           ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);

    }else if constexpr (is_same_v<T,float>){
        complex<double> alpha(1,0);
        complex<double> beta(0,0);
        status = mkl_sparse_z_dotmv(SPARSE_OPERATION_TRANSPOSE,alpha,sm.sparse_matrix_handler()
                                                                           ,descr,v.begin(),beta,res.begin(),&d);
        check_sparse_operation_status(status);
    }

    return res;
}



//Arithmetic Operation
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator +(const Scalar &val,const Matrix<T,N> &m){
//    Matrix<T,N> R(m);
//    return R+=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>> operator +(const Scalar &val,const Matrix<T,N,mtype> &m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val + ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val + R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val + ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator +(const Scalar &val,Matrix<T,N> &&m){
//    Matrix<T,N> R(m);
//    return R+=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator +(const Scalar &val,Matrix<T,N,mtype> &&m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val + ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val + R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val + ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator +(const Matrix<T,N> &m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R+=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator +(const Matrix<T,N,mtype> &m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele + val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j) + val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele + val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator +(Matrix<T,N> &&m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R+=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator +(Matrix<T,N,mtype> &&m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele + val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R+=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j) + val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele + val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N,MATRIX_TYPE mtype>
//inline std::enable_if_t<std::is_arithmetic_v<Scalar> || std::is_same_v<T,std::complex<double>>
//                        || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long double>>,
//                        Matrix<T,N,mtype>> operator -(const Scalar &val,const Matrix<T,N,mtype> &m){
//    Matrix<T,N,mtype> R(m);
//    std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
//    return R;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator -(const Scalar &val,const Matrix<T,N,mtype> &m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
            return R;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val - ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
            return R;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val - R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val - ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator -(const Scalar &val,Matrix<T,N> &&m){
//    Matrix<T,N> R(m);
//    std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
//    return R;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator -(const Scalar &val,Matrix<T,N,mtype> &&m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
            return R;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val - ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
            return R;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val - R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val - ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N,MATRIX_TYPE mtype>
//inline Matrix<T,N,mtype> operator -(const Matrix<T,N,mtype> &m,const Scalar &val){
//    Matrix<T,N,mtype> R(m);
//    return R-=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator -(const Matrix<T,N,mtype> &m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R-=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele - val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R-=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j) - val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele - val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator -(Matrix<T,N> &&m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R-=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator -(Matrix<T,N,mtype> &&m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R-=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele - val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R-=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j) - val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele - val;});
                return R;
            }
        }
    }
}

//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator *(const Scalar &val,const Matrix<T,N> &m){
//    Matrix<T,N> R(m);
//    return R*=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator *(const Scalar &val,const Matrix<T,N,mtype> &m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val*ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val*R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val*ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator *(const Scalar &val,Matrix<T,N> &&m){
//    Matrix<T,N> R(m);
//    return R*=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator *(const Scalar &val,Matrix<T,N,mtype> &&m){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val*ele;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = val*R(i,j);
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return val*ele;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator *(const Matrix<T,N> &m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R*=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator *(const Matrix<T,N,mtype> &m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele*val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j)*val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele*val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator *(Matrix<T,N> &&m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R*=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator *(Matrix<T,N,mtype> &&m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele*val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R*=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j)*val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele*val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator /(const Matrix<T,N> &m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R/=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator /(const Matrix<T,N,mtype> &m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R/=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele/val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R/=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j)/val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele/val;});
                return R;
            }
        }
    }
}
//template<typename T,typename Scalar,size_t N>
//inline Matrix<T,N> operator /(Matrix<T,N> &&m,const Scalar &val){
//    Matrix<T,N> R(m);
//    return R/=val;
//}
template<typename T,typename Scalar,typename RT = common_type_t<T,Scalar>,size_t N,MATRIX_TYPE mtype>
inline enable_if_t<is_number<Scalar>(), conditional_t<is_complex<Scalar>() && (mtype == MATRIX_TYPE::HER),
                                                      Matrix<RT,N,MATRIX_TYPE::GEN>,Matrix<RT,N,mtype>>>  operator /(Matrix<T,N,mtype> &&m,const Scalar &val){

    //Scalar es de tipo real (ie: double,float,int)
    if constexpr (is_arithmetic_v<Scalar>){
        if constexpr (is_same_v<RT,T>){ //el tipo comun es T
            Matrix<RT,N,mtype> R(m);
            return R/=val;
        }else{ //el tipo comun es Scalar
            Matrix<RT,N,mtype> R(m.descriptor());
            transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele/val;});
            return R;
        }
    }else{ // Scalar es tipo complejo
        if constexpr (is_same_v<RT,T> && mtype != MATRIX_TYPE::HER){
            Matrix<RT,N,mtype> R(m);
            return R/=val;
        }else{ //matriz hermitica
            if constexpr (MATRIX_TYPE::HER == mtype){
                Matrix<RT,2,MATRIX_TYPE::GEN> R(m.descriptor());
                for (size_t i = 0; i < R.extent(0); ++i)
                    for (size_t j = 0; j < R.extent(1); ++j)
                        R(i,j) = R(i,j)/val;
                return R;
            }else{ //Scalar is the common type
                Matrix<RT,N,mtype> R(m.descriptor());
                transform(m.begin(),m.end(),R.begin(),[val](const auto &ele){return ele/val;});
                return R;
            }
        }
    }
}

//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator +(const Matrix<T,N,mtype1> &m1,const Matrix<U,N,mtype2> &m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_add(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R+=m2;
        }else{
            Matrix<RT,N,mtype1> R(m2);
            return R+=m1;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
            transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1+v2;});
            return R;
        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) + m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator +(Matrix<T,N,mtype1> &&m1,const Matrix<U,N,mtype2> &m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_add(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R+=m2;
        }else{
            Matrix<RT,N,mtype1> R(m2);
            return R+=m1;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype1 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1);
                transform(R.begin(),R.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 + v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
                transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1+v2;});
                return R;
            }

        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) + m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator +(const Matrix<T,N,mtype1> &m1,Matrix<U,N,mtype2> &&m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_add(m1,m2);
        }else if constexpr (is_same_v<RT,T> && !is_same_v<T,U>){
            Matrix<RT,N,mtype1> R(m1);
            return R+=m2;
        }else{
            Matrix<RT,N,mtype1> R(m2);
            return R+=m1;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype2 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m2);
                transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 + v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
                transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1+v2;});
                return R;
            }
        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) + m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator +(Matrix<T,N,mtype1> &&m1,Matrix<U,N,mtype2> &&m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_add(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R+=m2;
        }else{
            Matrix<RT,N,mtype1> R(m2);
            return R+=m1;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype1 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1);
                transform(R.begin(),R.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 + v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m2);
                transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const T &v1,const U &v2){return v1+v2;});
                return R;
            }

        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) + m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator -(const Matrix<T,N,mtype1> &m1,const Matrix<U,N,mtype2> &m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_sub(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R-=m2;
        }else{
            assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                              m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));
            Matrix<RT,N,mtype1> R(m2.descriptor());
            transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
            return R;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
            transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1-v2;});
            return R;
        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) - m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator -(Matrix<T,N,mtype1> &&m1,const Matrix<U,N,mtype2> &m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_sub(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R-=m2;
        }else{
            assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                              m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));
            Matrix<RT,N,mtype1> R(m2.descriptor());
            transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
            return R;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype1 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1);
                transform(R.begin(),R.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
                transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1-v2;});
                return R;
            }

        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) - m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator -(const Matrix<T,N,mtype1> &m1,Matrix<U,N,mtype2> &&m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_sub(m1,m2);
        }else if constexpr (is_same_v<RT,T> && !is_same_v<T,U>){
            Matrix<RT,N,mtype1> R(m1);
            return R-=m2;
        }else{
            assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                              m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));
            Matrix<RT,N,mtype1> R(m2);
            transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
            return R;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype2 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m2);
                transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1.rows());
                transform(m1.begin(),m1.end(),m2.begin(),R.begin(),[](const T &v1,const U &v2){return v1-v2;});
                return R;
            }
        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) - m2(i,j);
            return R;
        }
    }
}
//NOTE: Aca no esta incluido el caso de una matriz diagonal de forma optimizada
template<typename T,typename U,typename RT = common_type_t<T,U>,size_t N,MATRIX_TYPE mtype1,MATRIX_TYPE mtype2>
inline auto operator -(Matrix<T,N,mtype1> &&m1,Matrix<U,N,mtype2> &&m2){
    if constexpr (mtype1 == mtype2){
        if constexpr (mtype1 == MATRIX_TYPE::CSR){
            return sparse_sub(m1,m2);
        }else if constexpr (is_same_v<RT,T>){
            Matrix<RT,N,mtype1> R(m1);
            return R-=m2;
        }else{
            assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                              m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));
            Matrix<RT,N,mtype1> R(m2);
            transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
            return R;
        }
    }else{
        static_assert (N==2,"For other Matrix Type diferent than GEN the dimension must be 2.");
        assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

        if constexpr ((mtype1 == MATRIX_TYPE::HER && mtype2 == MATRIX_TYPE::SYMM && is_arithmetic_v<U>)
                      || (mtype2 == MATRIX_TYPE::HER && mtype1 == MATRIX_TYPE::SYMM && is_arithmetic_v<T>)){
            if constexpr (mtype1 == MATRIX_TYPE::HER){
                Matrix<RT,2,MATRIX_TYPE::HER> R(m1);
                transform(R.begin(),R.end(),m2.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1 - v2;});
                return R;
            }else{
                Matrix<RT,2,MATRIX_TYPE::HER> R(m2);
                transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const T &v1,const U &v2){return v1-v2;});
                return R;
            }

        }else{
            Matrix<RT,2,MATRIX_TYPE::GEN> R(m1.descriptor());
            for (size_t i = 0; i < R.rows(); ++i)
                for (size_t j = 0; j < R.cols(); ++j)
                    R(i,j) = m1(i,j) - m2(i,j);
            return R;
        }
    }
}
//template<typename T,typename U,typename RT = std::common_type_t<T,U>,size_t N,MATRIX_TYPE mtype>
//inline Matrix<T,N,mtype> operator -(const Matrix<T,N,mtype> &m1,const Matrix<U,N,mtype> &m2){
//    if constexpr (std::is_same_v<RT,T>){
//        Matrix<RT,N,mtype> R(m1);
//        return R-=m2;
//    }else{
//        Matrix<RT,N,mtype> R(-m2);
//        return R+=m1;
//    }
//}
//template<typename T,typename U,typename RT = std::common_type_t<T,U>,size_t N>
//inline Matrix<T,N> operator -(Matrix<T,N> &&m1,const Matrix<U,N> &m2){
//    if constexpr (std::is_same_v<RT,T>){
//        Matrix<RT,N> R(m1);
//        return R-=m2;
//    }else{
//        Matrix<RT,N> R(-m2);
//        return R+=m1;
//    }
//}
//template<typename T,typename U,typename RT = std::common_type_t<T,U>,size_t N>
//inline Matrix<T,N> operator -(const Matrix<T,N> &m1,Matrix<U,N> &&m2){
//    assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
//                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));

//    if constexpr (std::is_same_v<RT,T> && !std::is_same_v<T,U>){
//        Matrix<RT,N> R(m1);
//        return R-=m2;
//    }else{
//        Matrix<RT,N> R(m2);
//        std::transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const T &v1,const T &v2){return v1-v2;});
//        return R;
//    }
//}
template <typename T>
inline T operator *(const Matrix<T,1> &v1, const Matrix<T,1> &v2){
    assert(v1.size() == v2.size());
    if constexpr (is_arithmetic_v<T>){
        return inner_product(v1.begin(),v1.end(),v2.begin(),0);
    }else{
        return inner_product(v1.begin(),v1.end(),v2.begin(),complex<double>(0,0));
    }
}
template <typename T,MATRIX_TYPE mtype>
inline Matrix<T,1> operator *(const Matrix<T,2,mtype> &m,const Matrix<T,1> &v){
    assert(m.cols() == v.size());
    Matrix<T,1> r(m.cols());
    if constexpr (mtype == MATRIX_TYPE::GEN){
        if constexpr (is_same_v<T,float>){
            cblas_sgemv(CblasRowMajor,CblasNoTrans,m.rows(),m.cols(),1.0,m.begin(),m.cols(),v.begin(),1,0,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,double>){
            cblas_dgemv(CblasRowMajor,CblasNoTrans,m.rows(),m.cols(),1.0,m.begin(),m.cols(),v.begin(),1,0,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,complex<float>>){
            double alpha = 1.0;
            double beta = 0.0;
            cblas_cgemv(CblasRowMajor,CblasNoTrans,m.rows(),m.cols(),&alpha,m.begin(),m.cols(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,complex<double>>){
            double alpha = 1.0;
            double beta = 0.0;
            cblas_zgemv(CblasRowMajor,CblasNoTrans,m.rows(),m.cols(),&alpha,m.begin(),m.cols(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }
    }

    if constexpr (mtype == MATRIX_TYPE::HER){
        double alpha = 1.0;
        double beta = 0.0;
        if constexpr (is_same_v<T,complex<float>>){
            cblas_chpmv(CblasRowMajor,CblasUpper,m.rows(),&alpha,m.begin(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }
        if constexpr (is_same_v<T,complex<double>>){
            cblas_zhpmv(CblasRowMajor,CblasUpper,m.rows(),&alpha,m.begin(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }

        throw ("ERROR: no matrix-vector multiplication implemented for data types");
    }

    if constexpr (mtype == MATRIX_TYPE::SYMM){
        if constexpr (is_same_v<T,float>){
            cblas_sspmv(CblasRowMajor,CblasUpper,m.rows(),1.0,m.begin(),v.begin(),1,0.0,r.begin(),1);
            return r;
        }
        if constexpr (is_same_v<T,double>){
            cblas_dspmv(CblasRowMajor,CblasUpper,m.rows(),1.0,m.begin(),v.begin(),1,0.0,r.begin(),1);
            return r;
        }
        throw ("ERROR: no matrix-vector multiplication implemented for data types");
    }

    if constexpr (mtype == MATRIX_TYPE::CSR){
        return sparse_matrix_vector_prod(m,v);
    }

    throw ("ERROR: no matrix-vector multiplication implemented for data types");
}
template <typename T,MATRIX_TYPE mtype>
inline Matrix<T,1> operator *(const Matrix<T,1> &v, const Matrix<T,2,mtype> &m){
    assert(m.cols() == v.size());
    Matrix<T,1> r(m.cols());
    if constexpr (mtype == MATRIX_TYPE::GEN){
        if constexpr (is_same_v<T,float>){
            cblas_sgemv(CblasRowMajor,CblasTrans,m.rows(),m.cols(),1.0,m.begin(),m.cols(),v.begin(),1,0,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,double>){
            cblas_dgemv(CblasRowMajor,CblasTrans,m.rows(),m.cols(),1.0,m.begin(),m.cols(),v.begin(),1,0,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,complex<float>>){
            double alpha = 1.0;
            double beta = 0.0;
            cblas_cgemv(CblasRowMajor,CblasTrans,m.rows(),m.cols(),&alpha,m.begin(),m.cols(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }else if constexpr (is_same_v<T,complex<double>>){
            double alpha = 1.0;
            double beta = 0.0;
            cblas_zgemv(CblasRowMajor,CblasTrans,m.rows(),m.cols(),&alpha,m.begin(),m.cols(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }
    }

    if constexpr (mtype == MATRIX_TYPE::HER){
        double alpha = 1.0;
        double beta = 0.0;
        if constexpr (is_same_v<T,complex<float>>){
            cblas_chpmv(CblasRowMajor,CblasUpper,m.rows(),&alpha,m.begin(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }
        if constexpr (is_same_v<T,complex<double>>){
            cblas_zhpmv(CblasRowMajor,CblasUpper,m.rows(),&alpha,m.begin(),v.begin(),1,&beta,r.begin(),1);
            return r;
        }

        throw ("ERROR: no matrix-vector multiplication implemented for data types");
    }

    if constexpr (mtype == MATRIX_TYPE::SYMM){
        if constexpr (is_same_v<T,float>){
            cblas_sspmv(CblasRowMajor,CblasUpper,m.rows(),1.0,m.begin(),v.begin(),1,0.0,r.begin(),1);
            return r;
        }
        if constexpr (is_same_v<T,double>){
            cblas_dspmv(CblasRowMajor,CblasUpper,m.rows(),1.0,m.begin(),v.begin(),1,0.0,r.begin(),1);
            return r;
        }
        throw ("ERROR: no matrix-vector multiplication implemented for data types");
    }

    if constexpr (mtype == MATRIX_TYPE::CSR){
        return vector_sparse_matrix_prod(v,m);
    }
    throw ("ERROR: no matrix-vector multiplication implemented for data types");
}
template<typename T,typename U,typename RT = common_type_t<T,U>>
inline Matrix<RT,2> operator*(const Matrix<T,2> &m1,const Matrix<U,2> &m2){
    size_t m1_rows = m1.rows();
    size_t m1_cols = m1.cols();
    size_t m2_rows = m2.rows();
    size_t m2_cols = m2.cols();
    assert(m1_cols == m2_rows);
    Matrix<RT,2> res(m1_rows,m2_cols);
    if constexpr (is_same_v<T,float> && is_same_v<U,float>){
        cblas_sgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    int(m1_rows),int(m2_cols),int(m1_cols),1.0,m1.begin(),
                    int(m1_cols),m2.begin(),int(m2_cols),0.0,res.begin(),int(m2_cols));
        return res;

    }
    if constexpr (is_same_v<T,double> && is_same_v<U,double>){
        cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    int(m1_rows),int(m2_cols),int(m1_cols),1.0,m1.begin(),
                    int(m1_cols),m2.begin(),int(m2_cols),0.0,res.begin(),int(m2_cols));
        return res;

    }
    if constexpr (is_same_v<T,complex<float>> || is_same_v<U,complex<float>>){
        double alpha = 1.0;
        double beta = 0.0;
        cblas_cgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    int(m1_rows),int(m2_cols),int(m1_cols),&alpha,m1.begin(),
                    int(m1_cols),m2.begin(),int(m2_cols),&beta,res.begin(),int(m2_cols));
        return res;
    }
    if constexpr (is_same_v<T,complex<double>> || is_same_v<U,complex<double>>){
        double alpha = 1.0;
        double beta = 0.0;
        cblas_zgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                    int(m1_rows),int(m2_cols),int(m1_cols),&alpha,m1.begin(),
                    int(m1_cols),m2.begin(),int(m2_cols),&beta,res.begin(),int(m2_cols));
        return res;
    }

    throw ("ERROR: no matrix-matrix multiplication implemented for data types");

}

//sparse ops

template<typename T>
inline Matrix<T,2> operator *(const Matrix<T,2,MATRIX_TYPE::CSR> &sm,const Matrix<T,2> &dm){
    assert(sm.cols() == dm.rows());

    int row = sm.rows();
    int col = dm.cols();

    Matrix<T,2> R(row,col);

    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    sparse_operation_t op = SPARSE_OPERATION_NON_TRANSPOSE;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_layout_t sparseLayout = SPARSE_LAYOUT_ROW_MAJOR;
    if constexpr (is_same_v<T,float>){
        status = mkl_sparse_s_mm(op,1,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),col,col,0,R.begin(),col);
    }else if constexpr (is_same_v<T,double>) {
        status = mkl_sparse_d_mm(op,1,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),col,col,0,R.begin(),col);

    }else if constexpr (is_same_v<T,complex<float>>) {
        complex<float> alpha(1,0);
        complex<float> beta(0,0);
        status = mkl_sparse_c_mm(op,alpha,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),col,col,beta,R.begin(),col);
    }else if constexpr (is_same_v<T,complex<double>>) {
        complex<double> alpha(1,0);
        complex<double> beta(0,0);
        status = mkl_sparse_z_mm(op,alpha,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),col,col,beta,R.begin(),col);
    }
    check_sparse_operation_status(status);
    return R;
}
template<typename T>
inline Matrix<T,2> operator *(const Matrix<T,2> &dm,const Matrix<T,2,MATRIX_TYPE::CSR> &sm){
    assert(dm.cols() == sm.rows());

    int row = dm.rows();
    int col = sm.cols();

    Matrix<T,2> R(row,col);

    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    sparse_operation_t op = SPARSE_OPERATION_TRANSPOSE;
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_layout_t sparseLayout = SPARSE_LAYOUT_COLUMN_MAJOR;
    if constexpr (is_same_v<T,float>){
        status = mkl_sparse_s_mm(op,1,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),row,dm.cols(),0,R.begin(),dm.cols());
    }else if constexpr (is_same_v<T,double>) {
        status = mkl_sparse_d_mm(op,1,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),row,dm.cols(),0,R.begin(),dm.cols());

    }else if constexpr (is_same_v<T,complex<float>>) {
        complex<float> alpha(1,0);
        complex<float> beta(0,0);
        status = mkl_sparse_c_mm(op,alpha,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),row,dm.cols(),beta,R.begin(),dm.cols());
    }else if constexpr (is_same_v<T,complex<double>>) {
        complex<double> alpha(1,0);
        complex<double> beta(0,0);
        status = mkl_sparse_z_mm(op,alpha,sm.sparse_matrix_handler(),descr,sparseLayout,dm.begin(),row,dm.cols(),beta,R.begin(),dm.cols());
    }
    check_sparse_operation_status(status);
    return R;
}
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR> operator *(const Matrix<T,2,MATRIX_TYPE::CSR> &sm1,const Matrix<T,2,MATRIX_TYPE::CSR> &sm2){
    sparse_matrix_t handlerC; //result sparse matrix handler;
    sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
    sparse_status_t status1 = SPARSE_STATUS_NOT_SUPPORTED;

    sparse_index_base_t index;
    int rows;
    int cols;
    int *rowStart;
    int *rowEnd;
    int *columns;
    T *values;

    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE,sm1.sparse_matrix_handler(),sm2.sparse_matrix_handler(),&handlerC);
    check_sparse_operation_status(status); //checking for operation success
    status = mkl_sparse_order(handlerC);
    check_sparse_operation_status(status);

    if constexpr (is_same_v<T,float>){
        status1 = mkl_sparse_s_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }else if constexpr (is_same_v<T,double>) {
        status1 = mkl_sparse_d_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }else if  constexpr (is_same_v<T,complex<float>>){
        status1 = mkl_sparse_c_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }else if constexpr (is_same_v<T,complex<double>>){
        status1 = mkl_sparse_z_export_csr(handlerC,&index,&rows,&cols,&rowStart,&rowEnd,&columns,&values);
    }

    check_sparse_operation_status(status1);
    int nnz = rowEnd[rows-1]-rowStart[0];

    vector<int_t> rows_start(rowStart,rowStart+rows);
    vector<int_t> rows_end(rowEnd,rowEnd+rows);
    vector<int_t> columns_index(columns,columns+nnz);
    vector<T> vals(values,values+nnz);

    mkl_sparse_destroy(handlerC); //destroy handler

    return Matrix<T,2,MATRIX_TYPE::CSR>(rows,cols,move(rows_start),move(rows_end),move(columns_index),move(vals));
}



#endif // MATRIX_ARITHMETIC_OPS_H
