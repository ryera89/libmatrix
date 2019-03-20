#ifndef MATRIX_H
#define MATRIX_H

#include <valarray>
#include <vector>
#include <complex>
#include <type_traits>
#include "matrix_ref.h"
#include <iostream>
#include <iomanip>
#include <map>

//We must define MKL_Complex16 type before include the mkl.h header
//#define MKL_Complex16 std::complex<double>
//#define MKL_INT uint32_t
#include "mkl.h"



using std::vector;

enum class MATRIX_TYPE{GEN,SYMM,HER,UTRI,LTRI,DIAG,CSR,CSR3};

//using MATRIX_TYPE::SYMM=SYMM;

template<typename T,size_t N,MATRIX_TYPE type = MATRIX_TYPE::GEN,
         typename = typename std::enable_if_t<((std::is_arithmetic_v<T> || std::is_same_v<T,std::complex<double>>
                                              || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long int>>)
                                              && N > 2 && type == MATRIX_TYPE::GEN)
                                              || (std::is_arithmetic_v<T> && (type != MATRIX_TYPE::HER) && N == 2)
                                              || ((std::is_same_v<T,std::complex<double>>
                                              || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long int>>)
                                              && N == 2) || (N == 1 && (type == MATRIX_TYPE::GEN)) >>
class Matrix{
private:
    Matrix_Slice<N> m_desc;
    std::valarray<T> m_elems;
public:
    static constexpr size_t order = N;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::GEN;
    using value_type = T;

    //Default constructor and destructor
    Matrix() = default;
    ~Matrix() = default;
    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    //Initialization from extents
    template<typename... Exts,typename = typename std::enable_if_t<sizeof... (Exts) == N>>
    Matrix(Exts... exts):m_desc(0,{exts...}),m_elems(m_desc.m_size){}

//    Matrix(const Matrix_Ref<T,N> &ref){
//        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
//        m_desc.m_start = 0;
//        m_desc.m_extents = ref.descriptor().m_extents;
//        m_desc.init();
//        m_elems.resize(m_desc.m_size);
//        Matrix_Ref<T,N> mref{begin(),m_desc};
//        assing_slice_vals(ref,mref);
//    }
//    Matrix& operator = (const Matrix_Ref<T,N> &ref){
//        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
//        m_desc.m_start = 0;
//        m_desc.m_extents = ref.descriptor().m_extents;
//        m_desc.init();
//        m_elems.resize(m_desc.m_size);
//        Matrix_Ref<T,N> mref{begin(),m_desc};
//        assing_slice_vals(ref,mref);
//        return *this;
//    }
    //Construction and assignment from nested initializars
    Matrix(Matrix_Initializer<T,N> init){
        derive_extents(m_desc._extents,init);
        m_desc.init();    //Strides determination from extents
        m_elems.resize(m_desc.m_size);
        insert_flat(init,m_elems);
        assert(m_elems.size() == m_desc.m_size);
    }
    Matrix& operator=(Matrix_Initializer<T,N> init){
        derive_extents(m_desc.m_extents,init);
        m_elems.resize(m_desc.m_size);
        insert_flat(init,m_elems);
        assert(m_elems.size() == m_desc.m_size);
        return *this;
    }
    Matrix& operator=(const T &val){
        m_elems = val;
        return *this;
    }
    //Disable use of {} for extents
    template<typename U>
    Matrix(std::initializer_list<U>) = delete;
    template<typename U>
    Matrix& operator=(std::initializer_list<U>) = delete;

    const Matrix_Slice<N>& descriptor() const noexcept{
        return m_desc;
    }
    size_t size() const noexcept {return m_desc.m_size;}

    size_t extent(size_t n) const{
        assert(n < N);
        return m_desc.m_extents[n];
    }

    //std::valarray<T>& values() {return m_elems;}
    const std::valarray<T>& values() const{return m_elems;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    //Matrix apply(T (func)(T)){Matrix r(m_desc.m_extents); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(T)) const{Matrix r(*this); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    Matrix_Ref<T,N-1> row(size_t i){
        Matrix_Slice<N-1> row;
        slice_dim<0>(i,m_desc,row);
        return {begin(),row};
    }
    Matrix_Ref<const T,N-1> row(size_t i) const{
        Matrix_Slice<N-1> row;
        slice_dim<0>(i,m_desc,row);
        return {begin(),row};
    }
    Matrix_Ref<T,N-1> operator[](size_t i){
        return row(i);
    }
    Matrix_Ref<const T,N-1>
    operator[](size_t i) const{
        return row(i);
    }
    template<typename... Args>
    std::enable_if_t<requesting_element<Args...>(), T&>
    operator()(Args... args){
        assert(check_bounds(m_desc,args...));
        return m_elems[m_desc(args...)];
    }
    template<typename... Args>
    std::enable_if_t<requesting_element<Args...>(),const T&>
    operator()(Args... args) const{
        assert(check_bounds(m_desc,args...));
        return m_elems[m_desc(args...)];
    }
    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<T,N>>
    operator()(Args... args){
        Matrix_Slice<N> d;
        d.m_size = 1;
        d.m_start = do_slice(m_desc,d,args...);
        return {begin(),d};
    }
    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<const T,N>>
    operator()(Args... args) const{
        Matrix_Slice<N> d;
        d.m_size = 1;
        do_slice(m_desc,d,args...);
        return {begin(),d};
    }

    //Arithmetic
    //Unary minus
    Matrix operator-() const
    {
        Matrix res(m_desc.m_extents);
        std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        return res;
    }
    //Arithmetic operations.
    template<typename Scalar>
    Matrix& operator+=(const Scalar& val)
    {
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar& val)
    {
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar& val)
    {
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar& val)
    {
        m_elems/=val;
        return *this;
    }
    Matrix& operator+=(const Matrix<T,N> &m)
    {
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          m.descriptor().m_extents.begin(),m.descriptor().m_extents.end()));
        m_elems+=m.values();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,N> &m)
    {
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          m.descriptor().m_extents.begin(),m.descriptor().m_extents.end()));
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator+=(const Matrix<U,N> &m)
    {
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          m.descriptor().m_extents.begin(),m.descriptor().m_extents.end()));
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1+val2;});
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator-=(const Matrix<U,N> &m)
    {
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          m.descriptor().m_extents.begin(),m.descriptor().m_extents.end()));
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1-val2;});
        return *this;
    }
};
template<typename T>
class Matrix<T,2,MATRIX_TYPE::GEN>{
private:
    Matrix_Slice<2> m_desc;
    std::valarray<T> m_elems;
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::GEN;
    using value_type = T;

    Matrix() = default;
    ~Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(size_t m,size_t n):m_desc(0,m,n),m_elems(m*n){}
    Matrix(size_t m,size_t n,T val):m_desc(0,m,n),m_elems(val,m*n){}
    //Construction and assignment from nested initializars
    Matrix(Matrix_Initializer<T,2> init){
        derive_extents(m_desc.m_extents,init);
        m_desc.init();    //Strides determination from extents
        m_elems.resize(m_desc.m_size);
        insert_flat(init,m_elems);
        assert(m_elems.size() == m_desc.m_size);
    }
    Matrix& operator=(Matrix_Initializer<T,2> init){
        derive_extents(m_desc.m_extents,init);
        m_elems.resize(m_desc.m_size);
        insert_flat(init,m_elems);
        assert(m_elems.size() == m_desc.m_size);
        return *this;
    }
    Matrix(const Matrix_Ref<T,2> &ref):m_desc(0,ref.descriptor().m_extents[0],ref.descriptor().m_extents[1]),
                                          m_elems(m_desc.m_extents[0]*m_desc.m_extents[1])
    {
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = 0; j < cols(); ++j){
                m_elems[i*cols() + j] = ref(i,j);
            }
        }
    }
    Matrix& operator=(const Matrix_Ref<T,2> &ref){
        m_desc.m_start = 0;
        m_desc.m_extents = ref.descriptor().m_extents;
        m_desc.init();
        m_elems.resize(m_desc.m_size);
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = 0; j < cols(); ++j){
                m_elems[i*cols() + j] = ref(i,j);
            }
        }
    }
    //Disable use of {} for extents
    template<typename U>
    Matrix(std::initializer_list<U>) = delete;
    template<typename U>
    Matrix& operator=(std::initializer_list<U>) = delete;

    Matrix& operator =(T val){m_elems = val; return *this;}

    const Matrix_Slice<2>& descriptor() const noexcept{
        return m_desc;
    }
    size_t size() const noexcept {return m_desc.m_size;}

    size_t extent(size_t n) const{
        assert(n < 2);
        return m_desc.m_extents[n];
    }

    size_t rows() const{return m_desc.m_extents[0];}
    size_t cols() const{return m_desc.m_extents[1];}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    //std::valarray<T>& values() {return m_elems;}
    const std::valarray<T>& values() const{return m_elems;}

    Matrix apply(T (func)(T val)){Matrix r(rows(),cols()); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(rows(),cols()); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    Matrix_Ref<T,1> row(size_t i){
        assert(i < rows());
        Matrix_Slice<1> row;
        slice_dim<0>(i,m_desc,row);
        return {begin(),row};
    }
    Matrix_Ref<const T,1> row(size_t i) const{
        assert(i < rows());
        Matrix_Slice<1> row;
        slice_dim<0>(i,m_desc,row);
        return {begin(),row};
    }
    Matrix_Ref<T,1> column(size_t i){
        assert(i < cols());
        Matrix_Slice<1> col;
        slice_dim<1>(i,m_desc,col);
        return {begin(),col};
    }
    Matrix_Ref<const T,1> column(size_t i) const{
        assert(i < cols());
        Matrix_Slice<1> col;
        slice_dim<1>(i,m_desc,col);
        return {begin(),col};
    }
    Matrix_Ref<T,1> operator[](size_t i){
        return row(i);
    }
    Matrix_Ref<const T,1> operator[](size_t i) const{
        return row(i);
    }

    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<T,2>>
    operator()(Args... args){
         Matrix_Slice<2> d;
         d.m_size = 1;
         d.m_start = do_slice(m_desc,d,args...);
         return {begin(),d};
    }
    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<const T,2>>
    operator()(Args... args) const{
         Matrix_Slice<2> d;
         d.m_size = 1;
         do_slice(m_desc,d,args...);
         return {begin(),d};
    }
    T& operator()(size_t i,size_t j){
        assert(i < rows() && j < cols());
        return m_elems[i*cols() + j];
    }
    const T& operator()(size_t i,size_t j) const{
        assert(i < rows() && j < cols());
        return m_elems[i*cols() + j];
    }
    Matrix<T,2> operator()(const std::valarray<uint32_t> &indx1,const std::valarray<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    Matrix<T,2> operator()(const vector<uint32_t> &indx1,const vector<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    //unary minus
    Matrix operator-() const{
        Matrix r(rows(),cols());
        std::transform(this->begin(),this->end(),r.begin(),[](const auto &val){return -val;});
        return r;
    }
    template<typename Scalar>
    Matrix& operator+=(const Scalar &val){
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar &val){
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar &val){
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar &val){
        m_elems/=val;
        return *this;
    }
    Matrix& operator+=(const Matrix<T,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems+=m.values();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator+=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1+val2;});
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator-=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1-val2;});
        return *this;
    }
};

template<typename T>
class Matrix<T,2,MATRIX_TYPE::SYMM>{
private:
    size_t m_dim;
    std::valarray<T> m_elems;
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::SYMM;
    using value_type = T;

    Matrix() = default;
    ~Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(size_t m):m_dim(m),m_elems(0.5*m*(m+1)){}
    Matrix(size_t m,T val):m_dim(m),m_elems(val,0.5*m*(m+1)){}
    Matrix(const Matrix<T,2> &other):m_dim(other.rows()),m_elems(0.5*m_dim*(m_dim+1)){
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) = other(i,j);
            }
        }
    }
    Matrix& operator=(const Matrix<T,2> &other){
        m_dim = other.rows();
        m_elems.resize(0.5*m_dim*(m_dim+1));
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) = other(i,j);
            }
        }
        return *this;
    }
    Matrix& operator =(T val){m_elems = val; return *this;}

    size_t rows() const{return m_dim;}
    size_t cols() const{return m_dim;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    //std::valarray<T>& values() {return m_elems;}
    const std::valarray<T>& values() const{return m_elems;}

    Matrix apply(T (func)(T val)){Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    T& operator()(size_t i,size_t j){
        assert(i < rows() && j < cols());
        if (i > j) std::swap(i,j); //upper triangle storage
        return m_elems[j + 0.5*i*(2*m_dim - i - 1)];
    }
    const T& operator()(size_t i,size_t j) const{
        assert(i < rows() && j < cols());
        if (i > j) std::swap(i,j); //upper triangle storage
        return m_elems[j + 0.5*i*(2*m_dim - i - 1)];
    }
    Matrix<T,2> operator()(const std::valarray<uint32_t> &indx1,const std::valarray<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    Matrix<T,2> operator()(const vector<uint32_t> &indx1,const vector<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    Matrix operator-() const
    {
        Matrix res(rows(),cols());
        std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        return res;
    }
    //Arithmetic operations.
    template<typename Scalar>
    Matrix& operator+=(const Scalar& val)
    {
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar& val)
    {
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar& val)
    {
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar& val)
    {
        //TODO valorar no permitir division por cero
        m_elems/=val;
        return *this;
    }
    Matrix& operator+=(const Matrix<T,2,matrix_type> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems+=m.values();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,2,matrix_type> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator+=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1+val2;});
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator-=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1-val2;});
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols()); //symmetric matrizes are squared
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) += m(i,j);
            }
        }
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols()); //symmetric matrizes are squared
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) -= m(i,j);
            }
        }
        return *this;
    }
};

template<typename T>
class Matrix<T,2,MATRIX_TYPE::HER>{
private:
    size_t m_dim;
    std::valarray<T> m_elems;
    mutable T m_aux;
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::HER;
    using value_type = T;

    Matrix() = default;
    ~Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(size_t m):m_dim(m),m_elems(0.5*m*(m+1)){}
    Matrix(size_t m,T val):m_dim(m),m_elems(val,0.5*m*(m+1)){}
    Matrix(const Matrix<T,2> &other):m_dim(other.rows()),m_elems(0.5*m_dim*(m_dim+1)){
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) = other(i,j);
            }
        }
    }
    Matrix& operator=(const Matrix<T,2> &other){
        m_dim = other.rows();
        m_elems.resize(0.5*m_dim*(m_dim+1));
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) = other(i,j);
            }
        }
        return *this;
    }
    Matrix& operator =(T val){m_elems = val; return *this;}

    size_t rows() const{return m_dim;}
    size_t cols() const{return m_dim;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    //std::valarray<T>& values() {return m_elems;}
    const std::valarray<T>& values() const{return m_elems;}

    Matrix apply(T (func)(T val)){Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    T& operator()(size_t i,size_t j){
        assert(i < rows() && j < cols() && i <= j);
        //if (i > j) std::swap(i,j); //upper triangle storage
        return m_elems[j + 0.5*i*(2*m_dim - i - 1)];
    }
    const T& operator()(size_t i,size_t j) const{
        assert(i < rows() && j < cols());
        if (i > j){
            std::swap(i,j);
            m_aux = std::conj(m_elems[j + 0.5*i*(2*m_dim - i - 1)]);
            return m_aux;
        } //upper triangle storage
        return m_elems[j + 0.5*i*(2*m_dim - i - 1)];
    }
    Matrix<T,2> operator()(const std::valarray<uint32_t> &indx1,const std::valarray<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    Matrix<T,2> operator()(const vector<uint32_t> &indx1,const vector<uint32_t> &indx2) const{
        size_t nrow = indx1.size();
        size_t ncol = indx2.size();

        Matrix<T,2> res(nrow,ncol);
        for (size_t i = 0; i < nrow; ++i){
            for (size_t j = 0; j < ncol; ++j){
                res(i,j) = this->operator()(indx1[i],indx2[j]);
            }
        }
        return res;
    }
    Matrix operator-() const
    {
        Matrix res(rows(),cols());
        std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        return res;
    }
    //Arithmetic operations.
    template<typename Scalar>
    Matrix& operator+=(const Scalar& val)
    {
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar& val)
    {
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar& val)
    {
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar& val)
    {
        //TODO valorar no permitir division por cero
        m_elems/=val;
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        m_elems+=m.values();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,2,MATRIX_TYPE::SYMM> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),[](const auto &v1,const auto &v2){return v1+v2;});
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2,MATRIX_TYPE::SYMM> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),[](const auto &v1,const auto &v2){return v1-v2;});
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols()); //symmetric matrizes are squared
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) += m(i,j);
            }
        }
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols()); //symmetric matrizes are squared
        for (size_t i = 0; i < rows(); ++i){
            for (size_t j = i; j < cols(); ++j){
                this->operator()(i,j) -= m(i,j);
            }
        }
        return *this;
    }
};

typedef int int_t;
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR>{
private:
    uint32_t m_rows;
    uint32_t m_cols;
    vector<int_t> m_rows_start;
    vector<int_t> m_rows_end;
    vector<int_t> m_columns;
    vector<T> m_elems;
    static constexpr T zero_val = T();
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR;
    using value_type = T;
    using iterator = typename vector<T>::iterator;
    using const_iterator = typename vector<T>::const_iterator;

    Matrix() = default;
    ~Matrix() = default;
    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;


    Matrix(uint32_t rows,uint32_t cols,const vector<int_t> &rows_start,const vector<int_t> &rows_end,
           const vector<int_t> &columns,const vector<T> &vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),
                                                                             m_rows_end(rows_end),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
        m_elems.shrink_to_fit();
        m_columns.shrink_to_fit();
        m_rows_start.shrink_to_fit();
        m_rows_end.shrink_to_fit();
    }

    Matrix(uint32_t rows,uint32_t cols,vector<int_t> &&rows_start,vector<int_t> &&rows_end,vector<int_t> &&columns,
           vector<T> &&vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),m_rows_end(rows_end),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
        m_elems.shrink_to_fit();
        m_columns.shrink_to_fit();
        m_rows_start.shrink_to_fit();
        m_rows_end.shrink_to_fit();
    }
    Matrix(const Matrix<T,2> &m):m_rows(m.rows()),m_cols(m.cols()),m_rows_start(m_rows),m_rows_end(m_rows){
        for (size_t i = 0; i < m_rows; ++i){
            bool row_first_nonzero = true;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (row_first_nonzero){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        row_first_nonzero = false;
                    }
                }
            }
            if (row_first_nonzero){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}
        }
    }

    size_t nnz() const{return m_elems.size();}

    Matrix& operator=(const Matrix<T,2> &m){
        m_rows = m.rows();
        m_cols = m.cols();
        m_rows_start.resize(m_rows);
        m_rows_end.resize(m_rows);
        m_columns.clear();
        m_elems.clear();
        for (size_t i = 0; i < m_rows; ++i){
            bool row_first_nonzero = true;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (row_first_nonzero){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        row_first_nonzero = false;
                    }
                }
            }
            if (row_first_nonzero){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}
        }
        return *this;
    }
    const T& operator()(size_t i,size_t j) const{
        assert(i < m_rows && j < m_cols);
        int_t beg = m_rows_start[i];
        int_t end = m_rows_end[i];

        if (beg == end) return zero_val; //row i is full of 0
        if (j < m_columns[beg] || j > m_columns[end-1]) return zero_val;

        std::pair<vector<int_t>::const_iterator,vector<int_t>::const_iterator> ip;
        ip = std::equal_range(m_columns.cbegin()+beg,m_columns.cbegin()+end,j);
        int tmp = std::distance(ip.first,ip.second);
        if (tmp > 0){
            size_t pos = ip.first - m_columns.begin();
            return m_elems[pos];
        }
        return zero_val;
    }
    Matrix operator()(const vector<uint32_t> &iindex,const vector<uint32_t> &jindex) const{
        uint32_t nrow = iindex.size();
        uint32_t ncol = jindex.size();

        vector<T> elems;
        elems.reserve(m_elems.size());
        vector<int_t> columns;
        columns.reserve(m_elems.size());
        vector<int_t> pointerB(nrow);
        vector<int_t> pointerE(nrow);


        for (uint32_t i = 0; i < nrow; ++i){
            bool rfirst_inclusion =  true;
            uint32_t ii = iindex[i];
            int_t ibeg = m_rows_start[ii];
            int_t iend = m_rows_end[ii];
            if (ibeg == iend){ //fila todos zeros
                pointerB[i] = elems.size(); pointerE[i] = elems.size();
                continue;
            }
            int_t icolb = m_columns[ibeg];
            int_t icole = m_columns[iend-1];
            for (int_t j = 0; j < ncol; ++j){
                int_t jj = jindex[j];
                if (jj < icolb || jj > icole) continue; //A(ii,jj) = 0; no esta entre los valores
                //std::pair<vector<int_t>::const_iterator,vector<int_t>::const_iterator> ip;
                //ip = std::equal_range(m_columns.cbegin()+ibeg,m_columns.cbegin()+iend,jj);
                //vector<int_t>::const_iterator it = std::lower_bound(m_columns.cbegin()+ibeg,m_columns.cbegin()+iend,jj);
                //int tmp = std::distance(ip.first,ip.second);
                vector<int_t>::const_iterator bit = m_columns.cbegin()+ibeg;
                vector<int_t>::const_iterator eit = m_columns.cbegin()+iend;
                vector<int_t>::const_iterator it;
                int64_t count = std::distance(bit,eit);
                int64_t step;
                while(count > 0){
                    it = bit;
                    step = count/2;
                    std::advance(it,step);
                    if (*it == jj){ //value found
                        int64_t pos = std::distance(m_columns.begin(),it);
                        T val = m_elems[pos];
                        elems.push_back(val);
                        columns.push_back(j);
                        if (rfirst_inclusion){ //se ejecuta maximo solo una ves del loop principal i
                            pointerB[i] = elems.size()-1;
                            rfirst_inclusion = false;
                        }
                        break;
                    }
                    if (*it < jj){
                        bit = ++it;
                        count -= step+1;
                    }else{
                        count = step;
                    }
                }
//                if (*it == jj){
//                    //size_t pos = ip.first - m_columns.begin();
//                    size_t pos = it - m_columns.begin();
//                    T val = m_elems[pos];
//                    elems.push_back(val);
//                    columns.push_back(j);
//                    if (rfirst_inclusion){ //se ejecuta maximo solo una ves del loop principal i
//                        pointerB[i] = elems.size()-1;
//                        rfirst_inclusion = false;
//                    }
//                }
            }
            if (rfirst_inclusion){ //una fila llena de zeros
                pointerB[i] = elems.size(); pointerE[i] = elems.size();
            }else{ pointerE[i] = elems.size();}
        }
        return Matrix(nrow,ncol,pointerB,pointerE,columns,elems);
    }

    uint32_t rows() const{return m_rows;}
    uint32_t cols() const{return m_cols;}

    void printData() const{
        for (size_t i = 0; i < m_rows_start.size(); ++i){
            uint32_t beg = m_rows_start[i];
            uint32_t end = m_rows_end[i];
            for (;beg<end;++beg){
                std::cout << "(" << i << "," << m_columns[beg] << ") " << m_elems[beg] << "\n";
            }
        }
    }

    const vector<T>& values() const{return m_elems;}
    const vector<int_t>& columns() const{return m_columns;}
    const vector<int_t>& row_start() const{return m_rows_start;}
    const vector<int_t>& row_end() const{return m_rows_end;}

    vector<int_t>::iterator beginColumns(){return m_columns.begin();}
    vector<int_t>::const_iterator beginColumns() const {return m_columns.cbegin();}
    vector<int_t>::iterator endColumns(){return m_columns.end();}
    vector<int_t>::const_iterator endColumns() const {return m_columns.cend();}

    vector<int_t>::iterator beginRowsStart(){return m_rows_start.begin();}
    vector<int_t>::const_iterator beginRowsStart() const{return m_rows_start.cbegin();}
    vector<int_t>::iterator endRowsStart(){return m_rows_start.end();}
    vector<int_t>::const_iterator endRowsStart() const{return m_rows_start.cend();}

    vector<int_t>::iterator beginRowsEnd(){return m_rows_end.begin();}
    vector<int_t>::const_iterator beginRowsEnd() const{return m_rows_end.cbegin();}
    vector<int_t>::iterator endRowsEnd(){return m_rows_end.end();}
    vector<int_t>::const_iterator endRowsEnd() const{return m_rows_end.cend();}

    iterator beginValues(){return m_elems.begin();}
    const_iterator beginValues() const{return m_elems.cbegin();}
    iterator endValues(){return m_elems.end();}
    const_iterator endValues() const{return m_elems.cend();}

    int_t* columnsData(){return m_columns.data();}
    const int_t* columnsData() const {return m_columns.data();}

    int_t* rowsStartData(){return m_rows_start.data();}
    const int_t* rowsStartData() const{return m_rows_start.data();}

    int_t* rowsEndData(){return m_rows_end.data();}
    const int_t* rowsEndData() const{return m_rows_end.data();}

    T* valuesData(){return m_elems.data();}
    const T* valuesData() const{return m_elems.data();}
};
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR3>{
private:
    uint32_t m_rows;
    uint32_t m_cols;
    vector<int_t> m_rowIndex;
    vector<int_t> m_columns;
    vector<T> m_elems;
    static constexpr T zero_val = T();
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR3;
    using value_type = T;
    using iterator = typename vector<T>::iterator;
    using const_iterator = typename vector<T>::const_iterator;

    Matrix() = default;
    ~Matrix() = default;
    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;


    Matrix(uint32_t rows,uint32_t cols,const vector<int_t> &rowIndex,const vector<int_t> &columns,const vector<T> &vals):
                                                                     m_rows(rows),m_cols(cols),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_rows == (m_rowIndex.size()-1));

    }
    Matrix(uint32_t rows,uint32_t cols,vector<int_t> &&rowIndex,vector<int_t> &&columns,vector<T> &&vals):
                                                                     m_rows(rows),m_cols(cols),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_rows == (m_rowIndex.size()-1));
    }
    Matrix(const Matrix<T,2> &m):m_rows(m.rows()),m_cols(m.cols()),m_rowIndex(m_rows+1){
        for (uint32_t i = 0; i < m_rows; ++i){
            bool first_row_inclusion = true;
            for (uint32_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (first_row_inclusion){ //se ejecuta maximo solo una ves del loop principal i
                        m_rowIndex[i] = m_elems.size()-1;
                        first_row_inclusion = false;
                    }
                }
            }
            if (first_row_inclusion){ //una fila llena de zeros
                if (m_elems.size() == 0) {m_rowIndex[i] = 0;} /*primeras filas == 0*/
                else {m_rowIndex[i] = m_elems.size();}
            }
        }
        m_rowIndex[m_rows] = m_elems.size();
    }
    Matrix& operator=(const Matrix<T,2> &m){
        m_rows = m.rows();
        m_cols = m.cols();
        m_rowIndex.resize(m_rows+1);
        m_columns.clear();
        m_elems.clear();
        for (uint32_t i = 0; i < m_rows; ++i){
            bool first_row_inclusion = true;
            for (uint32_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (first_row_inclusion){ //se ejecuta maximo solo una ves del loop principal i
                        m_rowIndex[i] = m_elems.size()-1;
                        first_row_inclusion = false;
                    }
                }
            }
            if (first_row_inclusion){ //una fila llena de zeros
                if (m_elems.size() == 0) {m_rowIndex[i] = 0;} /*primeras filas == 0*/
                else {m_rowIndex[i] = m_elems.size();}
            }
        }
        m_rowIndex[m_rows] = m_elems.size();
        return *this;
    }
    const T& operator()(uint32_t i,uint32_t j) const{
        assert(i < m_rows && j < m_cols);
        int_t beg = m_rowIndex[i];
        int_t end = m_rowIndex[i+1];

        if (beg == end) return zero_val; //i row full of zeros
        if (j < m_columns[beg] || j > m_columns[end-1]) return zero_val;

        std::pair<vector<int_t>::const_iterator,vector<int_t>::const_iterator> ip;
        ip = std::equal_range(m_columns.cbegin()+beg,m_columns.cbegin()+end,j);
        int tmp = std::distance(ip.first,ip.second);
        if (tmp > 0){
            size_t pos = ip.first - m_columns.begin();
            return m_elems[pos];
        }
        return zero_val;
    }
    Matrix operator()(const vector<uint32_t> &iindex,const vector<uint32_t> &jindex) const{
        uint32_t nrow = iindex.size();
        uint32_t ncol = jindex.size();

        vector<T> elems;
        elems.reserve(m_elems.size());
        vector<int_t> columns;
        columns.reserve(m_elems.size());
        vector<int_t> rowIndex(nrow+1);

        for (uint32_t i = 0; i < nrow; ++i){
            bool rfirst_inclusion =  true;
            uint32_t ii = iindex[i];
            int_t ibeg = m_rowIndex[ii];
            int_t iend = m_rowIndex[ii+1];
            if (ibeg == iend){ //fila todos zeros
                rowIndex[i] = elems.size(); rowIndex[i+1] = elems.size();
                continue;
            }
            int_t icolb = m_columns[ibeg];
            int_t icole = m_columns[iend-1];
            for (int_t j = 0; j < ncol; ++j){
                uint32_t jj = jindex[j];
                if (jj < icolb || jj > icole) continue; //A(ii,jj) = 0; no esta entre los valores
                std::pair<vector<int_t>::const_iterator,vector<int_t>::const_iterator> ip;
                ip = std::equal_range(m_columns.cbegin()+ibeg,m_columns.cbegin()+iend,jj);
                int tmp = std::distance(ip.first,ip.second);
                if (tmp > 0){
                    size_t pos = ip.first - m_columns.begin();
                    T val = m_elems[pos];
                    elems.push_back(val);
                    columns.push_back(j);
                    if (rfirst_inclusion){ //se ejecuta maximo solo una ves del loop principal i
                        rowIndex[i] = elems.size()-1;
                        rfirst_inclusion = false;
                    }
                }
            }
            if (rfirst_inclusion){ //una fila llena de zeros
                rowIndex[i] = elems.size(); rowIndex[i+1] = elems.size();
            }else{ rowIndex[i+1] = elems.size();}
        }
        return Matrix(nrow,ncol,rowIndex,columns,elems);
    }

    void printData() const{
        for (size_t i = 0; i < m_rows; ++i){
            uint32_t beg = m_rowIndex[i];
            uint32_t end = m_rowIndex[i+1];
            for (;beg<end;++beg){
                std::cout << "(" << i << "," << m_columns[beg] << ") " << m_elems[beg] << "\n";
            }
        }
    }

    size_t rows() const{return m_rows;}
    size_t cols() const{return m_cols;}

    const vector<T>& values() const{return m_elems;}
    const vector<int_t>& columns() const{return m_columns;}
    const vector<int_t>& rowIndex() const{return m_rowIndex;}

    vector<int_t>::iterator beginColumns(){return m_columns.begin();}
    vector<int_t>::const_iterator beginColumns() const {return m_columns.cbegin();}
    vector<int_t>::iterator endColumns(){return m_columns.end();}
    vector<int_t>::const_iterator endColumns() const {return m_columns.cend();}

    vector<int_t>::iterator beginRowsStart(){return m_rowIndex.begin();}
    vector<int_t>::const_iterator beginRowsStart() const{return m_rowIndex.cbegin();}
    vector<int_t>::iterator endRowsStart(){return m_rowIndex.end();}
    vector<int_t>::const_iterator endRowsStart() const{return m_rowIndex.cend();}

    iterator beginValues(){return m_elems.begin();}
    const_iterator beginValues() const{return m_elems.cbegin();}
    iterator endValues(){return m_elems.end();}
    const_iterator endValues() const{return m_elems.cend();}

    int_t* columnsData(){return m_columns.data();}
    const int_t* columnsData() const {return m_columns.data();}

    int_t* rowsStartData(){return m_rowIndex.data();}
    const int_t* rowsStartData() const{return m_rowIndex.data();}

    T* valuesData(){return m_elems.data();}
    const T* valuesData() const{return m_elems.data();}
};

template<typename T>
class Matrix<T,1>{
    Matrix_Slice<1> m_desc;
    std::valarray<T> m_elems;
public:
    static constexpr size_t order = 1;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::GEN;
    using value_type = T;

    //Default constructor and destructor
    Matrix() = default;
    ~Matrix() = default;

    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;

    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;

    Matrix(size_t ext):m_desc(0,ext),m_elems(ext){}
    Matrix(const std::array<size_t,1> &exts):m_desc{0,exts},m_elems(m_desc.m_size){}

    //TODO ver que hacer aqui
    Matrix(const Matrix_Ref<T,1> &ref):m_desc(0,ref.descriptor().m_extents[0]),
                                                       m_elems(m_desc.m_size)
    {
        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        for (size_t i = 0; i < m_desc.m_extents[0]; ++i) m_elems[i] = ref(i);
    }
    Matrix& operator = (const Matrix_Ref<T,1> &ref){
        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        m_desc.m_start = 0;
        m_desc.m_extents = ref.descriptor().m_extents;
        m_desc.init();
        m_elems.resize(m_desc.m_size);
        for (size_t i = 0; i < m_desc.m_extents[0]; ++i) m_elems[i] = ref(i);
        return *this;
    }
    Matrix& operator=(const T &val){
        std::for_each(begin(),end(),[&val](T &elem){elem = val;});
        return *this;
    }
    Matrix(std::initializer_list<T> list):m_elems(list){
        m_desc.m_start = 0;
        m_desc.m_size = m_elems.size();
        m_desc.m_extents[0] = m_elems.size();
        m_desc.m_strides[0] = 1;
    }
    Matrix& operator=(std::initializer_list<T> list){
        m_elems = list;
        m_desc.m_start = 0;
        m_desc.m_size = m_elems.size();
        m_desc.m_extents[0] = m_elems.size();
        m_desc.m_strides[0] = 1;
        return *this;
    }

    size_t size() const noexcept{return m_elems.size();}

    //std::valarray<T>& values() {return m_elems;}
    const std::valarray<T>& values() const{return m_elems;}

    const Matrix_Slice<1>& descriptor() const noexcept{
        return m_desc;
    }

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}

    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    //Arithmetic ops
    template<typename F>
    Matrix& apply(F fun){
        std::for_each(begin(),end(),fun);
        return *this;
    }
    Matrix apply(T (func)(T val)){Matrix r(size()); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(size()); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    //Unary minus
    Matrix operator-() const{
        Matrix res(this->size());
        std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        return res;
    }

    //Arithmetic operations.
    template<typename Scalar>
    Matrix& operator+=(const Scalar& val)
    {
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar& val)
    {
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar& val)
    {
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar& val)
    {
        m_elems/=val;
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,1> &m)
    {
        assert(this->size() == m.size());
        m_elems+=m.values();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,1> &m)
    {
        assert(this->size() == m.size());
        m_elems-=m.values();
        return *this;
    }

    //access functions
    T& operator()(const size_t &i)
    {
        assert(i < size());
        return m_elems[i];
    }
    const T& operator()(const size_t &i) const
    {
        assert(i < size());
        return m_elems[i];
    }
    Matrix_Ref<T,1> operator()(const std::slice &s){
        Matrix_Slice<1> d;
        d.m_start = s.start();
        d.m_size = s.stride();
        d.m_extents[0] = s.size();
        d.m_strides[0] = s.stride();
        //d._start = matrix_impl::do_slice(_desc,d,args...);
        return {begin(),d};
    }

    Matrix_Ref<T,1> operator()(const std::slice &s) const{
        Matrix_Slice<1> d;
        d.m_start = s.start();
        d.m_size = s.stride();
        d.m_extents[0] = s.size();
        d.m_strides[0] = s.stride();
        //d._start = matrix_impl::do_slice(_desc,d,args...);
        return {begin(),d};
    }
    std::enable_if_t<std::is_arithmetic_v<T>,T>
    norm() const{
        T res = std::inner_product(begin(),end(),begin(),0);
        return std::sqrt(res);
    }
    std::enable_if_t<std::is_arithmetic_v<T>,T>
    sqr_norm() const{
        return std::inner_product(begin(),end(),begin(),0);
    }
};

//************ostream operations*********************************************************************
template<typename T,MATRIX_TYPE mtype>
inline  std::ostream &operator << (std::ostream &os,const Matrix<T,2,mtype> &m){
    std::ios_base::fmtflags ff = std::ios::scientific;
    ff |= std::ios::showpos;
    os.setf(ff);

    for (size_t i = 0; i < m.rows(); ++i){
        for (size_t j = 0; j < m.cols(); ++j)
            os  << m(i,j) << '\t' ;
        os << '\n';
    }
    os.unsetf(ff);
    return os;
}
template<typename T>
inline  std::ostream &operator << (std::ostream &os,const Matrix<T,1> &m){
    std::ios_base::fmtflags ff = std::ios::scientific;
    ff |= std::ios::showpos;
    os.setf(ff);
    for (size_t i = 0; i < m.size(); ++i){
        os  << m(i) << '\n';
    }
    os.unsetf(ff);
    return os;
}
/*********************************************************************************/

/***************************arithmetic ops*****************************************/
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator +(const Scalar &val,const Matrix<T,N> &m){
    Matrix<T,N> R(m);
    return R+=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator +(const Scalar &val,Matrix<T,N> &&m){
    Matrix<T,N> R(m);
    return R+=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator +(const Matrix<T,N> &m,const Scalar &val){
    Matrix<T,N> R(m);
    return R+=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator +(Matrix<T,N> &&m,const Scalar &val){
    Matrix<T,N> R(m);
    return R+=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator -(const Scalar &val,const Matrix<T,N> &m){
    Matrix<T,N> R(m);
    std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
    return R;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator -(const Scalar &val,Matrix<T,N> &&m){
    Matrix<T,N> R(m);
    std::for_each(R.begin(),R.end(),[&val](auto &elem){elem = val-elem;});
    return R;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator -(const Matrix<T,N> &m,const Scalar &val){
    Matrix<T,N> R(m);
    return R-=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator -(Matrix<T,N> &&m,const Scalar &val){
    Matrix<T,N> R(m);
    return R-=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator *(const Scalar &val,const Matrix<T,N> &m){
    Matrix<T,N> R(m);
    return R*=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator *(const Scalar &val,Matrix<T,N> &&m){
    Matrix<T,N> R(m);
    return R*=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator *(const Matrix<T,N> &m,const Scalar &val){
    Matrix<T,N> R(m);
    return R*=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator *(Matrix<T,N> &&m,const Scalar &val){
    Matrix<T,N> R(m);
    return R*=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator /(const Matrix<T,N> &m,const Scalar &val){
    Matrix<T,N> R(m);
    return R/=val;
}
template<typename T,typename Scalar,size_t N>
inline Matrix<T,N> operator /(Matrix<T,N> &&m,const Scalar &val){
    Matrix<T,N> R(m);
    return R/=val;
}
template<typename T,size_t N>
inline Matrix<T,N> operator+(const Matrix<T,N> &m1,const Matrix<T,N> &m2){
    Matrix<T,N> R(m1);
    return R+=m2;
}
template<typename T,size_t N>
inline Matrix<T,N> operator+(Matrix<T,N> &&m1,const Matrix<T,N> &m2){
    Matrix<T,N> R(m1);
    return R+=m2;
}
template<typename T,size_t N>
inline Matrix<T,N> operator+(const Matrix<T,N> &m1,Matrix<T,N> &&m2){
    Matrix<T,N> R(m2);
    return R+=m1;
}
template<typename T,size_t N>
inline Matrix<T,N> operator-(const Matrix<T,N> &m1,const Matrix<T,N> &m2){
    Matrix<T,N> R(m1);
    return R-=m2;
}
template<typename T,size_t N>
inline Matrix<T,N> operator-(Matrix<T,N> &&m1,const Matrix<T,N> &m2){
    Matrix<T,N> R(m1);
    return R-=m2;
}
template<typename T,size_t N>
inline Matrix<T,N> operator-(const Matrix<T,N> &m1,Matrix<T,N> &&m2){
    assert(std::equal(m1.descriptor().m_extents.begin(),m1.descriptor().m_extents.end(),
                          m2.descriptor().m_extents.begin(),m2.descriptor().m_extents.end()));
    Matrix<T,N> R(m2);
    std::transform(m1.begin(),m1.end(),R.begin(),R.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
    return R;
}
/*****************************specializations*********************************/
typedef std::complex<double> complexd;

template<size_t N>
inline Matrix<complexd,N> operator+(const Matrix<complexd,N> &cm,const Matrix<double,N> &m){
    Matrix<complexd,N> r(cm);
    return r+=m;
}
//template<size_t N>
//inline Matrix<complexd,N> operator+(Matrix<complexd,N> &&cm,const Matrix<double,N> &m){
//    Matrix<complexd,N> r(cm);
//    return r+=m;
//}
template<size_t N>
inline Matrix<complexd,N> operator+(const Matrix<double,N> &m,const Matrix<complexd,N> &cm){
    Matrix<complexd,N> r(cm);
    return r+=m;
}
//template<size_t N>
//inline Matrix<complexd,N> operator+(const Matrix<double,N> &&m,const Matrix<complexd,N> &cm){
//    Matrix<complexd,N> r(cm);
//    return r+=m;
//}
//template<size_t N>
//inline Matrix<complexd,N> operator+(const Matrix<double,N> &m,Matrix<complexd,N> &&cm){
//    Matrix<complexd,N> r(cm);
//    return r+=m;
//}
template<size_t N>
inline Matrix<complexd,N> operator-(const Matrix<complexd,N> &cm,const Matrix<double,N> &m){
    Matrix<complexd,N> r(cm);
    return r-=m;
}
//template<size_t N>
//inline Matrix<complexd,N> operator-(Matrix<complexd,N> &&cm,const Matrix<double,N> &m){
//    Matrix<complexd,N> r(cm);
//    return r-=m;
//}
template<size_t N>
inline Matrix<complexd,N> operator-(const Matrix<double,N> &m,const Matrix<complexd,N> &cm){
    assert(std::equal(m.descriptor().m_extents.begin(),m.descriptor().m_extents.end(),
                      cm.descriptor().m_extents.begin(),cm.descriptor().m_extents.end()));
    Matrix<complexd,N> r(cm);
    std::transform(m.begin(),m.end(),r.begin(),r.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
    return r;
}
//template<size_t N>
//inline Matrix<complexd,N> operator-(const Matrix<double,N> &&m,const Matrix<complexd,N> &cm){
//    assert(std::equal(m.descriptor().m_extents.begin(),m.descriptor().m_extents.end(),
//                      cm.descriptor().m_extents.begin(),cm.descriptor().m_extents.end()));
//    Matrix<complexd,N> r(cm);
//    std::transform(m.begin(),m.end(),r.begin(),r.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
//    return r;
//}
//template<size_t N>
//inline Matrix<complexd,N> operator-(const Matrix<double,N> &m,Matrix<complexd,N> &&cm){
//    assert(std::equal(m.descriptor().m_extents.begin(),m.descriptor().m_extents.end(),
//                      cm.descriptor().m_extents.begin(),cm.descriptor().m_extents.end()));
//    Matrix<complexd,N> r(cm);
//    std::transform(m.begin(),m.end(),r.begin(),r.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
//    return r;
//}

/*****************************************************************************/
/******************************sparse matrix operations************************/
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR> operator+(const Matrix<T,2,MATRIX_TYPE::CSR> &spm1,
                                                   const Matrix<T,2,MATRIX_TYPE::CSR> &spm2){
    size_t nrows = spm1.rows();
    size_t ncols = spm1.cols();
    assert(nrows == spm2.rows() && ncols == spm2.cols());

    if (spm1.values().size() == 0 && spm2.values().size() == 0) return spm1;
    if (spm1.values().size() == 0 ) return spm2;
    if (spm2.values().size() == 0) return spm1;

    vector<int_t> rowStart(nrows);
    vector<int_t> rowEnd(nrows);
    vector<int_t> columns;
    columns.reserve(spm1.values().size() + spm2.values().size());
    vector<T> vals;
    vals.reserve(spm1.values().size() + spm2.values().size());
    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        int_t beg1 = spm1.row_start()[i];
        int_t end1 = spm1.row_end()[i];
        int_t beg2 = spm2.row_start()[i];
        int_t end2 = spm2.row_end()[i];

        int_t col1 = std::numeric_limits<int_t>::max();
        int_t col2 = std::numeric_limits<int_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<int_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<int_t>::max();

            if (col1 < col2){
                T val1 = spm1.values()[beg1];
                vals.push_back(val1);
                columns.push_back(col1);
                ++beg1;
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 > col2){
                T val2 = spm2.values()[beg2];
                vals.push_back(val2);
                columns.push_back(col2);
                ++beg2; //beg2 pasa al nuevo valor
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 == col2){
                T val1 = spm1.values()[beg1];
                T val2 = spm2.values()[beg2];
                vals.push_back(val1+val2);
                columns.push_back(col1);
                ++beg1;
                ++beg2;
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
        }
        if (first_rinclusion){ //full zeros row
            rowStart[i] = vals.size();
            rowEnd[i] = vals.size();
        }else{
            rowEnd[i] = vals.size();}
    }

    return Matrix<T,2,MATRIX_TYPE::CSR>(nrows,ncols,rowStart,rowEnd,columns,vals);

}
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR> operator-(const Matrix<T,2,MATRIX_TYPE::CSR> &spm1,
                                                const Matrix<T,2,MATRIX_TYPE::CSR> &spm2){
    size_t nrows = spm1.rows();
    size_t ncols = spm1.cols();
    assert(nrows == spm2.rows() && ncols == spm2.cols());

    if (spm1.values().size() == 0 && spm2.values().size() == 0) return spm1;
    if (spm1.values().size() == 0 ) return spm2;
    if (spm2.values().size() == 0) return spm1;

    vector<int_t> rowStart(nrows);
    vector<int_t> rowEnd(nrows);
    vector<int_t> columns;
    vector<T> vals;

    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        int_t beg1 = spm1.row_start()[i];
        int_t end1 = spm1.row_end()[i];
        int_t beg2 = spm2.row_start()[i];
        int_t end2 = spm2.row_end()[i];

        int_t col1 = std::numeric_limits<int_t>::max();
        int_t col2 = std::numeric_limits<int_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<int_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<int_t>::max();

            if (col1 < col2){
                T val1 = spm1.values()[beg1];
                vals.push_back(val1);
                columns.push_back(col1);
                ++beg1;
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 > col2){
                T val2 = spm2.values()[beg2];
                vals.push_back(-val2);
                columns.push_back(col2);
                ++beg2; //beg2 pasa al nuevo valor
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 == col2){
                T val1 = spm1.values()[beg1];
                T val2 = spm2.values()[beg2];
                vals.push_back(val1-val2);
                columns.push_back(col1);
                ++beg1;
                ++beg2;
                if (first_rinclusion){
                    rowStart[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
        }
        if (first_rinclusion){ //full zeros row
            rowStart[i] = vals.size();
            rowEnd[i] = vals.size();
        }else{
            rowEnd[i] = vals.size();
        }
    }

    return Matrix<T,2,MATRIX_TYPE::CSR>(nrows,ncols,rowStart,rowEnd,columns,vals);

}

/******************************************************************************/
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR3> operator+(const Matrix<T,2,MATRIX_TYPE::CSR3> &spm1,
                                                 const Matrix<T,2,MATRIX_TYPE::CSR3> &spm2){
    size_t nrows = spm1.rows();
    size_t ncols = spm1.cols();
    assert(nrows == spm2.rows() && ncols == spm2.cols());

    if (spm1.values().size() == 0 && spm2.values().size() == 0) return spm1;
    if (spm1.values().size() == 0 ) return spm2;
    if (spm2.values().size() == 0) return spm1;

    vector<int_t> rowIndex(nrows+1);
    vector<int_t> columns;
    columns.reserve(spm1.values().size() + spm2.values().size());
    vector<T> vals;
    vals.reserve(spm1.values().size() + spm2.values().size());
    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        int_t beg1 = spm1.rowIndex()[i];
        int_t end1 = spm1.rowIndex()[i+1];
        int_t beg2 = spm2.rowIndex()[i];
        int_t end2 = spm2.rowIndex()[i+1];

        int_t col1 = std::numeric_limits<int_t>::max();
        int_t col2 = std::numeric_limits<int_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<int_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<int_t>::max();

            if (col1 < col2){
                T val1 = spm1.values()[beg1];
                vals.push_back(val1);
                columns.push_back(col1);
                ++beg1;
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 > col2){
                T val2 = spm2.values()[beg2];
                vals.push_back(val2);
                columns.push_back(col2);
                ++beg2; //beg2 pasa al nuevo valor
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 == col2){
                T val1 = spm1.values()[beg1];
                T val2 = spm2.values()[beg2];
                vals.push_back(val1+val2);
                columns.push_back(col1);
                ++beg1;
                ++beg2;
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
        }
        if (first_rinclusion){ //full zeros row
            rowIndex[i] = vals.size();
            //rowIndex[i+1] = vals.size();
        }else{
            //rowIndex[i+1] = vals.size();}
        }
        rowIndex[nrows] = vals.size();
    }

    return Matrix<T,2,MATRIX_TYPE::CSR3>(nrows,ncols,rowIndex,columns,vals);

}
template<typename T>
inline Matrix<T,2,MATRIX_TYPE::CSR3> operator-(const Matrix<T,2,MATRIX_TYPE::CSR3> &spm1,
                                                 const Matrix<T,2,MATRIX_TYPE::CSR3> &spm2){
    size_t nrows = spm1.rows();
    size_t ncols = spm1.cols();
    assert(nrows == spm2.rows() && ncols == spm2.cols());

    if (spm1.values().size() == 0 && spm2.values().size() == 0) return spm1;
    if (spm1.values().size() == 0 ) return spm2;
    if (spm2.values().size() == 0) return spm1;

    vector<int_t> rowIndex(nrows+1);
    vector<int_t> columns;
    columns.reserve(spm1.values().size() + spm2.values().size());
    vector<T> vals;
    vals.reserve(spm1.values().size() + spm2.values().size());
    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        int_t beg1 = spm1.rowIndex()[i];
        int_t end1 = spm1.rowIndex()[i+1];
        int_t beg2 = spm2.rowIndex()[i];
        int_t end2 = spm2.rowIndex()[i+1];

        int_t col1 = std::numeric_limits<int_t>::max();
        int_t col2 = std::numeric_limits<int_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<int_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<int_t>::max();

            if (col1 < col2){
                T val1 = spm1.values()[beg1];
                vals.push_back(val1);
                columns.push_back(col1);
                ++beg1;
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 > col2){
                T val2 = spm2.values()[beg2];
                vals.push_back(-val2);
                columns.push_back(col2);
                ++beg2; //beg2 pasa al nuevo valor
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
            if (col1 == col2){
                T val1 = spm1.values()[beg1];
                T val2 = spm2.values()[beg2];
                vals.push_back(val1-val2);
                columns.push_back(col1);
                ++beg1;
                ++beg2;
                if (first_rinclusion){
                    rowIndex[i] = vals.size()-1;
                    first_rinclusion = false;
                }
                continue;
            }
        }
        if (first_rinclusion){ //full zeros row
            rowIndex[i] = vals.size();
            //rowIndex[i+1] = vals.size();
        }else{
            //rowIndex[i+1] = vals.size();}
        }
    }
    rowIndex[nrows] = vals.size();

    return Matrix<T,2,MATRIX_TYPE::CSR3>(nrows,ncols,rowIndex,columns,vals);

}
/*****************************Matrix Multiplication********************************/
template<typename T>
inline T operator*(const Matrix<T,1> &v1,const Matrix<T,1> &v2){
    assert(v1.size() == v2.size());
    return std::inner_product(v1.begin(),v1.end(),v2.begin(),T());
}
template<typename T>
inline Matrix<T,2> dyadic_product(const Matrix<T,1> &a,const Matrix<T,1> &b){
    Matrix<T,2> res(a.size(),b.size());
    for (size_t i = 0; i < a.size(); ++i){
        T a_val = a(i);
        for (size_t j = 0; j < b.size(); ++j){
            res(i,j) = a_val*b(j);
        }
    }
    return res;
}
template<typename T>
inline Matrix<T,1> operator*(const Matrix<T,2> &m,const Matrix<T,1> &v){
    assert(m.cols() == v.size());
    Matrix<T,1> r(m.rows());

    for (size_t i = 0; i < r.size(); ++i){
        Matrix_Ref<const T,1> mref = m.row(i);
        r(i) = std::inner_product(mref.data(),mref.data()+mref.size(),v.begin(),0.0);
    }
    return r;
}
template<typename T>
inline Matrix<T,1> operator*(const Matrix<T,1> &v,const Matrix<T,2> &m){
    assert(m.rows() == v.size());
    Matrix<T,1> r(m.cols(),T());

    for (size_t i = 0; i < v.size(); ++i){
        T val = v(i);
        for (size_t j = 0; j < r.size(); ++j){
            r(j) += val*m(i,j);
        }
    }
    return r;
}
inline Matrix<double,2> operator*(const Matrix<double,2> &m1,const Matrix<double,2> &m2){
    size_t m1_rows = m1.rows();
    size_t m1_cols = m1.cols();
    size_t m2_rows = m2.rows();
    size_t m2_cols = m2.cols();
    assert(m1_cols == m2_rows);
    Matrix<double,2> res(m1_rows,m2_cols);
    cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
                int(m1_rows),int(m2_cols),int(m1_cols),1.0,m1.begin(),
                int(m1_cols),m2.begin(),int(m2_cols),0.0,res.begin(),int(m2_cols));
    return res;
}
/**********************************************************************************/
#endif // MATRIX_H
