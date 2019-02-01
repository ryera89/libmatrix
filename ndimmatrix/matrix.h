#ifndef MATRIX_H
#define MATRIX_H

#include <valarray>
#include <vector>
#include <complex>
#include <type_traits>
#include "matrix_ref.h"
#include <iostream>
#include <iomanip>
#include "mkl.h"

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

    template<typename... Exts>
    Matrix(Exts... exts):m_desc(0,{exts...}),m_elems(m_desc.m_size){}

    Matrix(const Matrix_Ref<T,N> &ref){
        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        m_desc.m_start = 0;
        m_desc.m_extents = ref.descriptor().m_extents;
        m_desc.init();
        m_elems.resize(m_desc.m_size);
        Matrix_Ref<T,N> mref{begin(),m_desc};
        assing_slice_vals(ref,mref);
    }
    Matrix& operator = (const Matrix_Ref<T,N> &ref){
        //static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        m_desc.m_start = 0;
        m_desc.m_extents = ref.descriptor().m_extents;
        m_desc.init();
        m_elems.resize(m_desc.m_size);
        Matrix_Ref<T,N> mref{begin(),m_desc};
        assing_slice_vals(ref,mref);
        return *this;
    }
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

    //std::valarray<T>& valarray() {return m_elems;}
    const std::valarray<T>& valarray() const{return m_elems;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    Matrix apply(T (func)(T val)){Matrix r(m_desc.m_extents); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(m_desc.m_extents); std::transform(begin(),end(),r.begin(),func); return r;}
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
        m_elems+=m.valarray();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,N> &m)
    {
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          m.descriptor().m_extents.begin(),m.descriptor().m_extents.end()));
        m_elems-=m.valarray();
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

    //std::valarray<T>& valarray() {return m_elems;}
    const std::valarray<T>& valarray() const{return m_elems;}

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
    Matrix<T,2> operator()(const std::vector<uint32_t> &indx1,const std::vector<uint32_t> &indx2) const{
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
        m_elems+=m.valarray();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.valarray();
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

    //std::valarray<T>& valarray() {return m_elems;}
    const std::valarray<T>& valarray() const{return m_elems;}

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
    Matrix<T,2> operator()(const std::vector<uint32_t> &indx1,const std::vector<uint32_t> &indx2) const{
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
        m_elems+=m.valarray();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,2,matrix_type> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.valarray();
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

    //std::valarray<T>& valarray() {return m_elems;}
    const std::valarray<T>& valarray() const{return m_elems;}

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
    Matrix<T,2> operator()(const std::vector<uint32_t> &indx1,const std::vector<uint32_t> &indx2) const{
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
        m_elems+=m.valarray();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        m_elems-=m.valarray();
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
//template<typename T>
//class Matrix<T,2,MATRIX_TYPE::CSR>{
//private:
//    MKL_INT m_row;
//    MKL_INT m_col;
//    std::vector<MKL_INT> m_rows_end;
//    std::vector<MKL_INT> m_rows_start;
//    std::vector<MKL_INT> m_columns;
//    std::vector<T> m_elems;
//    static constexpr T zero_val = T();

//public:
//    static constexpr size_t order = 2;
//    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR;
//    using value_type = T;

//    Matrix() = default;
//    ~Matrix() = default;
//    //Move constructor and assignment
//    Matrix(Matrix&&) = default;
//    Matrix& operator=(Matrix&&) = default;
//    //Copy constructor and assignment
//    Matrix(const Matrix&) = default;
//    Matrix& operator=(const Matrix&) = default;


//    Matrix(MKL_INT row,MKL_INT col,const std::vector<MKL_INT> &pointerE,const std::vector<MKL_INT> &pointerB,
//           const std::vector<MKL_INT> &columns,const std::vector<T> &vals):m_row(row),m_col(col),m_rows_end(pointerE),
//                                                                               m_rows_start(pointerB),m_columns(columns),m_elems(vals){
//        assert(m_columns.size() == m_elems.size() && m_row == m_rows_start.size() && m_row == m_rows_end.size());
//    }

//    Matrix(MKL_INT row,MKL_INT col,std::vector<MKL_INT> &&pointerE,std::vector<MKL_INT> &&pointerB,
//           std::vector<MKL_INT> &&columns,std::vector<T> &&vals):m_row(row),m_col(col),m_rows_end(pointerE),
//                                                                     m_rows_start(pointerB),m_columns(columns),m_elems(vals){
//        assert(m_columns.size() == m_elems.size() && m_row == m_rows_start.size() && m_row == m_rows_end.size());
//    }
//    Matrix(const Matrix<T,2> &m):m_row(m.rows()),m_col(m.cols()),m_rows_end(m_row),m_rows_start(m_row){
//        for (size_t i = 0; i < m_row; ++i){
//            size_t iitmp = m_row;
//            for (size_t j = 0; j < m_col; ++j){
//                T val = m(i,j);
//                if (val != T()){ //si el valor es distinto de cero
//                    m_elems.push_back(val);
//                    m_columns.push_back(j);
//                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
//                        m_rows_start[i] = m_elems.size()-1;
//                        iitmp = i;
//                    }
//                }
//            }
//            if (iitmp != i){ //una fila llena de zeros
//                if (m_elems.size() == 0) {m_rows_start[i] = 0; m_rows_end[i] = 0;} /*primeras filas == 0*/
//                else {m_rows_start[i] = m_elems.size(); m_rows_end[i] = m_elems.size();}
//            }else{ m_rows_end[i] = m_elems.size();}

//        }
//    }
//    Matrix& operator=(const Matrix<T,2> &m){
//        m_row = m.rows();
//        m_col = m.cols();
//        m_rows_end.resize(m_row);
//        m_rows_start.resize(m_row);
//        m_columns.clear();
//        m_elems.clear();
//        for (size_t i = 0; i < m_row; ++i){
//            size_t iitmp = m_row;
//            for (size_t j = 0; j < m_col; ++j){
//                T val = m(i,j);
//                if (val != T()){ //si el valor es distinto de cero
//                    m_elems.push_back(val);
//                    m_columns.push_back(j);
//                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
//                        m_rows_start[i] = m_elems.size()-1;
//                        iitmp = i;
//                    }
//                }
//            }
//            if (iitmp != i){ //una fila llena de zeros
//                if (m_elems.size() == 0) {m_rows_start[i] = 0; m_rows_end[i] = 0;} /*primeras filas == 0*/
//                else {m_rows_start[i] = m_elems.size(); m_rows_end[i] = m_elems.size();}
//            }else{ m_rows_end[i] = m_elems.size();}

//        }
//    }
//    const T& operator()(uint32_t i,uint32_t j) const{
//        assert(i < m_row && j < m_col);
//        uint32_t beg = m_rows_start[i];
//        uint32_t end = m_rows_end[i];

//        if (beg == end) return zero_val;
//        if (j < m_columns[beg]) return zero_val;
//        if (j > m_columns[end-1]) return zero_val;

//        for (;beg<end;++beg){
//            if (m_columns[beg] == j) return m_elems[beg];
//        }
//        return zero_val;
//    }
//    Matrix operator()(const std::valarray<uint32_t> &iindex,const std::valarray<uint32_t> &jindex) const{
//        size_t nrow = iindex.size();
//        size_t ncol = jindex.size();

//        std::vector<T> elems;
//        std::vector<uint32_t> columns;
//        std::vector<uint32_t> pointerB(nrow);
//        std::vector<uint32_t> pointerE(nrow);

//        for (uint32_t i = 0; i < nrow; ++i){
//            uint32_t itmp = nrow;
//            uint32_t ii = iindex[i];
//            for (uint32_t j = 0; j < ncol; ++j){
//                uint32_t jj = jindex[j];
//                T val = this->operator()(ii,jj);
//                if (val != T()){ //si el valor es distinto de cero
//                    elems.push_back(val);
//                    columns.push_back(j);
//                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
//                        pointerB[i] = elems.size()-1;
//                        itmp = i;
//                    }
//                }
//            }
//            if (itmp != i){ //una fila llena de zeros
//                if (elems.size() == 0) {pointerB[i] = 0; pointerE[i] = 0;} /*primeras filas == 0*/
//                else {pointerB[i] = elems.size(); pointerE[i] = elems.size();}
//            }else{ pointerE[i] = elems.size();}
//        }

//        return Matrix(nrow,ncol,pointerE,pointerB,columns,elems);
//    }
//    Matrix operator()(const std::vector<uint32_t> &iindex,const std::vector<uint32_t> &jindex) const{
//        MKL_INT nrow = iindex.size();
//        MKL_INT ncol = jindex.size();

//        std::vector<T> elems;
//        std::vector<MKL_INT> columns;
//        std::vector<MKL_INT> pointerB(nrow);
//        std::vector<MKL_INT> pointerE(nrow);

//        for (MKL_INT i = 0; i < nrow; ++i){
//            MKL_INT itmp = nrow;
//            uint32_t ii = iindex[i];
//            for (MKL_INT j = 0; j < ncol; ++j){
//                uint32_t jj = jindex[j];
//                T val = this->operator()(ii,jj);
//                if (val != T()){ //si el valor es distinto de cero
//                    elems.push_back(val);
//                    columns.push_back(j);
//                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
//                        pointerB[i] = elems.size()-1;
//                        itmp = i;
//                    }
//                }
//            }
//            if (itmp != i){ //una fila llena de zeros
//                if (elems.size() == 0) {pointerB[i] = 0; pointerE[i] = 0;} /*primeras filas == 0*/
//                else {pointerB[i] = elems.size(); pointerE[i] = elems.size();}
//            }else{ pointerE[i] = elems.size();}
//        }

//        return Matrix(nrow,ncol,pointerE,pointerB,columns,elems);
//    }

//    MKL_INT rows() const{return m_row;}
//    MKL_INT cols() const{return m_col;}

//    void printData(){
//        printf("values: ( ");
//        for (auto &vals : m_elems) printf("%f ",vals);
//        printf(") \n");
//        printf("column: ( ");
//        for (auto &vals : m_columns) printf("%u ",vals);
//        printf(") \n");
//        printf("pointerB: ( ");
//        for (auto &vals : m_rows_start) printf("%u ",vals);
//        printf(") \n");
//        printf("pointerE: ( ");
//        for (auto &vals : m_rows_end) printf("%u ",vals);
//        printf(") \n");
//    }
//    auto begin(){return std::begin(m_elems);}
//    auto begin() const{return std::cbegin(m_elems);}

//    auto end(){return std::end(m_elems);}
//    auto end() const{return std::cend(m_elems);}

//    auto columnssData(){return m_columns.data();}
//    auto columnssData() const {return m_columns.data();}

//    auto rowsStartData(){return m_rows_start.data();}
//    auto rowsStartData() const{return m_rows_start.data();}

//    auto rowsEndData(){return m_rows_end.data();}
//    auto rowsEndData() const{return m_rows_end.data();}

//    auto valuesData(){return m_elems.data();}
//    auto valuesData() const{return m_elems.data();}
//};
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR>{
private:
    uint32_t m_rows;
    uint32_t m_cols;
    std::vector<uint32_t> m_rows_start;
    std::vector<uint32_t> m_rows_end;
    std::vector<uint32_t> m_columns;
    std::vector<T> m_elems;
    static constexpr T zero_val = T();
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR;
    using value_type = T;

    Matrix() = default;
    ~Matrix() = default;
    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;


    Matrix(uint32_t rows,uint32_t cols,const std::vector<uint32_t> &rows_start,const std::vector<uint32_t> &rows_end,
           const std::vector<uint32_t> &columns,const std::vector<T> &vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),
                                                                             m_rows_end(rows_end),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
    }

    Matrix(uint32_t rows,uint32_t cols,std::vector<uint32_t> &&rows_start,std::vector<uint32_t> &&rows_end,std::vector<uint32_t> &&columns,
           std::vector<T> &&vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),m_rows_end(rows_end),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
    }
    Matrix(const Matrix<T,2> &m):m_rows(m.rows()),m_cols(m.cols()),m_rows_start(m_rows),m_rows_end(m_rows){
        for (size_t i = 0; i < m_rows; ++i){
            size_t iitmp = m_rows;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        iitmp = i;
                    }
                }
            }
            if (iitmp != i){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}

        }
    }
    Matrix& operator=(const Matrix<T,2> &m){
        m_rows = m.rows();
        m_cols = m.cols();
        m_rows_start.resize(m_rows);
        m_rows_end.resize(m_rows);
        m_columns.clear();
        m_elems.clear();
        for (size_t i = 0; i < m_rows; ++i){
            size_t iitmp = m_rows;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        iitmp = i;
                    }
                }
            }
            if (iitmp != i){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}
        }
        return *this;
    }
    const T& operator()(uint32_t i,uint32_t j) const{
        assert(i < m_rows && j < m_cols);
        uint32_t beg = m_rows_start[i];
        uint32_t end = m_rows_end[i];

        if (beg == end) return zero_val;
        if (j < m_columns[beg]) return zero_val;
        if (j > m_columns[end-1]) return zero_val;

        for (;beg<end;++beg){
            if (m_columns[beg] == j) return m_elems[beg];
        }
        return zero_val;
    }
    Matrix operator()(const std::valarray<uint32_t> &iindex,const std::valarray<uint32_t> &jindex) const{
        size_t nrow = iindex.size();
        size_t ncol = jindex.size();

        std::vector<T> elems;
        std::vector<uint32_t> columns;
        std::vector<uint32_t> pointerB(nrow);
        std::vector<uint32_t> pointerE(nrow);

        for (uint32_t i = 0; i < nrow; ++i){
            uint32_t itmp = nrow;
            uint32_t ii = iindex[i];
            for (uint32_t j = 0; j < ncol; ++j){
                uint32_t jj = jindex[j];
                T val = this->operator()(ii,jj);
                if (val != T()){ //si el valor es distinto de cero
                    elems.push_back(val);
                    columns.push_back(j);
                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        pointerB[i] = elems.size()-1;
                        itmp = i;
                    }
                }
            }
            if (itmp != i){ //una fila llena de zeros
                if (elems.size() == 0) {pointerB[i] = 0; pointerE[i] = 0;} /*primeras filas == 0*/
                else {pointerB[i] = elems.size(); pointerE[i] = elems.size();}
            }else{ pointerE[i] = elems.size();}
        }

        return Matrix(nrow,ncol,pointerB,pointerE,columns,elems);
    }
    Matrix operator()(const std::vector<uint32_t> &iindex,const std::vector<uint32_t> &jindex) const{
        uint32_t nrow = iindex.size();
        uint32_t ncol = jindex.size();

        std::vector<T> elems;
        std::vector<uint32_t> columns;
        std::vector<uint32_t> pointerB(nrow);
        std::vector<uint32_t> pointerE(nrow);

        for (uint32_t i = 0; i < nrow; ++i){
            uint32_t itmp = nrow;
            uint32_t ii = iindex[i];
            for (uint32_t j = 0; j < ncol; ++j){
                uint32_t jj = jindex[j];
                T val = this->operator()(ii,jj);
                if (val != T()){ //si el valor es distinto de cero
                    elems.push_back(val);
                    columns.push_back(j);
                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        pointerB[i] = elems.size()-1;
                        itmp = i;
                    }
                }
            }
            if (itmp != i){ //una fila llena de zeros
                if (elems.size() == 0) {pointerB[i] = 0; pointerE[i] = 0;} /*primeras filas == 0*/
                else {pointerB[i] = elems.size(); pointerE[i] = elems.size();}
            }else{ pointerE[i] = elems.size();}
        }

        return Matrix(nrow,ncol,pointerB,pointerE,columns,elems);
    }

    uint32_t rows() const{return m_rows;}
    uint32_t cols() const{return m_cols;}

    void printData(){
        printf("values: ( ");
        for (auto &vals : m_elems) std::cout  << vals << " ";
        printf(") \n");
        printf("column: ( ");
        for (auto &vals : m_columns) std::cout << vals << " ";
        printf(") \n");
        printf("pointerB: ( ");
        for (auto &vals : m_rows_start) std::cout <<  vals << " ";
        printf(") \n");
        printf("pointerE: ( ");
        for (auto &vals : m_rows_end) std::cout << vals << " ";
        printf(") \n");
    }
    auto values() const{return m_elems;}
    auto columns() const{return m_columns;}
    auto row_start() const{return m_rows_start;}
    auto row_end() const{return m_rows_end;}

    auto beginColumns(){return m_columns.begin();}
    auto beginColumns() const {return m_columns.cbegin();}
    auto endColumns(){return m_columns.end();}
    auto endColumns() const {return m_columns.cend();}

    auto beginRowsStart(){return m_rows_start.begin();}
    auto beginRowsStart() const{return m_rows_start.cbegin();}
    auto endRowsStart(){return m_rows_start.end();}
    auto endRowsStart() const{return m_rows_start.cend();}

    auto beginRowsEnd(){return m_rows_end.begin();}
    auto beginRowsEnd() const{return m_rows_end.cbegin();}
    auto endRowsEnd(){return m_rows_end.end();}
    auto endRowsEnd() const{return m_rows_end.cend();}

    auto beginValues(){return std::begin(m_elems);}
    auto beginValues() const{return std::cbegin(m_elems);}
    auto endValues(){return std::end(m_elems);}
    auto endValues() const{return std::cend(m_elems);}

    auto columnsData(){return m_columns.data();}
    auto columnsData() const {return m_columns.data();}

    auto rowsStartData(){return m_rows_start.data();}
    auto rowsStartData() const{return m_rows_start.data();}

    auto rowsEndData(){return m_rows_end.data();}
    auto rowsEndData() const{return m_rows_end.data();}

    auto valuesData(){return m_elems.data();}
    auto valuesData() const{return m_elems.data();}
};
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR3>{
private:
    size_t m_row;
    size_t m_col;
    std::vector<uint32_t> m_rowIndex;
    std::vector<uint32_t> m_columns;
    std::vector<T> m_elems;
    static constexpr T zero_val = T();
public:
    static constexpr size_t order = 2;
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR3;
    using value_type = T;

    Matrix() = default;
    ~Matrix() = default;
    //Move constructor and assignment
    Matrix(Matrix&&) = default;
    Matrix& operator=(Matrix&&) = default;
    //Copy constructor and assignment
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;


    Matrix(size_t row,size_t col,const std::vector<uint32_t> &rowIndex,const std::vector<uint32_t> &columns,const std::vector<T> &vals):
                                                                     m_row(row),m_col(col),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_row == (m_rowIndex.size()-1));

    }
    Matrix(size_t row,size_t col,std::vector<uint32_t> &&rowIndex,std::vector<uint32_t> &&columns,std::vector<T> &&vals):
                                                                     m_row(row),m_col(col),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_row == (m_rowIndex.size()-1));
    }
    Matrix(const Matrix<T,2> &m):m_row(m.rows()),m_col(m.cols()),m_rowIndex(m_row+1){
        for (size_t i = 0; i < m_row; ++i){
            size_t iitmp = m_row;
            for (size_t j = 0; j < m_col; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        m_rowIndex[i] = m_elems.size()-1;
                        iitmp = i;
                    }
                }
            }
            if (iitmp != i){ //una fila llena de zeros
                if (m_elems.size() == 0) {m_rowIndex[i] = 0;} /*primeras filas == 0*/
                else {m_rowIndex[i] = m_elems.size();}
            }
        }
        m_rowIndex[m_row] = m_elems.size();
    }
    Matrix& operator=(const Matrix<T,2> &m){
        m_row = m.rows();
        m_col = m.cols();
        m_rowIndex.resize(m_row+1);
        m_columns.clear();
        m_elems.clear();
        for (size_t i = 0; i < m_row; ++i){
            size_t iitmp = m_row;
            for (size_t j = 0; j < m_col; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (iitmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        m_rowIndex[i] = m_elems.size()-1;
                        iitmp = i;
                    }
                }
            }
            if (iitmp != i){ //una fila llena de zeros
                if (m_elems.size() == 0) {m_rowIndex[i] = 0;} /*primeras filas == 0*/
                else {m_rowIndex[i] = m_elems.size();}
            }
        }
        m_rowIndex[m_row] = m_elems.size();
    }

    const T& operator()(uint32_t i,uint32_t j) const{
        assert(i < m_row && j < m_col);
        uint32_t beg = m_rowIndex[i];
        uint32_t end = m_rowIndex[i+1];

        if (beg == end) return zero_val;
        if (j < m_columns[beg]) return zero_val;
        if (j > m_columns[end-1]) return zero_val;

        for (;beg<end;++beg){
            if (m_columns[beg] == j) return m_elems[beg];
        }
        return zero_val;
    }
    Matrix operator()(const std::valarray<uint32_t> &iindex,const std::valarray<uint32_t> &jindex) const{
        size_t nrow = iindex.size();
        size_t ncol = jindex.size();

        std::vector<T> elems;
        std::vector<uint32_t> columns;
        std::vector<uint32_t> rowIndex(nrow+1);

        for (uint32_t i = 0; i < nrow; ++i){
            uint32_t itmp = nrow;
            uint32_t ii = iindex[i];
            for (uint32_t j = 0; j < ncol; ++j){
                uint32_t jj = jindex[j];
                T val = this->operator()(ii,jj);
                if (val != T()){ //si el valor es distinto de cero
                    elems.push_back(val);
                    columns.push_back(j);
                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        rowIndex[i] = elems.size()-1;
                        itmp = i;
                    }
                }
            }
            if (itmp != i){ //una fila llena de zeros
                if (elems.size() == 0) {rowIndex[i] = 0;} /*primeras filas == 0*/
                else {rowIndex[i] = elems.size();}
            }
        }
        rowIndex[nrow] = elems.size();

        return Matrix(nrow,ncol,rowIndex,columns,elems);
    }
    Matrix operator()(const std::vector<uint32_t> &iindex,const std::vector<uint32_t> &jindex) const{
        size_t nrow = iindex.size();
        size_t ncol = jindex.size();

        std::vector<T> elems;
        std::vector<uint32_t> columns;
        std::vector<uint32_t> rowIndex(nrow+1);

        for (uint32_t i = 0; i < nrow; ++i){
            uint32_t itmp = nrow;
            uint32_t ii = iindex[i];
            for (uint32_t j = 0; j < ncol; ++j){
                uint32_t jj = jindex[j];
                T val = this->operator()(ii,jj);
                if (val != T()){ //si el valor es distinto de cero
                    elems.push_back(val);
                    columns.push_back(j);
                    if (itmp != i){ //se ejecuta maximo solo una ves del loop principal i
                        rowIndex[i] = elems.size()-1;
                        itmp = i;
                    }
                }
            }
            if (itmp != i){ //una fila llena de zeros
                if (elems.size() == 0) {rowIndex[i] = 0;} /*primeras filas == 0*/
                else {rowIndex[i] = elems.size();}
            }
        }
        rowIndex[nrow] = elems.size();

        return Matrix(nrow,ncol,rowIndex,columns,elems);
    }

    void printData(){
        printf("values: ( ");
        for (auto &vals : m_elems) printf("%f ",vals);
        printf(") \n");
        printf("column: ( ");
        for (auto &vals : m_columns) printf("%u ",vals);
        printf(") \n");
        printf("rowIndex: ( ");
        for (auto &vals : m_rowIndex) printf("%u ",vals);
        printf(") \n");
    }

    size_t rows() const{return m_row;}
    size_t cols() const{return m_col;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}

    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    auto columnsData(){return std::begin(m_columns);}
    auto columnsData() const {return std::cbegin(m_columns);}

    auto rowIndexData(){return std::begin(m_rowIndex);}
    auto rowIndexData() const{return std::cbegin(m_rowIndex);}
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

    std::valarray<T>& valarray() {return m_elems;}
    const std::valarray<T>& valarray() const{return m_elems;}

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
        m_elems+=m.valarray();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,1> &m)
    {
        assert(this->size() == m.size());
        m_elems-=m.valarray();
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
template<size_t N>
inline Matrix<complexd,N> operator+(Matrix<complexd,N> &&cm,const Matrix<double,N> &m){
    Matrix<complexd,N> r(cm);
    return r+=m;
}
template<size_t N>
inline Matrix<complexd,N> operator+(const Matrix<double,N> &m,const Matrix<complexd,N> &cm){
    Matrix<complexd,N> r(cm);
    return r+=m;
}
template<size_t N>
inline Matrix<complexd,N> operator+(const Matrix<double,N> &m,Matrix<complexd,N> &&cm){
    Matrix<complexd,N> r(cm);
    return r+=m;
}
template<size_t N>
inline Matrix<complexd,N> operator-(const Matrix<complexd,N> &cm,const Matrix<double,N> &m){
    Matrix<complexd,N> r(cm);
    return r-=m;
}
template<size_t N>
inline Matrix<complexd,N> operator-(Matrix<complexd,N> &&cm,const Matrix<double,N> &m){
    Matrix<complexd,N> r(cm);
    return r-=m;
}
template<size_t N>
inline Matrix<complexd,N> operator-(const Matrix<double,N> &m,const Matrix<complexd,N> &cm){
    assert(std::equal(m.descriptor().m_extents.begin(),m.descriptor().m_extents.end(),
                      cm.descriptor().m_extents.begin(),cm.descriptor().m_extents.end()));
    Matrix<complexd,N> r(cm);
    std::transform(m.begin(),m.end(),r.begin(),r.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
    return r;
}
template<size_t N>
inline Matrix<complexd,N> operator-(const Matrix<double,N> &m,Matrix<complexd,N> &&cm){
    assert(std::equal(m.descriptor().m_extents.begin(),m.descriptor().m_extents.end(),
                      cm.descriptor().m_extents.begin(),cm.descriptor().m_extents.end()));
    Matrix<complexd,N> r(cm);
    std::transform(m.begin(),m.end(),r.begin(),r.begin(),[](const auto &v1,const auto &v2){return v1-v2;});
    return r;
}

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

    std::vector<uint32_t> rowStart(nrows);
    std::vector<uint32_t> rowEnd(nrows);
    std::vector<uint32_t> columns;
    std::vector<T> vals;

    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        uint32_t beg1 = spm1.row_start()[i];
        uint32_t end1 = spm1.row_end()[i];
        uint32_t beg2 = spm2.row_start()[i];
        uint32_t end2 = spm2.row_end()[i];

        uint32_t col1 = std::numeric_limits<uint32_t>::max();
        uint32_t col2 = std::numeric_limits<uint32_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<uint32_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<uint32_t>::max();

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

    std::vector<uint32_t> rowStart(nrows);
    std::vector<uint32_t> rowEnd(nrows);
    std::vector<uint32_t> columns;
    std::vector<T> vals;

    for (size_t i = 0; i < nrows; ++i){
        bool first_rinclusion = true;
        uint32_t beg1 = spm1.row_start()[i];
        uint32_t end1 = spm1.row_end()[i];
        uint32_t beg2 = spm2.row_start()[i];
        uint32_t end2 = spm2.row_end()[i];

        uint32_t col1 = std::numeric_limits<uint32_t>::max();
        uint32_t col2 = std::numeric_limits<uint32_t>::max();

        while (beg1 < end1 || beg2 < end2){

            if (beg1 < end1) col1 = spm1.columns()[beg1]; else col1 = std::numeric_limits<uint32_t>::max();
            if (beg2 < end2) col2 = spm2.columns()[beg2]; else col2 = std::numeric_limits<uint32_t>::max();

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
                T val2 = -spm2.values()[beg2];
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
            rowEnd[i] = vals.size();}
    }

    return Matrix<T,2,MATRIX_TYPE::CSR>(nrows,ncols,rowStart,rowEnd,columns,vals);

}

/******************************************************************************/
/*****************************Matrix Multiplication********************************/
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
