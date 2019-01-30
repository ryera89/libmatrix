#ifndef MATRIX_H
#define MATRIX_H

#include <valarray>
#include <complex>
#include <type_traits>
#include "matrix_ref.h"
#include <iostream>
#include <iomanip>

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
    template<typename U>
    Matrix& operator+=(const Matrix<U,N> &m)
    {
        assert(std::equal(this->begin(),this->end(),m.begin(),m.end()));
        m_elems+=m.valarray();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,N> &m)
    {
        assert(std::equal(this->begin(),this->end(),m.begin(),m.end()));
        m_elems-=m.valarray();
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
    template<typename U>
    Matrix& operator+=(const Matrix<U,2> &m){
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems+=m.valarray();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2> &m){
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.valarray();
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
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR>{
private:
    size_t m_row;
    size_t m_col;
    std::valarray<uint32_t> m_pointerE;
    std::valarray<uint32_t> m_pointerB;
    std::valarray<uint32_t> m_columns;
    std::valarray<T> m_elems;

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


    Matrix(size_t row,size_t col,const std::valarray<uint32_t> &pointerE,const std::valarray<uint32_t> &pointerB,
           const std::valarray<uint32_t> &columns,const std::valarray<T> &vals):m_row(row),m_col(col),m_pointerE(pointerE),
                                                                             m_pointerB(pointerB),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_row == (m_pointerB.size()-1) && m_row == (m_pointerE.size()-1));
    }

    Matrix(size_t row,size_t col,std::valarray<uint32_t> &&pointerE,std::valarray<uint32_t> &&pointerB,
           std::valarray<uint32_t> &&columns,std::valarray<T> &&vals):m_row(row),m_col(col),m_pointerE(pointerE),
                                                                                   m_pointerB(pointerB),m_columns(columns),m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_row == (m_pointerB.size()-1) && m_row == (m_pointerE.size()-1));
    }

    void printData(){
        printf("values: ( ");
        for (auto &vals : m_elems) printf("%f ",vals);
        printf(") \n");
        printf("column: ( ");
        for (auto &vals : m_columns) printf("%u ",vals);
        printf(") \n");
        printf("pointerB: ( ");
        for (auto &vals : m_pointerB) printf("%u ",vals);
        printf(") \n");
        printf("pointerE: ( ");
        for (auto &vals : m_pointerE) printf("%u ",vals);
        printf(") \n");
    }
    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}

    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    auto columnssData(){return std::begin(m_columns);}
    auto columnssData() const {return std::cbegin(m_columns);}

    auto pointerBData(){return std::begin(m_pointerB);}
    auto pointerBData() const{return std::cbegin(m_pointerB);}

    auto pointerEData(){return std::begin(m_pointerE);}
    auto pointerEData() const{return std::cbegin(m_pointerE);}
};
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR3>{
private:
    size_t m_row;
    size_t m_col;
    std::valarray<uint32_t> m_rowIndex;
    std::valarray<uint32_t> m_columns;
    std::valarray<T> m_elems;

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


    Matrix(size_t row,size_t col,const std::valarray<uint32_t> &rowIndex,const std::valarray<uint32_t> &columns,const std::valarray<T> &vals):
                                                                     m_row(row),m_col(col),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_row == (m_rowIndex.size()-1));

    }
    Matrix(size_t row,size_t col,std::valarray<uint32_t> &&rowIndex,std::valarray<uint32_t> &&columns,std::valarray<T> &&vals):
                                                                     m_row(row),m_col(col),m_elems(vals),m_columns(columns),m_rowIndex(rowIndex){
        assert(m_columns.size() == m_elems.size() && m_row == (m_rowIndex.size()-1));
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

template<typename T,MATRIX_TYPE mtype,typename = typename std::enable_if_t<mtype != MATRIX_TYPE::CSR
                                                                           && mtype != MATRIX_TYPE::CSR3>>
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
template<typename T,MATRIX_TYPE mtype,typename = typename std::enable_if_t<mtype != MATRIX_TYPE::CSR
                                                                           && mtype != MATRIX_TYPE::CSR3>>
inline  std::ostream &operator << (std::ostream &os,const Matrix<T,1,mtype> &m){
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

#endif // MATRIX_H
