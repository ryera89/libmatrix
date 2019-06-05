#ifndef HERM_MATRIX_H
#define HERM_MATRIX_H

#include "ndmatrix.h"

template<typename T>
class Matrix<T,2,MATRIX_TYPE::HER>{
private:
    size_t m_dim;
    Matrix_Slice<2> m_desc;
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

    Matrix(size_t m):m_dim(m),m_desc(0,m,m),m_elems(0.5*m*(m+1)){}
    Matrix(size_t m,T val):m_dim(m),m_desc(0,m,m),m_elems(val,0.5*m*(m+1)){}
    Matrix(Matrix_Slice<2> desc):m_dim(desc.m_extents[0]),m_desc(desc),m_elems(0.5*m_dim*(m_dim+1)){
        assert(desc.m_extents[0] == desc.m_extents[1]);
    }
    Matrix(const Matrix<T,2> &other):m_dim(other.rows()),m_desc(0,m_dim,m_dim),m_elems(0.5*m_dim*(m_dim+1)){
        assert(other.rows() == other.cols()); //must be a square matrix
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = i; j < cols(); ++j)
                this->operator()(i,j) = other(i,j);
    }
    Matrix& operator=(const Matrix<T,2> &other){
        assert(other.rows() == other.cols()); //must be a square matrix
        m_dim = other.rows();
        m_desc = other.descriptor();
        m_elems.resize(0.5*m_dim*(m_dim+1));
        for (size_t i = 0; i < rows(); ++i)
            for (size_t j = i; j < cols(); ++j)
                this->operator()(i,j) = other(i,j);

        return *this;
    }
    Matrix& operator =(T val){m_elems = val; return *this;}

    size_t rows() const{return m_dim;}
    size_t cols() const{return m_dim;}

    auto begin(){return std::begin(m_elems);}
    auto begin() const{return std::cbegin(m_elems);}
    auto end(){return std::end(m_elems);}
    auto end() const{return std::cend(m_elems);}

    T* data(){return std::begin(m_elems);}
    const T* data() const{return std::begin(m_elems);}

    const Matrix_Slice<2>& descriptor() const{return m_desc;}
    const std::valarray<T>& values() const{return m_elems;}

    Matrix apply(T (func)(T val)){Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(const T&)) const{Matrix r(m_dim); std::transform(begin(),end(),r.begin(),func); return r;}
    //Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

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
        Matrix res(m_dim);
        res.m_elems = -m_elems;
        //std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        return res;
    }
    //Arithmetic operations.
    template<typename Scalar>
    enable_if_t<is_arithmetic_v<Scalar>,Matrix&> operator+=(const Scalar& val)
    {
        m_elems+=val;
        return *this;
    }
    template<typename Scalar>
    enable_if_t<is_arithmetic_v<Scalar>,Matrix&> operator-=(const Scalar& val)
    {
        m_elems-=val;
        return *this;
    }
    template<typename Scalar>
    enable_if_t<is_arithmetic_v<Scalar>,Matrix&> operator*=(const Scalar& val)
    {
        m_elems*=val;
        return *this;
    }
    template<typename Scalar>
    enable_if_t<is_arithmetic_v<Scalar>,Matrix&> operator/=(const Scalar& val)
    {
        //TODO valorar no permitir division por cero
        m_elems/=val;
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows()); //hermitian matrizes are squared
        m_elems+=m.values();
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,2,matrix_type> &m)
    {
        assert(rows() == m.rows()); //hermitian matrizes are squared
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    enable_if_t<is_arithmetic_v<U>,Matrix&> operator+=(const Matrix<U,2,MATRIX_TYPE::SYMM> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),[](const auto &v1,const auto &v2){return v1+v2;});
        return *this;
    }
    template<typename U>
    enable_if_t<is_arithmetic_v<U>,Matrix&> operator-=(const Matrix<U,2,MATRIX_TYPE::SYMM> &m)
    {
        assert(rows() == m.rows()); //symmetric matrizes are squared
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),[](const auto &v1,const auto &v2){return v1-v2;});
        return *this;
    }
};

#endif // HERM_MATRIX_H
