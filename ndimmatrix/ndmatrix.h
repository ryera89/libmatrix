#ifndef NDMATRIX_H
#define NDMATRIX_H

#include <valarray>
#include <vector>
#include <complex>
#include <type_traits>
#include "matrix_ref.h"
//#include <iostream>
//#include <iomanip>
//#include <map>

//We must define MKL_Complex16 type before include the mkl.h header
//#define MKL_Complex16 std::complex<double>
//#define MKL_INT uint32_t
#include "mkl.h"

using std::vector;
using std::valarray;
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
    valarray<T> m_elems;
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
    Matrix(Matrix_Initializer<T,N> init);
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
    Matrix_Slice<N> descriptor() const noexcept{return m_desc;}
    size_t size() const noexcept {return m_desc.m_size;} //element number
    size_t extent(size_t n) const{ //number of elements in dim = n
        assert(n < N);
        return m_desc.m_extents[n];
    }
    const valarray<T>& values() const{return m_elems;}
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
        Matrix res(*this);
        std::for_each(res.begin(),res.end(),[](T &val){val = -val;});
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
        Matrix_Slice<N> mdesc = m.descriptor();
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          mdesc.m_extents.begin(),mdesc().m_extents.end()));
        m_elems+=m.values();
        return *this;
    }
    Matrix& operator-=(const Matrix<T,N> &m)
    {
        Matrix_Slice<N> mdesc = m.descriptor();
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          mdesc.m_extents.begin(),mdesc().m_extents.end()));
        m_elems-=m.values();
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator+=(const Matrix<U,N> &m)
    {
        Matrix_Slice<N> mdesc = m.descriptor();
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          mdesc.m_extents.begin(),mdesc().m_extents.end()));

        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const T &val1,const U &val2){return val1+val2;});
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator-=(const Matrix<U,N> &m)
    {
        Matrix_Slice<N> mdesc = m.descriptor();
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          mdesc.m_extents.begin(),mdesc().m_extents.end()));
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const T &val1,const U &val2){return val1-val2;});
        return *this;
    }
};

template<typename T>
class Matrix<T,2,MATRIX_TYPE::GEN>{
private:
    Matrix_Slice<2> m_desc;
    valarray<T> m_elems;
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
        return this;
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
    const valarray<T>& values() const{return m_elems;}

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
    Matrix<T,2> operator()(const valarray<uint32_t> &indx1,const valarray<uint32_t> &indx2) const{
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
        std::transform(this->begin(),this->end(),r.begin(),[](const T &val){return -val;});
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

//IO operations
template<typename T,MATRIX_TYPE mtype>
std::ostream &operator << (std::ostream &os,const Matrix<T,2,mtype> &m){
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

#endif // NDMATRIX_H
