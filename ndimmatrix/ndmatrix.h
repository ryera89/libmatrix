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

//We must define MKL_Complex16 and MKL_Complex8 types before include the mkl.h header
#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>
//#define MKL_INT uint32_t
#include "mkl.h"

using std::vector;
using std::valarray;
using namespace std;
enum class MATRIX_TYPE{GEN,SYMM,HER,UTRI,LTRI,DIAG,CSR,SCSR,HCSR};
/* SCSR: symmetric sparse
 * HCSR: hermitian sparse */

template<typename T>
constexpr bool is_complex(){
    return is_same_v<T,complex<double>> || is_same_v<T,complex<float>> || is_same_v<T,complex<long double>>;
}
template<typename T>
constexpr bool is_number(){ //es complejo o real
    return is_arithmetic_v<T> || is_complex<T>();
}

//using MATRIX_TYPE::SYMM=SYMM;

template<typename T,size_t N,MATRIX_TYPE type = MATRIX_TYPE::GEN,
          typename = typename std::enable_if_t<is_number<T>() && ((N!=2 && type == MATRIX_TYPE::GEN) ||
                                                                 (N==2 && (((type == MATRIX_TYPE::HER || type == MATRIX_TYPE::HCSR) && is_complex<T>())
                                                                             || (type != MATRIX_TYPE::HER || type != MATRIX_TYPE::HCSR)))) >>
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
    Matrix(Matrix_Slice<N> desc):m_desc(desc),m_elems(m_desc.m_size){}

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
    const Matrix_Slice<N>& descriptor() const noexcept{return m_desc;}
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

    T* data(){return std::begin(m_elems);}
    const T* data() const{return std::begin(m_elems);}

    //Matrix apply(T (func)(T)){Matrix r(m_desc.m_extents); std::transform(begin(),end(),r.begin(),func); return r;}
    Matrix apply(T (func)(T)) const{
        Matrix r;
        r.m_desc = m_desc;
        r.m_elems = m_elems.apply(func);
        //std::transform(begin(),end(),r.begin(),func);
        return r;
    }
    Matrix apply(T (func)(const T&)) const{
        Matrix r;
        r.m_desc = m_desc;
        r.m_elems = m_elems.apply(func);
        //std::transform(begin(),end(),r.begin(),func);
        return r;
    }
    //Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

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
        Matrix res;
        res.m_desc = m_desc;
        res.m_elems = -m_elems;
        //std::for_each(res.begin(),res.end(),[](T &val){val = -val;});
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
        m_elems+=m.m_elems;
        return *this;
    }
    Matrix& operator-=(const Matrix<T,N> &m)
    {
        Matrix_Slice<N> mdesc = m.descriptor();
        assert(std::equal(m_desc.m_extents.begin(),m_desc.m_extents.end(),
                          mdesc.m_extents.begin(),mdesc().m_extents.end()));
        m_elems-=m.m_elems;
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

    Matrix(size_t m,size_t n):m_desc(0,m,n),m_elems(m_desc.m_size){}
    Matrix(size_t m,size_t n,T val):m_desc(0,m,n),m_elems(val,m_desc.m_size){}
    Matrix(Matrix_Slice<2> desc):m_desc(desc),m_elems(m_desc.m_size){}
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

    T* data(){return std::begin(m_elems);}
    const T* data() const{return std::begin(m_elems);}

    Matrix apply(T (func)(T val)){
        Matrix r;
        r.m_desc = m_desc;
        r.m_elems = m_elems.apply(func);
        //std::transform(begin(),end(),r.begin(),func);
        return r;
    }
    Matrix apply(T (func)(const T&)) const{
        Matrix r;
        r.m_desc = m_desc;
        r.m_elems = m_elems.apply(func);
        //std::transform(begin(),end(),r.begin(),func);
        return r;
    }

    //Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

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
        Matrix r;//(rows(),cols());
        r.m_desc = m_desc;
        r.m_elems = -m_elems;
        //std::transform(this->begin(),this->end(),r.begin(),[](const T &val){return -val;});
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
        m_elems+=m.m_elems;
        return *this;
    }
    Matrix& operator-=(const Matrix<T,2> &m)
    {
        assert(rows() == m.rows() && cols() == m.cols());
        m_elems-=m.m_elems;
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
    std::enable_if_t<std::is_arithmetic_v<T> || std::is_same_v<T,std::complex<double>>
                         || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long double>>,double>
    norm2() const{
        if constexpr (std::is_arithmetic_v<T>){
            double res = std::inner_product(begin(),end(),begin(),0.0);
            return std::sqrt(res);
        }else{
            double res = std::accumulate(begin(),end(),0.0,[](double init,T val)->double{return std::move(init) + norm(val);});
            return std::sqrt(res);
        }

    }
    std::enable_if_t<std::is_arithmetic_v<T> || std::is_same_v<T,std::complex<double>>
                         || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long double>>,double>
    norm2sqr() const{
        if constexpr (std::is_arithmetic_v<T>){
            return std::inner_product(begin(),end(),begin(),0.0);
            //return std::sqrt(res);
        }else{
            return std::accumulate(begin(),end(),0.0,[](double init,T val)->double{return std::move(init) + norm(val);});
            //return std::sqrt(res);
        }

    }
};
template<typename T>
class Matrix<T,1,MATRIX_TYPE::GEN>{
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
    Matrix(Matrix_Slice<1> desc):m_desc(desc),m_elems(m_desc.m_size){}
    //Matrix(const std::array<size_t,1> &exts):m_desc{0,exts},m_elems(m_desc.m_size){}

    template<typename U>
    Matrix(const Matrix_Ref<U,1> &ref):m_desc(0,ref.descriptor().m_extents[0]),
                                                       m_elems(m_desc.m_size)
    {
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        for (size_t i = 0; i < m_desc.m_extents[0]; ++i) m_elems[i] = ref(i);
    }
    template<typename U>
    Matrix& operator = (const Matrix_Ref<T,1> &ref){
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        m_desc.m_start = 0;
        m_desc.m_extents = ref.descriptor().m_extents;
        m_desc.init();
        m_elems.resize(m_desc.m_size);
        for (size_t i = 0; i < m_desc.m_extents[0]; ++i) m_elems[i] = ref(i);
        return *this;
    }
    Matrix& operator=(const T &val){
        //std::for_each(begin(),end(),[&val](T &elem){elem = val;});
        m_elems = val;
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

    T* data(){return std::begin(m_elems);}
    const T* data() const{return std::begin(m_elems);}

    //Arithmetic ops
    Matrix apply(T (func)(T val)){
        Matrix r;
        r.m_desc = m_desc;
        //std::transform(begin(),end(),r.begin(),func);
        r.m_elems = m_elems.apply(func);
        return r;
    }
    Matrix apply(T (func)(const T&)) const{
        Matrix r;
        r.m_desc = m_desc;
        //std::transform(begin(),end(),r.begin(),func);
        r.m_elems = m_elems.apply(func);
        return r;
    }
    //Matrix& apply(void (func)(T&)){std::for_each(begin(),end(),func); return *this;}

    //Unary minus
    Matrix operator-() const{
        Matrix res(this->size());
        //std::transform(this->begin(),this->end(),res.begin(),[](const T &elem){return -elem;});
        res.m_elems = -m_elems;
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
    Matrix& operator+=(const Matrix &m)
    {
        assert(this->size() == m.size());
        m_elems+=m.m_elems;
        return *this;
    }
    Matrix& operator-=(const Matrix &m)
    {
        assert(this->size() == m.size());
        m_elems-=m.m_elems;
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator+=(const Matrix<U,1> &m)
    {
        assert(this->size() == m.size());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1+val2;});
        return *this;
    }
    template<typename U>
    std::enable_if_t<std::is_convertible_v<U,T>,Matrix&> operator-=(const Matrix<U,1> &m)
    {
        assert(this->size() == m.size());
        std::transform(this->begin(),this->end(),m.begin(),this->begin(),
                       [](const auto &val1,const auto &val2){return val1-val2;});
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
    std::enable_if_t<std::is_arithmetic_v<T> || std::is_same_v<T,std::complex<double>>
                         || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long double>>,double>
    norm2() const{
        if constexpr (std::is_arithmetic_v<T>){
            double res = std::inner_product(begin(),end(),begin(),0.0);
            return std::sqrt(res);
        }else{
            double res = std::accumulate(begin(),end(),0.0,[](double init,T val)->double{return std::move(init) + norm(val);});
            return std::sqrt(res);
        }

    }
    std::enable_if_t<std::is_arithmetic_v<T> || std::is_same_v<T,std::complex<double>>
                         || std::is_same_v<T,std::complex<float>> || std::is_same_v<T,std::complex<long double>>,double>
    norm2sqr() const{
        if constexpr (std::is_arithmetic_v<T>){
            return std::inner_product(begin(),end(),begin(),0.0);
            //return std::sqrt(res);
        }else{
            return std::accumulate(begin(),end(),0.0,[](double init,T val)->double{return std::move(init) + norm(val);});
            //return std::sqrt(res);
        }

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

#endif // NDMATRIX_H
