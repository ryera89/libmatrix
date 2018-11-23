#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "matrix_impl.h"
#include <cmath>

template<typename T,size_t N>
class Matrix
{
private:
    matrix_impl::Matrix_Slice<N> _desc;
    std::vector<T> _elems;
public:
    //Common aliases
    static constexpr size_t order = N;
    using value_type = T;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

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
    Matrix(Exts... exts):_desc(0,{exts...}),_elems(_desc._size){}

    template<typename U>
    Matrix(const Matrix<U,N> &other):_desc(other.descriptor()),_elems{other.begin(),other.end()}{
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
    }

    template<typename U>
    Matrix& operator = (const Matrix<U,N> &other){
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        _desc = other._desc;
        _elems.assign(other.begin(),other.end());
        return *this;

    }

    template<typename U>
    Matrix(const matrix_impl::Matrix_Ref<U,N> &ref){
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        _desc._start = 0;
        _desc._extents = ref.descriptor()._extents;
        _desc.init();
        _elems.resize(_desc._size);
        matrix_impl::Matrix_Ref<T,N> mref{data(),_desc};
        matrix_impl::assing_slice_vals(ref,mref);
    }
    template<typename U>
    Matrix& operator = (const matrix_impl::Matrix_Ref<U,N> &ref){
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        _desc._start = 0;
        _desc._extents = ref.descriptor()._extents;
        _desc.init();
        _elems.resize(_desc._size);
        matrix_impl::Matrix_Ref<T,N> mref{data(),_desc};
        matrix_impl::assing_slice_vals(ref,mref);
        return *this;
    }

    //Construction and assignment from nested initializars
    Matrix(matrix_impl::Matrix_Initializer<T,N> init){
        matrix_impl::derive_extents(_desc._extents,init);
        _desc.init();    //Strides determination from extents
        _elems.reserve(_desc._size);
        matrix_impl::insert_flat(init,_elems);
        assert(_elems.size() == _desc._size);
    }
    Matrix& operator=(matrix_impl::Matrix_Initializer<T,N> init){
        matrix_impl::derive_extents(_desc._extents,init);
        _elems.reserve(_desc._size);
        matrix_impl::insert_flat(init,_elems);
        assert(_elems.size() == _desc._size);
        return *this;
    }

    Matrix& operator=(const T &val){
        std::fill(begin(),end(),val);
        //std::for_each(begin(),end(),[&val](T &elem){elem = val;});
        return *this;
    }


    //Disable use of {} for extents
    template<typename U>
    Matrix(std::initializer_list<U>) = delete;
    template<typename U>
    Matrix& operator=(std::initializer_list<U>) = delete;

    size_t extent(size_t n) const{
        assert(n < N);
        return _desc._extents[n];
    }

    template<typename F>
    Matrix<T,N>& apply(F fun){
        std::for_each(begin(),end(),fun);
        return *this;
    }

    //Unary minus
    Matrix<T,N> operator-() const
    {
        Matrix<T,N> res(*this);
        std::for_each(res.begin(),res.end(),[](T &elem){elem = -elem;});
        return res;
    }
    //Arithmetic operations.
    template<typename Scalar>
    Matrix& operator+=(const Scalar& val)
    {
        std::for_each(begin(),end(),[&val](T &elem){elem+=val;});
        return *this;
    }
    template<typename Scalar>
    Matrix& operator-=(const Scalar& val)
    {
        std::for_each(begin(),end(),[&val](T &elem){elem-=val;});
        return *this;
    }
    template<typename Scalar>
    Matrix& operator*=(const Scalar& val)
    {
        std::for_each(begin(),end(),[&val](T &elem){elem*=val;});
        return *this;
    }
    template<typename Scalar>
    Matrix& operator/=(const Scalar& val)
    {
        std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
        return *this;
    }
    template<typename U>
    Matrix& operator+=(const Matrix<U,N> &m)
    {
        assert(this->size() == m.size());
        std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,N> &m)
    {
        assert(this->size() == m.size());
        std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
        return *this;
    }

    size_t rows() const {return extent(0);}
    size_t columns() const {return extent(1);}

    size_t size() const {return _desc._size;}

    const matrix_impl::Matrix_Slice<N>& descriptor() const{
        return _desc;
    }

    iterator begin(){return _elems.begin();}
    const_iterator begin() const{return _elems.cbegin();}

    iterator end(){return _elems.end();}
    const_iterator end() const{return _elems.cend();}

    T* data() {return _elems.data();}
    const T* data() const {return _elems.data();}

    std::conditional_t<(N>=2),matrix_impl::Matrix_Ref<T,N-1>,T&>
    operator[](size_t i){
        if constexpr (N>=2){
            return row(i);
        }else{
            return _elems[i];
        }
    }
    std::conditional_t<(N>=2),matrix_impl::Matrix_Ref<const T,N-1>,const T&>
    operator[](size_t i) const{
        if constexpr (N>=2){
            return row(i);
        }else{
            return _elems[i];
        }
    }

    std::conditional_t<(N>=2),matrix_impl::Matrix_Ref<T,N-1>,T&>
    row(size_t i){
        if constexpr (N>=2){
            matrix_impl::Matrix_Slice<N-1> row;
            matrix_impl::slice_dim<0>(i,_desc,row);
            return {data(),row};
        }else{
            return _elems[i];
        }
    }
    std::conditional_t<(N>=2),matrix_impl::Matrix_Ref<const T,N-1>,const T&>
    row(size_t i) const{
        if constexpr (N>=2){
            matrix_impl::Matrix_Slice<N-1> row;
            matrix_impl::slice_dim<0>(i,_desc,row);
            return {data(),row};
        }else{
            return _elems[i];
        }

    }


    //matrix_impl::Matrix_Ref<const T,N-1> row(size_t i) const;
    /*template<typename Ref = matrix_impl::Matrix_Ref<T,N-1>>
    std::enable_if_t<(N>=2),Ref>
    column(size_t i){
        matrix_impl::Matrix_Slice<N-1> col;
        matrix_impl::slice_dim<1>(i,_desc,col);
        return {data(),col};
    }
    template<typename Ref = matrix_impl::Matrix_Ref<const T,N-1>>
    std::enable_if_t<(N>=2),Ref>
    column(size_t i) const{
        matrix_impl::Matrix_Slice<N-1> col;
        matrix_impl::slice_dim<1>(i,_desc,col);
        return {data(),col};
    }*/
    //matrix_impl::Matrix_Ref<T,N-1> column(size_t i);
    //matrix_impl::Matrix_Ref<const T,N-1> column(size_t i) const;

    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(), T&>
    operator()(Args... args){
        assert(matrix_impl::check_bounds(_desc,args...));
        return *(data() + _desc(args...));
    }
    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(),const T&>
    operator()(Args... args) const{
        assert(matrix_impl::check_bounds(_desc,args...));
        return *(data() + _desc(args...));
    }

    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_slice<Args...>(),matrix_impl::Matrix_Ref<T,N>>
    operator()(Args... args){
         matrix_impl::Matrix_Slice<N> d;
         d._size = 1;
         d._start = matrix_impl::do_slice(_desc,d,args...);
         return {data(),d};
    }

    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_slice<Args...>(),matrix_impl::Matrix_Ref<const T,N>>
    operator()(Args... args) const{
         matrix_impl::Matrix_Slice<N> d;
         d._size = 1;
         matrix_impl::do_slice(_desc,d,args...);
         return {data(),d};
    }

};

    template<typename T>
    class Matrix<T,1>{
        matrix_impl::Matrix_Slice<1> _desc;
        std::vector<T> _elems;
    public:
        static constexpr size_t order = 1;
        using value_type = T;
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        //Default constructor and destructor
        Matrix() = default;
        ~Matrix() = default;

        //Move constructor and assignment
        Matrix(Matrix&&) = default;
        Matrix& operator=(Matrix&&) = default;

        //Copy constructor and assignment
        Matrix(const Matrix&) = default;
        Matrix& operator=(const Matrix&) = default;

        Matrix(size_t ext):_desc(0,ext),_elems(ext){}
        Matrix(const std::array<size_t,1> &exts):_desc{0,exts},_elems(_desc._size){}

        template<typename U>
        Matrix(const Matrix<U,1> &other):_desc(other.descriptor()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }

        template<typename U>
        Matrix& operator = (const Matrix<U,1> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _desc = other._desc;
            _elems.assign(other.begin(),other.end());
            return *this;

        }

        //TODO ver que hacer aqui
        template<typename U>
        Matrix(const matrix_impl::Matrix_Ref<U,1> &ref){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _desc._start = 0;
            _desc._extents = ref.descriptor()._extents;
            _desc.init();
            _elems.resize(_desc._size);
            _elems.assign(ref.begin,ref.end());
            //matrix_impl::Matrix_Ref<T,N> mref{data(),_desc};
            //matrix_impl::assing_slice_vals(ref,mref);
        }
        template<typename U>
        Matrix& operator = (const matrix_impl::Matrix_Ref<U,1> &ref){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _desc._start = 0;
            _desc._extents = ref.descriptor()._extents;
            _desc.init();
            _elems.resize(_desc._size);
            _elems.assign(ref.begin,ref.end());
            //matrix_impl::Matrix_Ref<T,N> mref{data(),_desc};
            //matrix_impl::assing_slice_vals(ref,mref);
            return *this;
        }

//        //Construction and assignment from nested initializars
//        Matrix(matrix_impl::Matrix_Initializer<T,1> init){
//            matrix_impl::derive_extents(_desc._extents,init);
//            _desc.init();    //Strides determination from extents
//            _elems.reserve(_desc._size);
//            matrix_impl::insert_flat(init,_elems);
//            assert(_elems.size() == _desc._size);
//        }
//        Matrix& operator=(matrix_impl::Matrix_Initializer<T,1> init){
//            matrix_impl::derive_extents(_desc._extents,init);
//            _elems.reserve(_desc._size);
//            matrix_impl::insert_flat(init,_elems);
//            assert(_elems.size() == _desc._size);
//            return *this;
//        }

        Matrix& operator=(const T &val){
            std::for_each(begin(),end(),[&val](T &elem){elem = val;});
            return *this;
        }


        //Disable use of {} for extents
        //template<typename U>
        Matrix(std::initializer_list<T> list):_elems(list){
           _desc._start = 0;
           _desc._size = _elems.size();
           _desc._extents[0] = _elems.size();
           _desc._strides[0] = 1;
        }

        //template<typename U>
        Matrix& operator=(std::initializer_list<T> list){
            _elems = list;
            _desc._start = 0;
            _desc._size = _elems.size();
            _desc._extents[0] = _elems.size();
            _desc._strides[0] = 1;
            return *this;
        }

        //Arithmetic ops
        template<typename F>
        Matrix<T,1>& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        template<typename F>
        Matrix<T,1>& apply(const Matrix<T,1> &vec,F fun){
            std::transform(vec.begin(),vec.end(),this->begin(),fun);
            return *this;
        }

        //Unary minus
        Matrix<T,1> operator-() const{
            Matrix<T,1> res(*this);
            std::for_each(res.begin(),res.end(),[](T &elem){elem = -elem;});
            return res;
        }

        //Arithmetic operations.
        template<typename Scalar>
        Matrix& operator+=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](T &elem){elem+=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator-=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](T &elem){elem-=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator*=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](T &elem){elem*=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator/=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,1> &m)
        {
            assert(this->size() == m.size());
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,1> &m)
        {
            assert(this->size() == m.size());
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }

        size_t size() const {return _elems.size();}
        size_t rows() const {return size();}

        const matrix_impl::Matrix_Slice<1>& descriptor() const{
            return _desc;
        }

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        T& operator()(const size_t &i)
        {
            assert(i < size());
            return _elems[i];
        }
        const T& operator()(const size_t &i) const
        {
            assert(i < size());
            return _elems[i];
        }
        matrix_impl::Matrix_Ref<T,1> operator()(const matrix_impl::Slice &s){
             matrix_impl::Matrix_Slice<1> d;
             d._start = s._start;
             d._size = s._length;
             d._extents[0] = s._length;
             d._strides[0] = s._stride;
             //d._start = matrix_impl::do_slice(_desc,d,args...);
             return {data(),d};
        }

        matrix_impl::Matrix_Ref<const T,1> operator()(const matrix_impl::Slice &s) const{
             matrix_impl::Matrix_Slice<1> d;
            d._start = s._start;
            d._size = s._length;
            d._extents[0] = s._length;
            d._strides[0] = s._stride;
            //d._start = matrix_impl::do_slice(_desc,d,args...);
            return {data(),d};
        }

        template<typename RT = double>
        std::enable_if_t<std::is_arithmetic_v<T>,RT>
        norm() const{
            RT res = std::inner_product(begin(),end(),begin(),RT(0.0));
            return std::sqrt(res);
        }
        template<typename RT = double>
        std::enable_if_t<std::is_arithmetic_v<T>,RT>
        square_norm() const{
            RT res = std::inner_product(begin(),end(),begin(),RT(0.0));
            return res;
        }
    };

    typedef Matrix<double,1> VecDoub;
    typedef Matrix<double,2> MatDoub;

    template<typename T>
    class Matrix<T,0>{
        T elem;
    public:
        static constexpr size_t order = 0;
        using value_type = T;

        Matrix(const T &val):elem(val){}
        Matrix& operator=(const T &val){
            elem = val;
            return *this;
        }

        T& row(size_t i) = delete;

        T& operator()(){return elem;}
        const T& operator()()const {return elem;}

        //Conversion operator Matrix<T,0> to T
        operator T&(){return elem;}
        operator const T&(){return elem;}

    };


    //TODO: Think about type asserts
    //Matrix Binary Arithmetic Operations
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator+(const Scalar &val,const Matrix<T,N> &m){
        Matrix<RT,N> res(m);
        res+=val;
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator+(const Matrix<T,N> &m,const Scalar &val){
        Matrix<RT,N> res(m);
        res+=val;
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator-(const Matrix<T,N> &m,const Scalar &val){
        Matrix<RT,N> res(m);
        res-=val;
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator-(const Scalar &val,const Matrix<T,N> &m){
        Matrix<RT,N> res(m.descriptor()._extents);
        std::transform(m.begin(),m.end(),res.begin(),[&val](const T &elem){return val-elem;});
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator*(const Scalar &val,const Matrix<T,N> &m){
        Matrix<RT,N> res(m);
        res*=val;
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator*(const Matrix<T,N> &m,const Scalar &val){
        Matrix<RT,N> res(m);
        res*=val;
        return res;
    }
    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
    inline Matrix<RT,N> operator/(const Matrix<T,N> &m,const Scalar &val){
        Matrix<RT,N> res(m);
        res/=val;
        return res;
    }
    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>,size_t N>
    inline Matrix<RT,N> operator+(const Matrix<T1,N> &m1,const Matrix<T2,N> &m2){
        Matrix<RT,N> res(m1);
        res+=m2;
        return res;
    }
    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>,size_t N>
    inline Matrix<RT,N> operator-(const Matrix<T1,N> &m1,const Matrix<T2,N> &m2){
        Matrix<RT,N> res(m1);
        res-=m2;
        return res;
    }
    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>>
    inline Matrix<RT,1> operator *(const Matrix<T1,2> &m2,const Matrix<T2,1> m1){
        assert(m2.columns() == m1.size());
        Matrix<RT,1> res(m2.rows());
        for (size_t i = 0; i < m2.rows(); ++i){
            res(i) = std::inner_product(m2.row(i).begin(),m2.row(i).end(),m1.begin(),0);
        }
        return res;
    }



    //TODO optimizar estas operaciones y para tipos genericos

    inline double operator*(const matrix_impl::Matrix_Ref<double,1> &mref,const Matrix<double,1> &v){
        assert(mref.size() == v.size());

        double temp = 0.0;
        for (size_t i = 0; i < v.size(); ++i)  temp+=mref(i)*v(i);

        return temp;
    }

    inline double operator *(const VecDoub &u,const VecDoub &v){
        assert(u.size() == v.size());
        return std::inner_product(u.begin(),u.end(),v.begin(),0.0);
    }
    inline VecDoub operator *(const MatDoub &M,const VecDoub &v){
        assert(M.columns() == v.size());

        VecDoub r(M.rows());

        for (size_t i = 0; i < M.rows(); ++i){
            double tmp = 0.0;
            for (size_t j = 0; j < M.columns(); ++j){
                tmp += M(i,j)*v(j);
            }
            r(i) = tmp;
        }
        return r;
    }
    inline VecDoub operator *(const VecDoub &v,const MatDoub &M){
        assert(M.rows() == v.size());

        VecDoub r(M.columns());

        for (size_t i = 0; i < M.columns(); ++i){
            double tmp = 0.0;
            for (size_t j = 0; j < M.rows(); ++j){
                tmp += v(j)*M(j,i);
            }
            r(i) = tmp;
        }
        return r;
    }
    inline MatDoub operator*(const MatDoub &M1,const MatDoub &M2){

        assert(M1.columns() == M2.rows());

        const size_t m = M1.rows();
        const size_t n = M2.columns();
        const size_t p = M1.columns();
        MatDoub R(m,n);

        for (size_t i = 0; i < m ; ++i){
            for (size_t j = 0; j < n; ++j){
                double temp = 0.0;
                for (size_t k = 0; k < p; ++k){
                    temp+=M1(i,k)*M2(k,j);
                }
                R(i,j) = temp;
            }
        }
        return R;
    }

    inline MatDoub diadic_product(const VecDoub &v1,const VecDoub &v2){
        MatDoub R(v1.size(),v2.size());

        for (size_t i = 0; i < v1.size(); ++i){
            for (size_t j = 0; j < v2.size(); ++j){
                R(i,j) = v1(i)*v2(j);
            }
        }
        return R;
    }
    inline Matrix<double,4> diadic_product(const VecDoub &v1,const VecDoub &v2,const VecDoub &v3,const VecDoub &v4){
        Matrix<double,4> R(v1.size(),v2.size(),v3.size(),v4.size());

        for (size_t i = 0; i < v1.size(); ++i){
            for (size_t j = 0; j < v2.size(); ++j){
                for (size_t k = 0; k < v3.size(); ++k){
                    for (size_t l = 0; l < v4.size(); ++l){
                        R(i,j,k,l) = v1(i)*v2(j)*v3(k)*v4(l);
                    }
                }

            }
        }
        return R;
    }
    /***********************************************************/
    //Para contraccion de tensores (in construction)
    //Tensor reduction facilities
    //TODO implementar para tipos genericos

    /*template<size_t N>
    void tensor_index_contraction(const Matrix_Ref<double,N> &imref,const VecDoub &v,Matrix_Ref<double,N-1> &omres){
        if constexpr (N==1){
            omres() = imref*v;
        }
    }*/

    /********************************************************/
    //OS operations
    //TODO: enable_if T tiene ostream definido
    template<typename T>
inline  std::ostream& operator << (std::ostream& os,const Matrix<T,2> &m){
        //std::ios_base::fmtflags ff = std::ios::scientific;
        //ff |= std::ios::showpos;
        //os.setf(ff);
        for (size_t i = 0; i < m.rows(); ++i){
            for (size_t j = 0; j < m.columns(); ++j)
                os << m(i,j) << '\t' ;
            os << '\n';
        }
        //os.unsetf(ff);
        return os;
    }
    template<typename T>
    std::ostream& operator << (std::ostream& os,const Matrix<T,1> &m);/*{
        std::ios_base::fmtflags ff = std::ios::scientific;
        ff |= std::ios::showpos;
        os.setf(ff);
        for (size_t i = 0; i < m.size(); ++i){
            os << m(i) << '\n';
        }
        os.unsetf(ff);
        return os;
    }*/
    template<typename T>
    std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,2> &m);/*{
        if (ofs.is_open()){
            std::ios_base::fmtflags ff = std::ios::scientific;
            ff |= std::ios::showpos;
            ofs.setf(ff);
            for (size_t i = 0; i < m.rows(); ++i){
                for (size_t j = 0; j < m.cols(); ++j)
                    ofs << m(i,j) << '\t' ;
                ofs << std::endl;
            }
            ofs.unsetf(ff);
        }
        return ofs;
    }*/
    template<typename T>
    std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,1> &m);/*{
        if (ofs.is_open()){
            std::ios_base::fmtflags ff = std::ios::scientific;
            ff |= std::ios::showpos;
            ofs.setf(ff);
            for (size_t i = 0; i < m.size(); ++i){
                ofs << i << "**" << m(i) << '\n';
            }
            ofs.unsetf(ff);
        }
        return ofs;
    }*/
    template<typename T>
    inline std::ostream& operator << (std::ostream& os,const Matrix<T,3> &m){
        //if (ofs.is_open()){
            //std::ios_base::fmtflags ff = std::ios::scientific;
            //ff |= std::ios::showpos;
            //os.setf(ff);
            for (size_t i = 0; i < m.extent(2); ++i){
                os << "(:,:," << i << ") \n";
                for (size_t j = 0; j < m.extent(0); ++j){
                    for (size_t k = 0; k < m.extent(1); ++k){

                            os << m(j,k,i) << '\t' ;
                    }

                    os << "\n";
                }
            }
            //os.unsetf(ff);
            return os;
        //}
    }
    template<typename T>
    std::ostream& operator << (std::ostream& os,const Matrix<T,4> &m);/*{
        //if (ofs.is_open()){
            std::ios_base::fmtflags ff = std::ios::scientific;
            ff |= std::ios::showpos;
            os.setf(ff);
            for (size_t i = 0; i < m.extent(3); ++i){
                for (size_t j = 0; j < m.extent(2); ++j){
                    os << "(:,:," << j << ", " << i << ") \n";
                    for (size_t k = 0; k < m.extent(0); ++k){
                        for (size_t l = 0; l < m.extent(1); ++l){
                            os << m(k,l,j,i) << '\t' ;
                        }
                        os << "\n";
                    }
                }
            }
            os.unsetf(ff);
            return os;
        //}
    }*/



#endif // MATRIX_H
