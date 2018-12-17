#ifndef MATRIX_H
#define MATRIX_H

#include <ostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "matrix_impl.h"
#include <cmath>
#include <complex>
#include "mkl.h"

enum class Matrix_Type{GEN,SYMM,HER,UTR,LTR};
//GEN:General, SYMM:symmetric, HER:hemitian, UTR:upper_triangular, LTR: lower triangular, SPRC:sparce
enum class Matrix_Storage_Scheme{FULL,UPP,LOW,CSR3};
//FULL: full storage //UPP:package upper triangular storage //LOW:package lower triangular storage

template<typename T,size_t N,Matrix_Type type,Matrix_Storage_Scheme strg_sch>
class Matrix{};

template<typename T,size_t N,Matrix_Type mtype>
class Matrix<T,N,mtype,Matrix_Storage_Scheme::FULL>{
private:
    matrix_impl::Matrix_Slice<N> _desc;
    std::vector<T> _elems;
public:
    //Common aliases
    static constexpr size_t order = N;
    static constexpr Matrix_Type type = mtype;
    static constexpr Matrix_Storage_Scheme storage = Matrix_Storage_Scheme::FULL;
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
    Matrix(const Matrix<U,N,type,storage> &other):_desc(other.descriptor()),_elems{other.begin(),other.end()}{
        static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
    }

    template<typename U>
    Matrix& operator = (const Matrix<U,N,type,storage> &other){
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
    Matrix& apply(F fun){
        std::for_each(begin(),end(),fun);
        return *this;
    }

    //Unary minus
    Matrix<T,N,type,storage> operator-() const
    {
        Matrix<T,N,type,storage> res(*this);
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
    Matrix& operator+=(const Matrix<U,N,type,storage> &m)
    {
        assert(this->size() == m.size());
        std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
        return *this;
    }
    template<typename U>
    Matrix& operator-=(const Matrix<U,N,type,storage> &m)
    {
        assert(this->size() == m.size());
        std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
        return *this;
    }

    size_t size() const noexcept {return _desc._size;}

    const matrix_impl::Matrix_Slice<N>& descriptor() const noexcept{
        return _desc;
    }
    iterator begin(){return _elems.begin();}
    const_iterator begin() const{return _elems.cbegin();}

    iterator end(){return _elems.end();}
    const_iterator end() const{return _elems.cend();}

    T* data() {return _elems.data();}
    const T* data() const {return _elems.data();}

    matrix_impl::Matrix_Ref<T,N-1> row(size_t i){
        matrix_impl::Matrix_Slice<N-1> row;
        matrix_impl::slice_dim<0>(i,_desc,row);
        return {data(),row};
    }
    matrix_impl::Matrix_Ref<const T,N-1> row(size_t i) const{
        matrix_impl::Matrix_Slice<N-1> row;
        matrix_impl::slice_dim<0>(i,_desc,row);
        return {data(),row};
    }
    matrix_impl::Matrix_Ref<T,N-1> operator[](size_t i){
        return row(i);
    }
    matrix_impl::Matrix_Ref<const T,N-1>
    operator[](size_t i) const{
        return row(i);
    }
    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(), T&>
    operator()(Args... args){
        assert(matrix_impl::check_bounds(_desc,args...));
        return _elems[_desc(args...)];
    }
    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(),const T&>
    operator()(Args... args) const{
        assert(matrix_impl::check_bounds(_desc,args...));
        return _elems[_desc(args...)];
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

    //TODO enable if constructors para los tipos de matrices
    template<typename T,Matrix_Type mtype>
    class Matrix<T,2,mtype,Matrix_Storage_Scheme::FULL>{
         matrix_impl::Matrix_Slice<2> _desc;
         std::vector<T> _elems;
    public:
         //Common aliases
         static constexpr size_t order = 2;
         static constexpr Matrix_Type type = mtype;
         static constexpr Matrix_Storage_Scheme storage = Matrix_Storage_Scheme::FULL;
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

         explicit Matrix(size_t m,size_t n):_desc(0,m,n),_elems(m*n){}
         template<typename U>
         Matrix(const Matrix<U,2,type,storage> &other):_desc(other.descriptor()),_elems{other.begin(),other.end()}{
             static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
         }
         template<typename U>
         Matrix& operator = (const Matrix<U,2,type,storage> &other){
             static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
             _desc = other._desc;
             _elems.assign(other.begin(),other.end());
             return *this;

         }
         //convertir de Matrix_Ref a Matrix
         template<typename U>
         Matrix(const matrix_impl::Matrix_Ref<U,2> &ref):_desc(0,ref.descriptor()._extents[0],ref.descriptor()._extents[1]),
         _elems(_desc._extents[0]*_desc._extents[1])
         {
             static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
             for (size_t i = 0; i < rows(); ++i){
                 for (size_t j = 0; j < cols(); ++j){
                     _elems[i*cols() + j] = ref(i,j);
                 }
             }
         }
         template<typename U>
         Matrix& operator=(const matrix_impl::Matrix_Ref<U,2> &ref){
             static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
             _desc._start = 0;
             _desc._extents = ref.descriptor()._extents;
             _desc.init();
             _elems.resize(size());
             for (size_t i = 0; i < rows(); ++i){
                 for (size_t j = 0; j < cols(); ++j){
                     _elems[i*cols() + j] = ref(i,j);
                 }
             }
         }
         //Construction and assignment from nested initializars
         Matrix(matrix_impl::Matrix_Initializer<T,2> init){
             matrix_impl::derive_extents(_desc._extents,init);
             _desc.init();    //Strides determination from extents
             _elems.reserve(_desc._size);
             matrix_impl::insert_flat(init,_elems);
             assert(_elems.size() == _desc._size);
         }
         Matrix& operator=(matrix_impl::Matrix_Initializer<T,2> init){
             matrix_impl::derive_extents(_desc._extents,init);
             _elems.reserve(_desc._size);
             matrix_impl::insert_flat(init,_elems);
             assert(_elems.size() == _desc._size);
             return *this;
         }
         Matrix& operator=(const T &val){
             std::fill(begin(),end(),val);
             return *this;
         }
         //Disable use of {} for extents
         template<typename U>
         Matrix(std::initializer_list<U>) = delete;
         template<typename U>
         Matrix& operator=(std::initializer_list<U>) = delete;

         size_t extent(size_t n) const{
             assert(n < 2);
             return _desc._extents[n];
         }
         size_t rows() const noexcept {return extent(0);}
         size_t cols() const noexcept {return extent(1);}

         size_t size() const noexcept {return _desc._size;}

         const matrix_impl::Matrix_Slice<2>& descriptor() const noexcept{
             return _desc;
         }

         iterator begin(){return _elems.begin();}
         const_iterator begin() const{return _elems.cbegin();}

         iterator end(){return _elems.end();}
         const_iterator end() const{return _elems.cend();}

         T* data() {return _elems.data();}
         const T* data() const {return _elems.data();}

         //Arithmetic ops
         template<typename F>
         Matrix& apply(F fun){
             std::for_each(begin(),end(),fun);
             return *this;
         }
         //Unary minus
         Matrix<T,2,type,storage> operator-() const
         {
             Matrix<T,2,type,storage> res(*this);
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
             //TODO valorar no permitir division por cero
             std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
             return *this;
         }
         template<typename U>
         Matrix& operator+=(const Matrix<U,2,type,storage> &m)
         {
             assert(rows() == m.rows() && cols() == m.cols());
             std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
             return *this;
         }
         template<typename U>
         Matrix& operator-=(const Matrix<U,2,type,storage> &m)
         {
             assert(size() == m.size());
             std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
             return *this;
         }

         //Access functions
         matrix_impl::Matrix_Ref<T,1> row(size_t i){
             assert(i < rows());
             matrix_impl::Matrix_Slice<1> row;
             matrix_impl::slice_dim<0>(i,_desc,row);
             return {data(),row};
         }
         matrix_impl::Matrix_Ref<const T,1> row(size_t i) const{
             assert(i < rows());
             matrix_impl::Matrix_Slice<1> row;
             matrix_impl::slice_dim<0>(i,_desc,row);
             return {data(),row};
         }
         matrix_impl::Matrix_Ref<T,1> column(size_t i){
             assert(i < cols());
             matrix_impl::Matrix_Slice<1> col;
             matrix_impl::slice_dim<1>(i,_desc,col);
             return {data(),col};
         }
         matrix_impl::Matrix_Ref<const T,1> column(size_t i) const{
             assert(i < cols());
             matrix_impl::Matrix_Slice<1> col;
             matrix_impl::slice_dim<1>(i,_desc,col);
             return {data(),col};
         }
         matrix_impl::Matrix_Ref<T,1> operator[](size_t i){
                 return row(i);
         }
         matrix_impl::Matrix_Ref<const T,1> operator[](size_t i) const{
                 return row(i);
         }
         T& operator()(size_t i,size_t j){
             assert(i < rows() && j < cols());
             return _elems[i*cols() + j];
         }
         const T& operator()(size_t i,size_t j) const{
             assert(i < rows() && j < cols());
             return _elems[i*cols() + j];
         }
         template<typename... Args>
         std::enable_if_t<matrix_impl::requesting_slice<Args...>(),matrix_impl::Matrix_Ref<T,2>>
         operator()(Args... args){
              matrix_impl::Matrix_Slice<2> d;
              d._size = 1;
              d._start = matrix_impl::do_slice(_desc,d,args...);
              return {data(),d};
         }

         template<typename... Args>
         std::enable_if_t<matrix_impl::requesting_slice<Args...>(),matrix_impl::Matrix_Ref<const T,2>>
         operator()(Args... args) const{
              matrix_impl::Matrix_Slice<2> d;
              d._size = 1;
              matrix_impl::do_slice(_desc,d,args...);
              return {data(),d};
         }
    };
    template<typename T>
    class Matrix<T,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::UPP>{
        size_t _dim;
        std::vector<T> _elems;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::SYMM;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::UPP;
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

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;

        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(i,j);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(i,j);
                }
            }
            return *this;
        }
        Matrix& operator=(const T &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<T,2,type,storage> operator-() const
        {
            Matrix<T,2,type,storage> res(*this);
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
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) -= m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) += m(i,j);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) -= m(i,j);
                }
            }
            return *this;
        }
        T& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols());
            if (i > j) std::swap(i,j); //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }
        const T& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i > j) std::swap(i,j); //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }
    };

    template<typename T>
    class Matrix<T,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::LOW>{
        size_t _dim;
        std::vector<T> _elems;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::SYMM;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::LOW;
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

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;

        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
            return *this;
        }
        Matrix& operator=(const T &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<T,2,type,storage> operator-() const
        {
            Matrix<T,2,type,storage> res(*this);
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
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) -= m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        T& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols());
            if (i < j) std::swap(i,j); //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }
        const T& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i < j) std::swap(i,j); //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }
    };
    template<typename T>
    class Matrix<T,2,Matrix_Type::UTR,Matrix_Storage_Scheme::UPP>{
        size_t _dim;
        std::vector<T> _elems;
        //TODO: pensar en mejor solucion
        static constexpr T offtr_val = 0;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::UTR;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::UPP;
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

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;
        }
        Matrix& operator=(const T &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<T,2,type,storage> operator-() const
        {
            Matrix<T,2,type,storage> res(*this);
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
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }

        T& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols() && i <= j);
            //if (i > j) return offtr_val; //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }
        const T& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i > j) return offtr_val; //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }

    };
    template<typename T>
    class Matrix<T,2,Matrix_Type::LTR,Matrix_Storage_Scheme::LOW>{
        size_t _dim;
        std::vector<T> _elems;
        static constexpr T offtr_val = 0;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::LTR;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::LOW;
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

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;
        }
        Matrix& operator=(const T &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<T,2,type,storage> operator-() const
        {
            Matrix<T,2,type,storage> res(*this);
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
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](T &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }

        T& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols() && i <= j);
            //if (i < j) return offtr_val; //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }
        const T& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i < j) return offtr_val; //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }

    };
    template<typename C>
    class Matrix<std::complex<C>,2,Matrix_Type::HER,Matrix_Storage_Scheme::UPP>{
        size_t _dim;
        std::vector<std::complex<C>> _elems;
        mutable std::complex<C> _aux_var;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::HER;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::UPP;
        using value_type = std::complex<C>;
        using iterator = typename std::vector<std::complex<C>>::iterator;
        using const_iterator = typename std::vector<std::complex<C>>::const_iterator;

        //Default constructor and destructor
        Matrix() = default;
        ~Matrix() = default;

        //Move constructor and assignment
        Matrix(Matrix&&) = default;
        Matrix& operator=(Matrix&&) = default;

        //Copy constructor and assignment
        Matrix(const Matrix&) = default;
        Matrix& operator=(const Matrix&) = default;

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;

        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = std::conj(other(j,i));
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = std::conj(other(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(i,j);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) = other(i,j);
                }
            }
            return *this;
        }
        Matrix& operator=(const std::vector<C> &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        std::vector<C>* data() {return _elems.data();}
        const std::vector<C>* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<std::complex<C>,2,type,storage> operator-() const
        {
            Matrix<std::complex<C>,2,type,storage> res(*this);
            std::for_each(res.begin(),res.end(),[](std::complex<C> &elem){elem = -elem;});
            return res;
        }
        //Arithmetic operations.
        template<typename Scalar>
        Matrix& operator+=(const Scalar &val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem+=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator-=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem-=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator*=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem*=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator/=(const Scalar& val)
        {
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<std::complex<C>>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<std::complex<C>>());
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) += std::conj(m(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::LOW> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) -= std::conj(m(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) += m(i,j);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = i; j < cols(); ++j){
                    this->operator()(i,j) -= m(i,j);
                }
            }
            return *this;
        }
        std::complex<C>& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols() && i <= j);
            //if (i > j) std::swap(i,j); //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }
        const std::complex<C>& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i > j){
                std::swap(i,j);
                _aux_var = std::conj(_elems[j + 0.5*i*(2*_dim - i - 1)]);
                return _aux_var;
            } //upper triangle storage
            return _elems[j + 0.5*i*(2*_dim - i - 1)];
        }
    };

    template<typename C>
    class Matrix<std::complex<C>,2,Matrix_Type::HER,Matrix_Storage_Scheme::LOW>{
        size_t _dim;
        std::vector<std::complex<C>> _elems;
        mutable std::complex<C> _aux_var;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::HER;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::LOW;
        using value_type = std::complex<C>;
        using iterator = typename std::vector<std::complex<C>>::iterator;
        using const_iterator = typename std::vector<std::complex<C>>::const_iterator;

        //Default constructor and destructor
        Matrix() = default;
        ~Matrix() = default;

        //Move constructor and assignment
        Matrix(Matrix&&) = default;
        Matrix& operator=(Matrix&&) = default;

        //Copy constructor and assignment
        Matrix(const Matrix&) = default;
        Matrix& operator=(const Matrix&) = default;

        explicit Matrix(size_t dim):_dim(dim),_elems(0.5*dim*(dim+1)){}
        template<typename U>
        Matrix(const Matrix<U,2,type,storage> &other):_dim(other.rows()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,storage> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.assign(other.begin(),other.end());
            return *this;
        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = std::conj(other(j,i));
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = std::conj(other(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other):_dim(other.rows()){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
        }
        template<typename U>
        Matrix& operator = (const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &other){
            static_assert (std::is_convertible_v<U,std::complex<C>>,"Matrix Constructor: Incompatible elements type.");
            _dim = other.rows();
            _elems.resize(0.5*_dim*(_dim+1));
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) = other(j,i);
                }
            }
            return *this;
        }
        Matrix& operator=(const std::complex<C> &val){
            std::fill(begin(),end(),val);
            return *this;
        }
        size_t extent(size_t n) const{
            assert(n < 2);
            return _dim;
        }
        size_t rows() const noexcept {return _dim;}
        size_t cols() const noexcept {return _dim;}

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        std::complex<C>* data() {return _elems.data();}
        const std::complex<C>* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        //Unary minus
        Matrix<std::complex<C>,2,type,storage> operator-() const
        {
            Matrix<std::complex<C>,2,type,storage> res(*this);
            std::for_each(res.begin(),res.end(),[](std::complex<C> &elem){elem = -elem;});
            return res;
        }
        //Arithmetic operations.
        template<typename Scalar>
        Matrix& operator+=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem+=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator-=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem-=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator*=(const Scalar& val)
        {
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem*=val;});
            return *this;
        }
        template<typename Scalar>
        Matrix& operator/=(const Scalar& val)
        {
            //TODO valorar no permitir division por cero
            std::for_each(begin(),end(),[&val](std::complex<C> &elem){elem/=val;});
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::plus<std::complex<C>>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,storage> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            std::transform(begin(),end(),m.begin(),begin(),std::minus<std::complex<C>>());
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += std::conj(m(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::UPP> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) -= std::conj(m(j,i));
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator+=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,2,type,Matrix_Storage_Scheme::FULL> &m)
        {
            assert(rows() == m.rows()); //symmetric matrizes are squared
            for (size_t i = 0; i < rows(); ++i){
                for (size_t j = 0; j <= i; ++j){
                    this->operator()(i,j) += m(j,i);
                }
            }
            return *this;
        }
        std::complex<C>& operator()(size_t i,size_t j){
            assert(i < rows() && j < cols() && i >= j);
            //if (i < j) std::swap(i,j); //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }
        const std::complex<C>& operator()(size_t i,size_t j) const{
            assert(i < rows() && j < cols());
            if (i < j){
                std::swap(i,j);
                _aux_var = std::conj(_elems[j + 0.5*i*(i + 1)]);
                return _aux_var;
            } //lower triangle storage
            return _elems[j + 0.5*i*(i + 1)];
        }
    };
    //Matrix General-Sparse CSR3:compress sparse row format
    template<typename T>
    class Matrix<T,2,Matrix_Type::GEN,Matrix_Storage_Scheme::CSR3>{
    private:
        int _current_row = -1;
        std::vector<int> _cols;
        std::vector<int> _rowIndex;
        std::vector<T> _elems;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::GEN;
        static constexpr Matrix_Storage_Scheme storage = Matrix_Storage_Scheme::CSR3;
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

        //explicit Matrix(size_t non_zero_elems):_cols(non_zero_elems),_elems(non_zero_elems){}

        void setVals(int row,int col,T val){
            if (_current_row == -1){
                _cols.clear();
                _rowIndex.clear();
                _elems.clear();
            }
            _cols.push_back(col);
            _elems.push_back(val);
            if (row != _current_row){
                _rowIndex.push_back(_cols.size()-1);
                _current_row = row; //updating row
            }
        }
        //matrix values and structure settings are finished
        void setValsFinished(){
            _rowIndex.push_back(_cols.size());
            _current_row = -1;
        }
        //Sets new vals keeping current matrix structure
        void setValsKeepingStructure(const std::vector<T> &elems){
            assert(elems.size() == _elems.size());
            _elems = elems;
        }
        void setValsKeepingStructure(std::vector<T> &&elems){
            assert(elems.size() == _elems.size());
            _elems = elems;
        }
        void printData(){
            printf("values: ( ");
            for (auto &vals : _elems) printf("%f ",vals);
            printf(") \n");
            printf("column: ( ");
            for (auto &vals : _cols) printf("%u ",vals);
            printf(") \n");
            printf("rowIndex: ( ");
            for (auto &vals : _rowIndex) printf("%u ",vals);
            printf(") \n");
        }
        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        int* colsData(){return _cols.data();}
        const int* colsData() const {return _cols.data();}

        int* rowIndexData(){return _rowIndex.data();}
        const int* rowIndexData() const{return _rowIndex.data();}
    };
    //Matrix Symmetric-Sparse CSR3:compress sparse row format
    template<typename T>
    class Matrix<T,2,Matrix_Type::SYMM,Matrix_Storage_Scheme::CSR3>{
    private:
        int _current_row = -1;
        std::vector<int> _cols;
        std::vector<int> _rowIndex;
        std::vector<T> _elems;
    public:
        //Common aliases
        static constexpr size_t order = 2;
        static constexpr Matrix_Type type = Matrix_Type::SYMM;
        static constexpr Matrix_Storage_Scheme storage = Matrix_Storage_Scheme::CSR3;
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

        //explicit Matrix(size_t non_zero_elems):_cols(non_zero_elems),_elems(non_zero_elems){}

        void setVals(int row,int col,T val){
            assert(row <= col);
            if (_current_row == -1){
                _cols.clear();
                _rowIndex.clear();
                _elems.clear();
            }
            _cols.push_back(col);
            _elems.push_back(val);
            if (row != _current_row){
                _rowIndex.push_back(_cols.size()-1);
                _current_row = row; //updating row
            }
        }
        //matrix values and structure settings are finished
        void setValsFinished(){
            _rowIndex.push_back(_cols.size());
            _current_row = -1;
        }
        //Sets new vals keeping current matrix structure
        void setValsKeepingStructure(const std::vector<T> &elems){
            assert(elems.size() == _elems.size());
            _elems = elems;
        }
        void setValsKeepingStructure(std::vector<T> &&elems){
            assert(elems.size() == _elems.size());
            _elems = elems;
        }

        void printData(){
            printf("values: ( ");
            for (auto &vals : _elems) printf("%f ",vals);
            printf(") \n");
            printf("column: ( ");
            for (auto &vals : _cols) printf("%u ",vals);
            printf(") \n");
            printf("rowIndex: ( ");
            for (auto &vals : _rowIndex) printf("%u ",vals);
            printf(") \n");
        }
        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        int* colsData(){return _cols.data();}
        const int* colsData() const {return _cols.data();}

        int* rowIndexData(){return _rowIndex.data();}
        const int* rowIndexData() const{return _rowIndex.data();}
    };
    template<typename T>
    class Matrix<T,1,Matrix_Type::GEN,Matrix_Storage_Scheme::FULL>{
        matrix_impl::Matrix_Slice<1> _desc;
        std::vector<T> _elems;
    public:
        static constexpr size_t order = 1;
        static constexpr Matrix_Type type = Matrix_Type::GEN;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::FULL;
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
        Matrix(const Matrix<U,1,type,storage> &other):_desc(other.descriptor()),_elems{other.begin(),other.end()}{
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
        }

        template<typename U>
        Matrix& operator = (const Matrix<U,1,type,storage> &other){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _desc = other._desc;
            _elems.assign(other.begin(),other.end());
            return *this;
        }
        //TODO ver que hacer aqui
        template<typename U>
        Matrix(const matrix_impl::Matrix_Ref<U,1> &ref):_desc(0,ref.descriptor()._extents[0]),
        _elems(_desc._size)
        {
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            for (size_t i = 0; i < _desc._extents[0]; ++i) _elems[i] = ref(i);
        }
        template<typename U>
        Matrix& operator = (const matrix_impl::Matrix_Ref<U,1> &ref){
            static_assert (std::is_convertible_v<U,T>,"Matrix Constructor: Incompatible elements type.");
            _desc._start = 0;
            _desc._extents = ref.descriptor()._extents;
            _desc.init();
            _elems.resize(_desc._size);
            for (size_t i = 0; i < _desc._extents[0]; ++i) _elems[i] = ref(i);
            return *this;
        }
        Matrix& operator=(const T &val){
            std::for_each(begin(),end(),[&val](T &elem){elem = val;});
            return *this;
        }
        Matrix(std::initializer_list<T> list):_elems(list){
           _desc._start = 0;
           _desc._size = _elems.size();
           _desc._extents[0] = _elems.size();
           _desc._strides[0] = 1;
        }
        Matrix& operator=(std::initializer_list<T> list){
            _elems = list;
            _desc._start = 0;
            _desc._size = _elems.size();
            _desc._extents[0] = _elems.size();
            _desc._strides[0] = 1;
            return *this;
        }

        size_t size() const noexcept{return _elems.size();}

        const matrix_impl::Matrix_Slice<1>& descriptor() const noexcept{
            return _desc;
        }

        iterator begin(){return _elems.begin();}
        const_iterator begin() const{return _elems.cbegin();}

        iterator end(){return _elems.end();}
        const_iterator end() const{return _elems.cend();}

        T* data() {return _elems.data();}
        const T* data() const {return _elems.data();}

        //Arithmetic ops
        template<typename F>
        Matrix& apply(F fun){
            std::for_each(begin(),end(),fun);
            return *this;
        }
        template<typename F>
        Matrix& apply(const Matrix<T,1,type,storage> &vec,F fun){
            std::transform(vec.begin(),vec.end(),this->begin(),fun);
            return *this;
        }

        //Unary minus
        Matrix<T,1,type,storage> operator-() const{
            Matrix<T,1,type,storage> res(*this);
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
        Matrix& operator+=(const Matrix<U,1,type,storage> &m)
        {
            assert(this->size() == m.size());
            std::transform(begin(),end(),m.begin(),begin(),std::plus<T>());
            return *this;
        }
        template<typename U>
        Matrix& operator-=(const Matrix<U,1,type,storage> &m)
        {
            assert(this->size() == m.size());
            std::transform(begin(),end(),m.begin(),begin(),std::minus<T>());
            return *this;
        }

        //access functions
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
    template<typename T>
    class Matrix<T,0,Matrix_Type::GEN,Matrix_Storage_Scheme::FULL>{
        T elem;
    public:
        static constexpr size_t order = 0;
        static constexpr Matrix_Type type = Matrix_Type::GEN;
        static constexpr Matrix_Storage_Scheme storage =  Matrix_Storage_Scheme::FULL;
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

    /****************************************************************************/
    //output operations
    //TODO: enable_if T tiene ostream definido
    template<typename T,Matrix_Type type,Matrix_Storage_Scheme storage>
    inline  std::ostream &operator << (std::ostream &os,const Matrix<T,2,type,storage> &m){
        std::ios_base::fmtflags ff = std::ios::scientific;
        ff |= std::ios::showpos;
        os.setf(ff);
        for (size_t i = 0; i < m.rows(); ++i){
            for (size_t j = 0; j < m.cols(); ++j)
                os << m(i,j) << '\t' ;
            os << '\n';
        }
        os.unsetf(ff);
        return os;
    }
    template<typename T,Matrix_Type type,Matrix_Storage_Scheme storage>
    std::ostream &operator << (std::ostream &os,const Matrix<T,1,type,storage> &m){
        std::ios_base::fmtflags ff = std::ios::scientific;
        ff |= std::ios::showpos;
        os.setf(ff);
        for (size_t i = 0; i < m.size(); ++i){
            os << m(i) << '\n';
        }
        os.unsetf(ff);
        return os;
    }
    /*********************************************************************************/

//    //TODO: Think about type asserts
//    //Matrix Binary Arithmetic Operations
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator+(const Scalar &val,const Matrix<T,N> &m){
//        Matrix<RT,N> res(m);
//        res+=val;
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator+(const Matrix<T,N> &m,const Scalar &val){
//        Matrix<RT,N> res(m);
//        res+=val;
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator-(const Matrix<T,N> &m,const Scalar &val){
//        Matrix<RT,N> res(m);
//        res-=val;
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator-(const Scalar &val,const Matrix<T,N> &m){
//        Matrix<RT,N> res(m);
//        std::for_each(res.begin(),res.end(),[&val](T &elem){elem = val-elem;});
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator*(const Scalar &val,const Matrix<T,N> &m){
//        Matrix<RT,N> res(m);
//        std::for_each(res.begin(),res.end(),[&val](T &elem){elem = val*elem;});
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator*(const Matrix<T,N> &m,const Scalar &val){
//        Matrix<RT,N> res(m);
//        res*=val;
//        return res;
//    }
//    template<typename Scalar,typename T,typename RT = std::common_type_t<Scalar,T>,size_t N>
//    inline Matrix<RT,N> operator/(const Matrix<T,N> &m,const Scalar &val){
//        //TODO controlar division por cero
//        Matrix<RT,N> res(m);
//        res/=val;
//        return res;
//    }
//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>,size_t N>
//    inline Matrix<RT,N> operator+(const Matrix<T1,N> &m1,const Matrix<T2,N> &m2){
//        Matrix<RT,N> res(m1);
//        res+=m2;
//        return res;
//    }
//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>,size_t N>
//    inline Matrix<RT,N> operator-(const Matrix<T1,N> &m1,const Matrix<T2,N> &m2){
//        Matrix<RT,N> res(m1);
//        res-=m2;
//        return res;
//    }

//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>>
//    inline RT operator*(const Matrix<T1,1> &v1,const Matrix<T2,1> &v2){
//        return std::inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
//    }
//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>>
//    inline Matrix<RT,1> operator*(const Matrix<T1,2> &m,const Matrix<T2,1> &v){
//        assert(m.cols() == v.size());
//        Matrix<RT,1> res(m.rows());
//        for (size_t i = 0; i < m.rows();++i){
//            res(i) = std::inner_product(m.row(i).data(),m.row(i).data() + m.cols(),v.begin(),0.0);
//        }
//        return res;
//    }
//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>>
//    inline Matrix<RT,2> operator*(const Matrix<T1,2> &m1,const Matrix<T2,2> &m2){
//        size_t m1m = m1.rows();
//        size_t m2n = m2.cols();
//        size_t m1n = m1.cols();
//        size_t m2m = m2.rows();
//        assert(m1n  == m2m);
//        Matrix<RT,2> res(m1m,m2n);
//        //double-double
//        if constexpr (std::is_same_v<T1,double> && std::is_same_v<T2,double>){
//           printf("dgemm \n");
//           cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor,CBLAS_TRANSPOSE::CblasNoTrans,CBLAS_TRANSPOSE::CblasNoTrans,
//                        m1m,m2n,m1n,1.0,m1.data(),m1n,m2.data(),m2n,0.0,res.data(),m2n);
//        //TODO implementar uso de cblas con float y complejos
//        }else{
//            //NAIVE implementation
//            printf("naive \n");
//            for (size_t i = 0; i < m1m; ++i){
//                for (size_t j = 0; j < m2n; ++j){
//                    RT tmp(0);
//                    for (size_t k = 0; k < m1n; ++k){
//                        tmp+=m1(i,k)*m2(k,j);
//                    }
//                    res(i,j) = tmp;
//                }
//            }
//        }
//        return res;
//    }

    /**************************************************************************************************/
//    template<typename T1,typename T2,typename RT = std::common_type_t<T1,T2>>
//    inline Matrix<RT,1> operator *(const Matrix<T1,2> &m2,const Matrix<T2,1> m1){
//        assert(m2.columns() == m1.size());
//        Matrix<RT,1> res(m2.rows());
//        for (size_t i = 0; i < m2.rows(); ++i){
//            res(i) = std::inner_product(m2.row(i).begin(),m2.row(i).end(),m1.begin(),0);
//        }
//        return res;
//    }

//    //TODO optimizar estas operaciones y para tipos genericos
//    inline double operator*(const matrix_impl::Matrix_Ref<double,1> &mref,const Matrix<double,1> &v){
//        assert(mref.size() == v.size());
//        double temp = 0.0;
//        for (size_t i = 0; i < v.size(); ++i)  temp+=mref(i)*v(i);
//        return temp;
//    }

//    inline double operator *(const VecDoub &u,const VecDoub &v){
//        assert(u.size() == v.size());
//        return std::inner_product(u.begin(),u.end(),v.begin(),0.0);
//    }
//    inline VecDoub operator *(const MatDoub &M,const VecDoub &v){
//        assert(M.cols() == v.size());

//        VecDoub r(M.rows());

//        for (size_t i = 0; i < M.rows(); ++i){
//            double tmp = 0.0;
//            for (size_t j = 0; j < M.cols(); ++j){
//                tmp += M(i,j)*v(j);
//            }
//            r(i) = tmp;
//        }
//        return r;
//    }
//    inline VecDoub operator *(const VecDoub &v,const MatDoub &M){
//        assert(M.rows() == v.size());

//        VecDoub r(M.cols());

//        for (size_t i = 0; i < M.cols(); ++i){
//            double tmp = 0.0;
//            for (size_t j = 0; j < M.rows(); ++j){
//                tmp += v(j)*M(j,i);
//            }
//            r(i) = tmp;
//        }
//        return r;
//    }
//    inline MatDoub operator*(const MatDoub &M1,const MatDoub &M2){

//        assert(M1.cols() == M2.rows());

//        const size_t m = M1.rows();
//        const size_t n = M2.cols();
//        const size_t p = M1.cols();
//        MatDoub R(m,n);

//        for (size_t i = 0; i < m ; ++i){
//            for (size_t j = 0; j < n; ++j){
//                double temp = 0.0;
//                for (size_t k = 0; k < p; ++k){
//                    temp+=M1(i,k)*M2(k,j);
//                }
//                R(i,j) = temp;
//            }
//        }
//        return R;
//    }

//    inline MatDoub diadic_product(const VecDoub &v1,const VecDoub &v2){
//        MatDoub R(v1.size(),v2.size());

//        for (size_t i = 0; i < v1.size(); ++i){
//            for (size_t j = 0; j < v2.size(); ++j){
//                R(i,j) = v1(i)*v2(j);
//            }
//        }
//        return R;
//    }
//    inline Matrix<double,4> diadic_product(const VecDoub &v1,const VecDoub &v2,const VecDoub &v3,const VecDoub &v4){
//        Matrix<double,4> R(v1.size(),v2.size(),v3.size(),v4.size());

//        for (size_t i = 0; i < v1.size(); ++i){
//            for (size_t j = 0; j < v2.size(); ++j){
//                for (size_t k = 0; k < v3.size(); ++k){
//                    for (size_t l = 0; l < v4.size(); ++l){
//                        R(i,j,k,l) = v1(i)*v2(j)*v3(k)*v4(l);
//                    }
//                }

//            }
//        }
//        return R;
//    }
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
//    template<typename T>
//    std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,2> &m);/*{
//        if (ofs.is_open()){
//            std::ios_base::fmtflags ff = std::ios::scientific;
//            ff |= std::ios::showpos;
//            ofs.setf(ff);
//            for (size_t i = 0; i < m.rows(); ++i){
//                for (size_t j = 0; j < m.cols(); ++j)
//                    ofs << m(i,j) << '\t' ;
//                ofs << std::endl;
//            }
//            ofs.unsetf(ff);
//        }
//        return ofs;
//    }*/
//    template<typename T>
//    std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,1> &m);/*{
//        if (ofs.is_open()){
//            std::ios_base::fmtflags ff = std::ios::scientific;
//            ff |= std::ios::showpos;
//            ofs.setf(ff);
//            for (size_t i = 0; i < m.size(); ++i){
//                ofs << i << "**" << m(i) << '\n';
//            }
//            ofs.unsetf(ff);
//        }
//        return ofs;
//    }*/
//    template<typename T>
//    inline std::ostream& operator << (std::ostream& os,const Matrix<T,3> &m){
//        //if (ofs.is_open()){
//            //std::ios_base::fmtflags ff = std::ios::scientific;
//            //ff |= std::ios::showpos;
//            //os.setf(ff);
//            for (size_t i = 0; i < m.extent(2); ++i){
//                os << "(:,:," << i << ") \n";
//                for (size_t j = 0; j < m.extent(0); ++j){
//                    for (size_t k = 0; k < m.extent(1); ++k){

//                            os << m(j,k,i) << '\t' ;
//                    }

//                    os << "\n";
//                }
//            }
//            //os.unsetf(ff);
//            return os;
//        //}
//    }
//    template<typename T>
//    std::ostream& operator << (std::ostream& os,const Matrix<T,4> &m);/*{
//        //if (ofs.is_open()){
//            std::ios_base::fmtflags ff = std::ios::scientific;
//            ff |= std::ios::showpos;
//            os.setf(ff);
//            for (size_t i = 0; i < m.extent(3); ++i){
//                for (size_t j = 0; j < m.extent(2); ++j){
//                    os << "(:,:," << j << ", " << i << ") \n";
//                    for (size_t k = 0; k < m.extent(0); ++k){
//                        for (size_t l = 0; l < m.extent(1); ++l){
//                            os << m(k,l,j,i) << '\t' ;
//                        }
//                        os << "\n";
//                    }
//                }
//            }
//            os.unsetf(ff);
//            return os;
//        //}
//    }*/



#endif // MATRIX_H
