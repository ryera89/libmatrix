#ifndef MATRIX_IMPL_H
#define MATRIX_IMPL_H

#include <assert.h>
#include <type_traits>
#include <initializer_list>
#include <array>
#include <numeric>
#include <algorithm>

namespace matrix_impl {

/*MATRIX_INIT**************Usadas para la inicializacion mediante initializar_lists*************/
template<typename T,size_t N>
struct Matrix_Init{
    using type = std::initializer_list<typename Matrix_Init<T,N-1>::type>;
};

template<typename T>
struct Matrix_Init<T,1>{
    using type = std::initializer_list<T>;
};
template<typename T>
struct Matrix_Init<T,0>; //no definido, seria un escalar

template<typename T,size_t N>
using Matrix_Initializer = typename Matrix_Init<T,N>::type;
/*_MATRIX_INIT************************************************************************************/

/************************clases para slicing y shape de las matrices*****************************/
//SLICE
struct Slice{
    size_t _start = 0;
    size_t _length = 0;
    size_t _stride = 0;

    constexpr Slice() = default;
    //explicit Slice(size_t s):_start(s){}
    constexpr explicit Slice(size_t start,size_t lenght,size_t stride = 1):
        _start(start),_length(lenght),_stride(stride){}

    constexpr void setSliceValues(size_t start,size_t length,size_t stride = 0) noexcept {
        _start=start;_length=length;_stride=stride;}
    constexpr size_t operator()(const size_t &i) const noexcept {return _start + i*_stride;}
};

//MATRIX_SLICE
template<size_t N>
struct Matrix_Slice
{
    //members
    size_t _start = 0;
    size_t _size = 0;
    std::array<size_t,N> _extents;
    std::array<size_t,N> _strides;

    Matrix_Slice() = default;
    Matrix_Slice(size_t s,const std::initializer_list<size_t> &exts):_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        init(); //computes strides and size
    }
    Matrix_Slice(size_t s,const std::initializer_list<size_t> &exts,std::initializer_list<size_t> strs):_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        std::copy(strs.begin(),strs.end(),_strides.begin());
        _size = std::accumulate(_extents.begin(),_extents.end(),1,std::multiplies<size_t>());
    }
    Matrix_Slice(size_t s,const std::array<size_t,N> &exts):_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        init(); //computes strides and size
    }
    void init(){
        _strides[N - 1] = 1;
        for (std::size_t i = N - 1; i != 0; --i) {
        _strides[i - 1] = _strides[i] * _extents[i];
        }
        _size = _extents[0] * _strides[0];
    }

    template<typename... Indexs>
    size_t operator()(const Indexs&... indexs){
        static_assert (sizeof...(Indexs) == N,"");
        size_t args[N] {size_t(indexs)...};
        return std::inner_product(args,args+N,_strides.begin(),size_t(0));
    }

    template<typename... Indexs>
    size_t operator()(const Indexs&... indexs) const{
        static_assert (sizeof...(Indexs) == N,"");
        size_t args[N] {size_t(indexs)...};
        return std::inner_product(args,args+N,_strides.begin(),size_t(0));
    }
};
//Especializacion para matrices de orden 4
template<>
struct Matrix_Slice<4>{
    size_t _start = 0;
    size_t _size = 0;
    std::array<size_t,4> _extents = {0,0,0,0};
    std::array<size_t,4> _strides = {0,0,0,0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext0,size_t ext1,size_t ext2,size_t ext3) noexcept :_start(s){
        _extents[0] = ext0;
        _extents[1] = ext1;
        _extents[2] = ext2;
        _extents[3] = ext3;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,std::initializer_list<size_t> exts) noexcept :_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        init(); //computes strides and size
    }

    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,4> exts) noexcept :_start(s){
        _extents[0] = exts[0];
        _extents[1] = exts[1];
        _extents[2] = exts[2];
        _extents[3] = exts[3];
        init(); //computes strides and size
    }

   constexpr void init() noexcept {
        _strides[3] = size_t(1);
        _strides[2] = _strides[3]*_extents[3];
        _strides[1] = _strides[2]*_extents[2];
        _strides[0] = _strides[1]*_extents[1];

        _size = _extents[0]*_strides[0];
    }
//    size_t operator()(const size_t& i,const size_t& j,const size_t& k,const size_t& l){
//        return i*_strides[0] + j*_strides[1] + k*_strides[2] + l;
//    }
    constexpr size_t operator()(size_t i,size_t j,size_t k,size_t l) const noexcept{
        return i*_strides[0] + j*_strides[1] + k*_strides[2] + l*_strides[3];
    }

};
//Especializacion para matrices de orden 3
template<>
struct Matrix_Slice<3>{
    size_t _start = 0;
    size_t _size = 0;
    std::array<size_t,3> _extents {0,0,0};
    std::array<size_t,3> _strides {0,0,0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext0,size_t ext1,size_t ext2) noexcept :_start(s){
        _extents[0] = ext0;
        _extents[1] = ext1;
        _extents[2] = ext2;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,std::initializer_list<size_t> exts) noexcept :_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        init(); //computes strides and size
    }

    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,3> exts) noexcept :_start(s){
        _extents[0] = exts[0];
        _extents[1] = exts[1];
        _extents[2] = exts[2];
        init(); //computes strides and size
    }

    constexpr void init() noexcept {
        _strides[2] = size_t(1);
        _strides[1] = _strides[2]*_extents[2];
        _strides[0] = _strides[1]*_extents[1];
        _size = _extents[0]*_strides[0];
    }

//    size_t operator()(const size_t& i,const size_t& j,const size_t& k){
//        return i*_strides[0] + j*_strides[1] + k;
//    }
    constexpr size_t operator()(size_t i,size_t j,size_t k) const noexcept{
        return i*_strides[0] + j*_strides[1] + k*_strides[2];
    }

};
//Especializacion para matrices de orden 2
template<>
struct Matrix_Slice<2>{
    size_t _start = 0;
    size_t _size = 0;
    std::array<size_t,2> _extents = {0,0};
    std::array<size_t,2> _strides = {0,0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext0,size_t ext1) noexcept :_start(s){
        _extents[0] = ext0;
        _extents[1] = ext1;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,std::initializer_list<size_t> exts) noexcept :_start(s){
        std::copy(exts.begin(),exts.end(),_extents.begin());
        init(); //computes strides and size
    }

    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,2> exts) noexcept :_start(s){
        _extents[0] = exts[0];
        _extents[1] = exts[1];
        init(); //computes strides and size
    }

    constexpr void init() noexcept {
        _strides[0] = _extents[1];
        _strides[1] = size_t(1);
        _size = _extents[0]*_extents[1];
    }

//    constexpr size_t operator()(const size_t& i,const size_t& j){
//        return i*_strides[0] + j;
//    }
    constexpr size_t operator()(size_t i,size_t j) const noexcept {
        return i*_strides[0] + j*_strides[1];
    }

};
template<>
struct Matrix_Slice<1>{
    size_t _start = 0;
    size_t _size = 0;
    std::array<size_t,1> _extents = {0};
    std::array<size_t,1> _strides = {0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext) noexcept :_start(s){
        _extents[0] = ext;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,1> exts) noexcept :_start(s){
        _extents[0] = exts[0];
        init(); //computes strides and size
    }

    constexpr void init() noexcept{
        _strides[0] = size_t(1);
        _size = _extents[0];
    }

//    constexpr size_t operator()(const size_t& i){
//        return i;
//    }
    constexpr size_t operator()(size_t i) const noexcept{
        return i*_strides[0];
    }

};
/*************************Funciones de ayuda para la implementacion************************************************/
template<typename List>
inline bool check_non_jagged(const List &list){ //check that all rows has the same number of elements
    auto i = list.begin();
    for (auto j = i+1; j!=list.end();++j){
        if (i->size()!=j->size()) return false;
    }
    return true;
}
template<std::size_t N,typename It,typename List>
inline void add_extents(It &first,const List &list){
    if constexpr (N == 1){
        *first = list.size();
    }else{
       assert(check_non_jagged(list));
       *first = list.size();
       add_extents<N-1>(++first,*list.begin());
    }

}
template<std::size_t N,typename List>
inline void derive_extents(std::array<size_t,N> &exts,const List &list){
    auto first  = exts.begin();
    add_extents<N>(first,list);
}
template<typename T,typename Vec>
inline void add_list(const T* first,const T* last,Vec &vec){
    vec.insert(vec.end(),first,last);
}
template<typename T,typename Vec>
inline void add_list(const std::initializer_list<T> *first,const std::initializer_list<T> *last,Vec &vec){
    for (;first!=last;++first)
        add_list(first->begin(),first->end(),vec);
}
template<typename T,typename Vec>
inline void insert_flat(std::initializer_list<T> list,Vec &vec){
    add_list(list.begin(),list.end(),vec);
}
template<typename... Args>
constexpr bool all(Args...args){
    return (args && ...);
}
template<typename... Args>
constexpr bool some(Args... args){
    return (args || ...);
}
template<typename... Args>
constexpr bool requesting_element(){
    return (std::is_convertible_v<Args,size_t> && ...);
}
template<typename... Args>
constexpr bool requesting_slice(){
    return all((std::is_convertible_v<Args,size_t> || std::is_same_v<Args,matrix_impl::Slice>)...)
               && some(std::is_same_v<Args,matrix_impl::Slice>...);
}
template<size_t N,typename... Dims>
inline bool check_bounds(const Matrix_Slice<N> &slice,Dims... dims){
    size_t indexs[N]{size_t(dims)...};
    return std::equal(indexs,indexs+N,slice._extents.begin(),std::less<size_t>{});
}
template<size_t Dim,size_t N,typename S>
constexpr size_t do_slice_dim(const Matrix_Slice<N> &os,Matrix_Slice<N> &ns,const S &s){
    size_t start{0};

    if constexpr (std::is_same_v<S,Slice>){
        assert(s._start + s._length <= os._extents[N-Dim]);
        ns._extents[N-Dim] = s._length;
        ns._strides[N-Dim] = os._strides[N-Dim]*s._stride;
        ns._size *= ns._extents[N-Dim];
        start = os._strides[N-Dim]*s._start;
    }else{    //then is an size_t type or convertible to it.
        assert(s < os._extents[N-Dim]);
        ns._extents[N-Dim] = 1;
        ns._strides[N-Dim] = os._strides[N-Dim];
        start = os._strides[N-Dim]*s;
    }

   return start;
}
template<size_t N>
constexpr size_t do_slice(const Matrix_Slice<N> &/*os*/,Matrix_Slice<N> &/*ns*/){

    return 0;
}
template<size_t N,typename T,typename... Args>
constexpr size_t do_slice(const Matrix_Slice<N> &os,Matrix_Slice<N> &ns,const T& s,const Args&... args){
    //return (do_slice_dim<sizeof... (Args)>(os,ns,args) + ...);
    size_t m = do_slice_dim<sizeof... (Args)+1>(os,ns,s);
    size_t n = do_slice(os,ns,args...);
    return n+m;
}
template<size_t Dim,size_t N>
constexpr void slice_dim(size_t n,const Matrix_Slice<N> &os,Matrix_Slice<N-1> &ns){
    ns._start = n*os._strides[Dim];
    if constexpr (Dim == 0){
        std::copy(&os._extents[1],os._extents.end(),ns._extents.begin());
        std::copy(&os._strides[1],os._strides.end(),ns._strides.begin());
        ns._size = os._size/os._extents[0];
    }
    if constexpr (Dim == 1){
        ns._extents[0] = os._extents[0];
        ns._strides[0] = os._strides[0];
        std::copy(&os._extents[2],os._extents.end(),&ns._extents[1]);
        std::copy(&os._strides[2],os._strides.end(),&ns._strides[1]);
        ns._size = os._size/os._extents[1];
    }

}
/*****************************Clase para referenciar matrices****************************************************************/
//TODO VALORAR IMPLEMENTACION DE MATRIX_REF_ITERATOR
//MATRIX_REF
template<typename T,size_t N>
class Matrix_Ref
{
    Matrix_Slice<N> _desc;
    T *_ptr;
public:
    static constexpr size_t order = N;
    using value_type = T;
    //using iterator = typename std::vector<T>::iterator;
    //using const_iterator = typename std::vector<T>::const_iterator;

    Matrix_Ref(T *p,const Matrix_Slice<N> &d):_desc{d},_ptr{p+d._start}{}

    //TODO Faltan constructores aqui...

    T* data(){return _ptr;}
    const T* data()const{return _ptr;}
    const matrix_impl::Matrix_Slice<N>& descriptor() const noexcept{return _desc;}
    size_t size() const noexcept {return _desc._size;}
    size_t extent(size_t n) const{
        assert(n < N);
        return _desc._extents[n];
    }
    matrix_impl::Matrix_Ref<T,N-1> row(size_t i){
        assert(i < extent(0));
        Matrix_Slice<N-1> row;
        slice_dim<0>(i,_desc,row);
        return {data(),row};
    }
    matrix_impl::Matrix_Ref<const T,N-1>
    row(size_t i) const{
        assert(i < extent(0));
        matrix_impl::Matrix_Slice<N-1> row;
        matrix_impl::slice_dim<0>(i,_desc,row);
        return {data(),row};
    }
    matrix_impl::Matrix_Ref<T,N-1> operator[](size_t i){
        return row(i);
    }
    matrix_impl::Matrix_Ref<const T,N-1> operator[](size_t i) const{
        return row(i);
    }
    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(), T&>
    operator()(Args... args){
        assert(matrix_impl::check_bounds(_desc,args...));
        return _ptr[_desc(args...)];
    }
    template<typename... Args>
    std::enable_if_t<matrix_impl::requesting_element<Args...>(),const T&>
    operator()(Args... args) const{
        assert(matrix_impl::check_bounds(_desc,args...));
        return _ptr[_desc(args...)];
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
    class Matrix_Ref<T,2>{
       Matrix_Slice<2> _desc;
       T* _ptr;

    public:
       static constexpr size_t order = 2;
       using value_type = T;

       Matrix_Ref(T *p,const Matrix_Slice<2> &d):_desc{d},_ptr{p+d._start}{}
       //TODO Faltan constructores aqui...

       T* data() noexcept{return _ptr;}
       const T* data() const noexcept{return _ptr;}
       const matrix_impl::Matrix_Slice<2>& descriptor() const noexcept { return _desc; }
       size_t size() const noexcept {return _desc._size;}
       size_t extent(size_t n) const{
           assert(n < 2);
           return _desc._extents[n];
       }
       size_t rows() const noexcept{return extent(0);}
       size_t cols() const noexcept{return extent(1);}

       matrix_impl::Matrix_Ref<T,1> row(size_t i){
           assert(i < extent(0));
           matrix_impl::Matrix_Slice<1> row;
           matrix_impl::slice_dim<0>(i,_desc,row);
           return {data(),row};
       }
       matrix_impl::Matrix_Ref<const T,1> row(size_t i) const{
           assert(i < extent(0));
           matrix_impl::Matrix_Slice<1> row;
           matrix_impl::slice_dim<0>(i,_desc,row);
           return {data(),row};
       }
       matrix_impl::Matrix_Ref<T,1> operator[](size_t i){
           return row(i);
       }
       matrix_impl::Matrix_Ref<const T,1> operator[](size_t i) const{
           return row(i);
       }
       T& operator()(size_t i,size_t j){
           assert(i < rows() && j < cols());
           return _ptr[_desc(i,j)];
       }
       const T& operator()(size_t i,size_t j) const{
           assert(i < rows() && j < cols());
           return _ptr[_desc(i,j)];
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
    class Matrix_Ref<T,1>{
        Matrix_Slice<1> _desc;
        T* _ptr;

     public:
        static constexpr size_t order = 1;
        using value_type = T;

        Matrix_Ref(T *p,const Matrix_Slice<1> &d):_desc{d},_ptr{p+d._start}{}
        //TODO Faltan constructores aqui...

        T* data() noexcept{return _ptr;}
        const T* data() const noexcept{return _ptr;}
        const matrix_impl::Matrix_Slice<1>& descriptor() const noexcept { return _desc; }
        size_t size() const noexcept {return _desc._size;}

        T& row(size_t i){
            assert(i < size());
            return _ptr[_desc(i)];
        }
        const T& row(size_t i) const{
            assert(i < size());
            return _ptr[_desc(i)];
        }

        T& operator[](size_t i){
            return row(i);
        }
        const T& operator[](size_t i) const{
            return row(i);
        }
        T& operator()(size_t i){
            assert(i < size());
            return _ptr[_desc(i)];
        }
        const T& operator()(size_t i) const{
            assert(i < size());
            return _ptr[_desc(i)];
        }
        matrix_impl::Matrix_Ref<T,1> operator()(const Slice &s){
             matrix_impl::Matrix_Slice<1> d;
             d._start = s._start;
             d._size = s._length;
             d._extents[0] = s._length;
             d._strides[0] = s._stride;
             //d._start = matrix_impl::do_slice(_desc,d,args...);
             return {data(),d};
        }
        matrix_impl::Matrix_Ref<const T,1> operator()(const Slice &s) const{
             matrix_impl::Matrix_Slice<1> d;
             d._start = s._start;
             d._size = s._length;
             d._extents[0] = s._length;
             d._strides[0] = s._stride;
             //d._start = matrix_impl::do_slice(_desc,d,args...);
             return {data(),d};
        }

    };
    template<typename T>
    class Matrix_Ref<T,0>{
        T* _ptr_elem;
    public:
        static constexpr size_t order = 0;
        using value_type = T;

        Matrix_Ref(T* ptr_elem):_ptr_elem(ptr_elem){}
        Matrix_Ref& operator=(const T &val){
            *_ptr_elem = val;
            return *this;
        }

        T& row(size_t i) = delete;

        T& operator()(){return *_ptr_elem;}
        const T& operator()()const {return *_ptr_elem;}

        //Conversion operator from Matrix_Ref<T,0> to T*
        operator T*(){return _ptr_elem;}
        operator const T*(){return _ptr_elem;}

    };

    template<typename U,typename T>
    inline void assing_slice_vals(const Matrix_Ref<U,1> &ref,Matrix_Ref<T,1> &mref){
        static_assert(std::is_convertible_v<U,T>,"assign_slice_vals: Incompatible elements type.");
        assert(ref.size() == mref.size());
        for (size_t i = 0; i < mref.size(); ++i) mref(i) = ref(i);
    }
    template<typename U,typename T,size_t N>
    inline void assing_slice_vals(const Matrix_Ref<U,N> &ref,Matrix_Ref<T,N> &mref){
        for (size_t i = 0; i < ref.descriptor()._extents[0];++i){
            Matrix_Ref<T,N-1> mref_aux = mref[i];
            assing_slice_vals(ref[i],mref_aux);
        }

    }

}

#endif // MATRIX_IMPL_H
