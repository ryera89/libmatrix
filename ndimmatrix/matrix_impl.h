#ifndef MATRIX_IMPL_H
#define MATRIX_IMPL_H

#include "matrix_slice.h"
#include <assert.h>


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


template<typename List>
inline bool check_non_jagged(const List &list){ //check that all rows has the same number of elements
    auto i = list.begin();
    for (auto j = i+1; j!=list.end();++j){
        if (i->size()!=j->size()) return false;
    }
    return true;
}
template<size_t N,typename It,typename List>
inline void add_extents(It &first,const List &list){
    if constexpr (N == 1){
        *first = list.size();
    }else{
       assert(check_non_jagged(list));
       *first = list.size();
       add_extents<N-1>(++first,*list.begin());
    }

}
template<size_t N,typename List>
inline void derive_extents(std::array<size_t,N> &exts,const List &list){
    auto first  = exts.begin();
    add_extents<N>(first,list);
}
template<typename T,typename Vec>
inline void add_list(const T* first,const T* last,Vec &vec,size_t &pos){
    //vec.insert(vec.end(),first,last);
    while (first!=last) {
        vec[pos++] = *first++;
        //++first;
    }
}
template<typename T,typename Vec>
inline void add_list(const std::initializer_list<T> *first,const std::initializer_list<T> *last,Vec &vec,size_t &pos){
    for (;first!=last;++first)
        add_list(first->begin(),first->end(),vec,pos);
}
template<typename T,typename Vec>
inline void insert_flat(std::initializer_list<T> list,Vec &vec){
    size_t pos = 0;
    add_list(list.begin(),list.end(),vec,pos);
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
    return all((std::is_convertible_v<Args,size_t> || std::is_same_v<Args,std::slice>)...)
               && some(std::is_same_v<Args,std::slice>...);
}
template<size_t N,typename... Dims>
inline bool check_bounds(const Matrix_Slice<N> &slice,Dims... dims){
    size_t indexs[N]{size_t(dims)...};
    return std::equal(indexs,indexs+N,slice._extents.begin(),std::less<size_t>{});
}
template<size_t Dim,size_t N,typename S>
constexpr size_t do_slice_dim(const Matrix_Slice<N> &os,Matrix_Slice<N> &ns,const S &s){
    size_t start{0};

    if constexpr (std::is_same_v<S,std::slice>){
        assert(s.start() + s.size() <= os.m_extents[N-Dim]);
        ns.m_extents[N-Dim] = s.size();
        ns.m_strides[N-Dim] = os.m_strides[N-Dim]*s.stride();
        ns.m_size *= ns.m_extents[N-Dim];
        start = os.m_strides[N-Dim]*s.start();
    }else{    //then is an size_t type or convertible to it.
        assert(s < os.m_extents[N-Dim]);
        ns.m_extents[N-Dim] = 1;
        ns.m_strides[N-Dim] = os.m_strides[N-Dim];
        start = os.m_strides[N-Dim]*s;
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
    ns.m_start = n*os.m_strides[Dim];
    if constexpr (Dim == 0){
        std::copy(&os.m_extents[1],os.m_extents.end(),ns.m_extents.begin());
        std::copy(&os.m_strides[1],os.m_strides.end(),ns.m_strides.begin());
        ns.m_size = os.m_size/os.m_extents[0];
    }
    if constexpr (Dim == 1){
        ns.m_extents[0] = os.m_extents[0];
        ns.m_strides[0] = os.m_strides[0];
        std::copy(&os._extents[2],os.m_extents.end(),&ns.m_extents[1]);
        std::copy(&os._strides[2],os.m_strides.end(),&ns.m_strides[1]);
        ns.m_size = os.m_size/os.m_extents[1];
    }

}

#endif // MATRIX_IMPL_H
