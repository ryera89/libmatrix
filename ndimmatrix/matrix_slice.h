#ifndef MATRIX_SLICE_H
#define MATRIX_SLICE_H

#include <array>
#include <numeric>
#include <valarray>

template<size_t N>
struct Matrix_Slice
{
    //members
    size_t m_start = 0;
    size_t m_size = 0;
    std::array<size_t,N> m_extents;
    std::array<size_t,N> m_strides;

    Matrix_Slice() = default;
    Matrix_Slice(size_t s,const std::initializer_list<size_t> &exts):m_start(s){
        std::copy(exts.begin(),exts.end(),m_extents.begin());
        init(); //computes strides and size
    }
    Matrix_Slice(size_t s,const std::initializer_list<size_t> &exts,std::initializer_list<size_t> strs):m_start(s){
        std::copy(exts.begin(),exts.end(),m_extents.begin());
        std::copy(strs.begin(),strs.end(),m_strides.begin());
        m_size = std::accumulate(m_extents.begin(),m_extents.end(),1,std::multiplies<size_t>());
    }
    Matrix_Slice(size_t s,const std::array<size_t,N> &exts):m_start(s){
        std::copy(exts.begin(),exts.end(),m_extents.begin());
        init(); //computes strides and size
    }
    void init(){
        m_strides[N - 1] = 1;
        for (std::size_t i = N - 1; i != 0; --i) {
        m_strides[i - 1] = m_strides[i] * m_extents[i];
        }
        m_size = m_extents[0] * m_strides[0];
    }

    template<typename... Indexs>
    size_t operator()(const Indexs&... indexs){
        static_assert (sizeof...(Indexs) == N,"");
        size_t args[N] {size_t(indexs)...};
        return std::inner_product(args,args+N,m_strides.begin(),m_start);
    }

    template<typename... Indexs>
    size_t operator()(const Indexs&... indexs) const{
        static_assert (sizeof...(Indexs) == N,"");
        size_t args[N] {size_t(indexs)...};
        return std::inner_product(args,args+N,m_strides.begin(),m_start);
    }
};
template<>
struct Matrix_Slice<2>{
    size_t m_start = 0;
    size_t m_size = 0;
    std::array<size_t,2> m_extents = {0,0};
    std::array<size_t,2> m_strides = {0,0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext0,size_t ext1) noexcept :m_start(s){
        m_extents[0] = ext0;
        m_extents[1] = ext1;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,std::initializer_list<size_t> exts) noexcept :m_start(s){
        std::copy(exts.begin(),exts.end(),m_extents.begin());
        init(); //computes strides and size
    }

    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,2> exts) noexcept :m_start(s){
        m_extents[0] = exts[0];
        m_extents[1] = exts[1];
        init(); //computes strides and size
    }

    constexpr void init() noexcept {
        m_strides[0] = m_extents[1];
        m_strides[1] = size_t(1);
        m_size = m_extents[0]*m_extents[1];
    }
    constexpr size_t operator()(size_t i,size_t j) const noexcept {
        return m_start + i*m_strides[0] + j*m_strides[1];
    }
};
template<>
struct Matrix_Slice<1>{
    size_t m_start = 0;
    size_t m_size = 0;
    std::array<size_t,1> m_extents = {0};
    std::array<size_t,1> m_strides = {0};

    constexpr Matrix_Slice() noexcept = default;
    constexpr explicit Matrix_Slice(size_t s,size_t ext) noexcept :m_start(s){
        m_extents[0] = ext;
        init();
    }
    constexpr explicit Matrix_Slice(size_t s,size_t ext,size_t stride) noexcept :m_start(s){
        m_extents[0] = ext;
        m_size = ext;
        m_strides[0] = stride;
    }
    constexpr explicit Matrix_Slice(size_t s,std::array<size_t,1> exts) noexcept :m_start(s){
        m_extents[0] = exts[0];
        init(); //computes strides and size
    }

    constexpr void init() noexcept{
        m_strides[0] = size_t(1);
        m_size = m_extents[0];
    }
    constexpr size_t operator()(size_t i) const noexcept{
        return m_start + i*m_strides[0];
    }
};

#endif // MATRIX_SLICE_H
