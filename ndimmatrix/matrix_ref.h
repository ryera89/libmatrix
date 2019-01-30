#ifndef MATRIX_REF_H
#define MATRIX_REF_H

#include "matrix_impl.h"
#include <complex>

template<typename T,size_t N>
class Matrix_Ref
{
    Matrix_Slice<N> m_desc;
    T *m_ptr;
public:
    static constexpr size_t order = N;
    using value_type = T;
    //using iterator = typename std::vector<T>::iterator;
    //using const_iterator = typename std::vector<T>::const_iterator;

    Matrix_Ref(T *p,const Matrix_Slice<N> &d):m_desc{d},m_ptr{p}{}

    //TODO Faltan constructores aqui...

    T* data(){return m_ptr;}
    const T* data()const{return m_ptr;}
    const Matrix_Slice<N>& descriptor() const noexcept{return m_desc;}
    size_t size() const noexcept {return m_desc.m_size;}
    size_t extent(size_t n) const{
        assert(n < N);
        return m_desc.m_extents[n];
    }
    Matrix_Ref<T,N-1> row(size_t i){
        assert(i < extent(0));
        Matrix_Slice<N-1> row;
        slice_dim<0>(i,m_desc,row);
        return {data(),row};
    }
    Matrix_Ref<const T,N-1>
    row(size_t i) const{
        assert(i < extent(0));
        Matrix_Slice<N-1> row;
        slice_dim<0>(i,m_desc,row);
        return {data(),row};
    }
    Matrix_Ref<T,N-1> operator[](size_t i){
        return row(i);
    }
    Matrix_Ref<const T,N-1> operator[](size_t i) const{
        return row(i);
    }
    template<typename... Args>
    std::enable_if_t<requesting_element<Args...>(), T&>
    operator()(Args... args){
        assert(check_bounds(m_desc,args...));
        return m_ptr[_desc(args...)];
    }
    template<typename... Args>
    std::enable_if_t<requesting_element<Args...>(),const T&>
    operator()(Args... args) const{
        assert(check_bounds(m_desc,args...));
        return m_ptr[m_desc(args...)];
    }

    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<T,N>>
    operator()(Args... args){
         Matrix_Slice<N> d;
         d.m_size = 1;
         d.m_start = do_slice(m_desc,d,args...);
         return {data(),d};
    }

    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<const T,N>>
    operator()(Args... args) const{
         Matrix_Slice<N> d;
         d.m_size = 1;
         do_slice(m_desc,d,args...);
         return {data(),d};
    }
};

template<typename T>
class Matrix_Ref<T,2>{
    Matrix_Slice<2> m_desc;
    T* m_ptr;

public:
    static constexpr size_t order = 2;
    using value_type = T;

    Matrix_Ref(T *p,const Matrix_Slice<2> &d):m_desc{d},m_ptr{p}{}
    //TODO Faltan constructores aqui...

    T* data() noexcept{return m_ptr;}
    const T* data() const noexcept{return m_ptr;}
    const Matrix_Slice<2>& descriptor() const noexcept { return m_desc; }
    size_t size() const noexcept {return m_desc.m_size;}
    size_t extent(size_t n) const{
        assert(n < 2);
        return m_desc.m_extents[n];
    }
    size_t rows() const noexcept{return extent(0);}
    size_t cols() const noexcept{return extent(1);}

    Matrix_Ref<T,1> row(size_t i){
        assert(i < extent(0));
        Matrix_Slice<1> row;
        slice_dim<0>(i,m_desc,row);
        return {data(),row};
    }
    Matrix_Ref<const T,1> row(size_t i) const{
        assert(i < extent(0));
        Matrix_Slice<1> row;
        slice_dim<0>(i,m_desc,row);
        return {data(),row};
    }
    Matrix_Ref<T,1> operator[](size_t i){
        return row(i);
    }
    Matrix_Ref<const T,1> operator[](size_t i) const{
        return row(i);
    }
    T& operator()(size_t i,size_t j){
        assert(i < rows() && j < cols());
        return m_ptr[m_desc(i,j)];
    }
    const T& operator()(size_t i,size_t j) const{
        assert(i < rows() && j < cols());
        return m_ptr[m_desc(i,j)];
    }
    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<T,2>>
    operator()(Args... args){
        Matrix_Slice<2> d;
        d.m_size = 1;
        d.m_start = do_slice(m_desc,d,args...);
        return {data(),d};
    }
    template<typename... Args>
    std::enable_if_t<requesting_slice<Args...>(),Matrix_Ref<const T,2>>
    operator()(Args... args) const{
        Matrix_Slice<2> d;
        d.m_size = 1;
        do_slice(m_desc,d,args...);
        return {data(),d};
    }
};
template<typename T>
class Matrix_Ref<T,1>{
    Matrix_Slice<1> m_desc;
    T* m_ptr;

public:
    static constexpr size_t order = 1;
    using value_type = T;

    Matrix_Ref(T *p,const Matrix_Slice<1> &d):m_desc{d},m_ptr{p}{}
    //TODO Faltan constructores aqui...

    T* data() noexcept{return m_ptr;}
    const T* data() const noexcept{return m_ptr;}
    const Matrix_Slice<1>& descriptor() const noexcept { return m_desc; }
    size_t size() const noexcept {return m_desc.m_size;}

    T& row(size_t i){
        assert(i < size());
        return m_ptr[m_desc(i)];
    }
    const T& row(size_t i) const{
        assert(i < size());
        return m_ptr[m_desc(i)];
    }

    T& operator[](size_t i){
        return row(i);
    }
    const T& operator[](size_t i) const{
        return row(i);
    }
    T& operator()(size_t i){
        assert(i < size());
        return m_ptr[m_desc(i)];
    }
    const T& operator()(size_t i) const{
        assert(i < size());
        return m_ptr[m_desc(i)];
    }
    Matrix_Ref<T,1> operator()(const std::slice &s){
        Matrix_Slice<1> d;
        d.m_start = s.start();
        d.m_size = s.size();
        d.m_extents[0] = s.size();
        d.m_strides[0] = s.stride();
        //d._start = matrix_impl::do_slice(_desc,d,args...);
        return {data(),d};
    }
    Matrix_Ref<T,1> operator()(const std::slice &s) const{
        Matrix_Slice<1> d;
        d.m_start = s.start();
        d.m_size = s.size();
        d.m_extents[0] = s.size();
        d.m_strides[0] = s.stride();
        //d._start = matrix_impl::do_slice(_desc,d,args...);
        return {data(),d};
    }

};
template<typename T>
class Matrix_Ref<T,0>{
    T* m_ptr_elem;
public:
    static constexpr size_t order = 0;
    using value_type = T;

    Matrix_Ref(T* ptr_elem):m_ptr_elem(ptr_elem){}
    Matrix_Ref& operator=(const T &val){
        *m_ptr_elem = val;
        return *this;
    }

    T& row(size_t i) = delete;

    T& operator()(){return *m_ptr_elem;}
    const T& operator()()const {return *m_ptr_elem;}

    //Conversion operator from Matrix_Ref<T,0> to T*
    operator T*(){return m_ptr_elem;}
    operator const T*(){return m_ptr_elem;}
};
template<typename T>
inline void assing_slice_vals(const Matrix_Ref<T,1> &ref,Matrix_Ref<T,1> &mref){
    //static_assert(std::is_convertible_v<U,T>,"assign_slice_vals: Incompatible elements type.");
    assert(ref.size() == mref.size());
    for (size_t i = 0; i < mref.size(); ++i) mref(i) = ref(i);
}
template<typename T,size_t N>
inline void assing_slice_vals(const Matrix_Ref<T,N> &ref,Matrix_Ref<T,N> &mref){
    for (size_t i = 0; i < ref.descriptor()._extents[0];++i){
        Matrix_Ref<T,N-1> mref_aux = mref[i];
        assing_slice_vals(ref[i],mref_aux);
    }

}

#endif // MATRIX_REF_H
