#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include "ndmatrix.h"
#include <iostream>
//#include "mkl_spblas.h"


inline void check_sparse_operation_status(sparse_status_t status){
    switch (status) {
    case SPARSE_STATUS_ALLOC_FAILED: throw("Internal memory allocation failed.");
    case SPARSE_STATUS_INVALID_VALUE:throw(" The input parameters contain an invalid value.");
    case SPARSE_STATUS_INTERNAL_ERROR: throw("An error in algorithm implementation occurred.");
    case SPARSE_STATUS_NOT_SUPPORTED: throw("The requested operation is not supported.");
    case SPARSE_STATUS_NOT_INITIALIZED: throw("The routine encountered an empty handle or matrix array.");
    case SPARSE_STATUS_EXECUTION_FAILED: throw("Execution failed.");
    default: break; //status == SPARSE_STATUS_SUCCESS
    }
}

typedef int int_t;
template<typename T>
class Matrix<T,2,MATRIX_TYPE::CSR>{
private:
    uint32_t m_rows;
    uint32_t m_cols;
    vector<int_t> m_rows_start; //most recent representation
    vector<int_t> m_rows_end; //most recent representation
    vector<int_t> m_rowsIndex; //representacion 3 Array
    vector<int_t> m_columns; //index de las columnas
    vector<T> m_elems; //elementos
    sparse_matrix_t m_handler;
    static constexpr T zero_val = T();

    void shrinkToFitDataVectors(){
        m_rows_start.shrink_to_fit();
        m_rows_end.shrink_to_fit();
        m_rowsIndex.shrink_to_fit();
        m_columns.shrink_to_fit();
        m_elems.shrink_to_fit();
    }
    void initSparseMatrixHandler(){
        if (nnz() == 0) return;
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        if constexpr (is_same_v<T,float>){
            status = mkl_sparse_s_create_csr(&m_handler,m_index,m_rows,m_cols,m_rows_start.data(),m_rows_end.data(),
                                             m_columns.data(),m_elems.data());
        }else if constexpr (is_same_v<T,double>) {
            status = mkl_sparse_d_create_csr(&m_handler,m_index,m_rows,m_cols,m_rows_start.data(),m_rows_end.data(),
                                             m_columns.data(),m_elems.data());
        }else if constexpr (is_same_v<T,complex<float>>) {
            status = mkl_sparse_c_create_csr(&m_handler,m_index,m_rows,m_cols,m_rows_start.data(),m_rows_end.data(),
                                             m_columns.data(),m_elems.data());
        }else if constexpr (is_same_v<T,complex<double>>) {
            status =  mkl_sparse_z_create_csr(&m_handler,m_index,m_rows,m_cols,m_rows_start.data(),m_rows_end.data(),
                                             m_columns.data(),m_elems.data());
        }
        check_sparse_operation_status(status);
    }
public:
    static constexpr size_t order = 2;
    static constexpr  sparse_index_base_t m_index = SPARSE_INDEX_BASE_ZERO;   //base index
    static constexpr MATRIX_TYPE matrix_type = MATRIX_TYPE::CSR;
    using value_type = T;
    using iterator = typename vector<T>::iterator;
    using const_iterator = typename vector<T>::const_iterator;

    Matrix() = default;
    ~Matrix(){if (nnz() != 0) mkl_sparse_destroy(m_handler);} //WARNING: may be memory leaks //= default;
    //Move constructor and assignment
    Matrix(Matrix&& other):m_rows(other.m_rows),m_cols(other.m_cols),m_rows_start(move(other.m_rows_start)),
                             m_rows_end(move(other.m_rows_end)),m_rowsIndex(move(other.m_rowsIndex)),m_columns(move(other.m_columns)),
                             m_elems(move(other.m_elems)){
        initSparseMatrixHandler();
    } //= default;
    Matrix& operator=(Matrix&& other){
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_index = other.m_index;
    m_rows_start = move(other.m_rows_start);
    m_rows_end = move(other.m_rows_end);
    m_rowsIndex = move(other.m_rowsIndex);
    m_columns = move(other.m_columns);
    m_elems = move(other.m_elems);
    initSparseMatrixHandler();
    return *this;
} //= default; //= default;
    //Copy constructor and assignment
    Matrix(const Matrix& other):m_rows(other.m_rows),m_cols(other.m_cols),m_rows_start(other.m_rows_start),
                                  m_rows_end(other.m_rows_end),m_rowsIndex(other.m_rowsIndex),m_columns(other.m_columns),
                                  m_elems(other.m_elems){

        initSparseMatrixHandler();
    } //= default;
    Matrix& operator=(const Matrix& other){
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_index = other.m_index;
        m_rows_start = other.m_rows_start;
        m_rows_end = other.m_rows_end;
        m_rowsIndex = other.m_rowsIndex;
        m_columns = other.m_columns;
        m_elems = other.m_elems;
        initSparseMatrixHandler();
        return *this;
    } //= default;


    Matrix(uint32_t rows,uint32_t cols,const vector<int_t> &rows_start,const vector<int_t> &rows_end,
           const vector<int_t> &columns,const vector<T> &vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),
                                                                  m_rows_end(rows_end),m_rowsIndex(rows_start),m_columns(columns),m_elems(vals){

        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
        m_rowsIndex.push_back(m_elems.size());
        shrinkToFitDataVectors();
        initSparseMatrixHandler(); //inicializando el handler
    }

    Matrix(uint32_t rows,uint32_t cols,vector<int_t> &&rows_start,vector<int_t> &&rows_end,vector<int_t> &&columns,
           vector<T> &&vals):m_rows(rows),m_cols(cols),m_rows_start(rows_start),m_rows_end(rows_end),m_rowsIndex(m_rows_start),
                               m_columns(columns), m_elems(vals){
        assert(m_columns.size() == m_elems.size() && m_rows == m_rows_start.size() && m_rows == m_rows_end.size());
        m_rowsIndex.push_back(m_elems.size());
        shrinkToFitDataVectors();
        initSparseMatrixHandler(); //inicializando el handler
    }

    Matrix(const Matrix<T,2> &m):m_rows(m.rows()),m_cols(m.cols()),m_rows_start(m_rows),
                                    m_rows_end(m_rows){
        for (size_t i = 0; i < m_rows; ++i){
            bool row_first_nonzero = true;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (row_first_nonzero){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        row_first_nonzero = false;
                    }
                }
            }
            if (row_first_nonzero){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}
        }
        m_rowsIndex = m_rows_start;
        m_rowsIndex.push_back(m_elems.size());
        shrinkToFitDataVectors();
        initSparseMatrixHandler();
    }

    Matrix& operator=(const Matrix<T,2> &m){
        m_rows = m.rows();
        m_cols = m.cols();
        m_rows_start.resize(m_rows);
        m_rows_end.resize(m_rows);
        m_columns.clear();
        m_elems.clear();
        for (size_t i = 0; i < m_rows; ++i){
            bool row_first_nonzero = true;
            for (size_t j = 0; j < m_cols; ++j){
                T val = m(i,j);
                if (val != T()){ //si el valor es distinto de cero
                    m_elems.push_back(val);
                    m_columns.push_back(j);
                    if (row_first_nonzero){ //se ejecuta maximo solo una ves del loop principal i
                        m_rows_start[i] = m_elems.size()-1;
                        row_first_nonzero = false;
                    }
                }
            }
            if (row_first_nonzero){ //una fila llena de zeros
                m_rows_start[i] = m_elems.size();
                m_rows_end[i] = m_elems.size();
            }else{ m_rows_end[i] = m_elems.size();}
        }

        m_rowsIndex = m_rows_start;
        m_rowsIndex.push_back(m_elems.size());
        shrinkToFitDataVectors();
        initSparseMatrixHandler();
        return *this;
    }

    size_t nnz() const{return m_elems.size();}
    uint32_t rows() const{return m_rows;}
    uint32_t cols() const{return m_cols;}

    Matrix operator-() const{
        Matrix R(*this);
        for_each(R.m_elems.begin(),R.m_elems.end(),[](T &ele){ele = -ele;});
        return R;
    }
    Matrix& operator*=(const T &val){
        for_each(m_elems.begin(),m_elems.end(),[&val](T &ele){ele*=val;});
        return *this;
    }
    Matrix& operator/=(const T &val){
        for_each(m_elems.begin(),m_elems.end(),[&val](T &ele){ele/=val;});
        return *this;
    }


    //TODO: revisar el algoritmo aca...
    const T& operator()(size_t i,size_t j) const{
        assert(i < m_rows && j < m_cols);
        int_t beg = m_rows_start[i];
        int_t end = m_rows_end[i];

        if (beg == end) return zero_val; //row i is full of 0
        if (j < m_columns[beg] || j > m_columns[end-1]) return zero_val;

        vector<int_t>::const_iterator first = m_columns.cbegin()+beg;
        vector<int_t>::const_iterator last = m_columns.cbegin()+end;
        vector<int_t>::const_iterator it;
        it = lower_bound(first,last,j);
        if (it != last && *it == j){
            int64_t pos = std::distance(m_columns.begin(),it);
            return m_elems[pos];
        }
        return zero_val;
    }
    //TODO: revisar el algoritmo aca... pensar ordenar el vector de indices de columna de entrada
    Matrix operator()(const vector<uint32_t> &iindex,const vector<uint32_t> &jindex) const{
        uint32_t nrow = iindex.size();
        uint32_t ncol = jindex.size();
        vector<T> elems;
        elems.reserve(m_elems.size()); //avoiding new memory allocation
        vector<int_t> columns;
        columns.reserve(m_elems.size());
        vector<int_t> pointerB(nrow);
        vector<int_t> pointerE(nrow);

        for (uint32_t i = 0; i < nrow; ++i){
            bool rfirst_inclusion =  true;
            uint32_t ii = iindex[i];
            int_t ibeg = m_rows_start[ii];
            int_t iend = m_rows_end[ii];
            if (ibeg == iend){ //fila todos zeros
                pointerB[i] = elems.size(); pointerE[i] = elems.size();
                continue;
            }
            //int_t icolb = m_columns[ibeg];
            //int_t icole = m_columns[iend-1];
            for (int_t j = 0; j < ncol; ++j){
                int_t jj = jindex[j];
                vector<int_t>::const_iterator first = m_columns.cbegin()+ibeg;
                vector<int_t>::const_iterator last = m_columns.cbegin()+iend;
                vector<int_t>::const_iterator it;
                it = lower_bound(first,last,jj);
                if (it != last && *it == jj){ //values found
                    int64_t pos = std::distance(m_columns.begin(),it);
                    T val = m_elems[pos];
                    elems.push_back(val);
                    columns.push_back(j);
                    if (rfirst_inclusion){ //se ejecuta maximo solo una ves del loop principal i
                        pointerB[i] = elems.size()-1;
                        rfirst_inclusion = false;
                    }
                }
            }
            if (rfirst_inclusion){ //una fila llena de zeros
                pointerB[i] = elems.size(); pointerE[i] = elems.size();
            }else{ pointerE[i] = elems.size();}
        }
        return Matrix(nrow,ncol,pointerB,pointerE,columns,elems);
    }
    void printData() const{
        for (size_t i = 0; i < m_rows_start.size(); ++i){
            uint32_t beg = m_rows_start[i];
            uint32_t end = m_rows_end[i];
            for (;beg<end;++beg){
                std::cout << "(" << i << "," << m_columns[beg] << ") " << m_elems[beg] << "\n";
            }
        }
    }

    //base one data
    vector<int_t> indexBasedOneColumns() const{
        vector<int_t> oneBasesIndex(m_columns);
        for_each(oneBasesIndex.begin(),oneBasesIndex.end(),[](int_t &index){++index;});
        return oneBasesIndex;
    }
    vector<int_t> indexBasedOneRowsStart() const{
        vector<int_t> oneBasesIndex(m_rows_start);
        for_each(oneBasesIndex.begin(),oneBasesIndex.end(),[](int_t &index){++index;});
        return oneBasesIndex;
    }
    vector<int_t> indexBasedOneRowsEnd() const{
        vector<int_t> oneBasesIndex(m_rows_end);
        for_each(oneBasesIndex.begin(),oneBasesIndex.end(),[](int_t &index){++index;});
        return oneBasesIndex;
    }
    vector<int_t> indexBasedOneRowsIndex() const{
        vector<int_t> oneBasesIndex(m_rowsIndex);
        for_each(oneBasesIndex.begin(),oneBasesIndex.end(),[](int_t &index){++index;});
        return oneBasesIndex;
    }

    const vector<T>& values() const{return m_elems;}
    const vector<int_t>& columns() const{return m_columns;}
    const vector<int_t>& rowsStart() const{return m_rows_start;}
    const vector<int_t>& rowsEnd() const{return m_rows_end;}
    const vector<int_t>& rowsIndex() const{return m_rowsIndex;} //rowIndex for 3 array variation
    const sparse_matrix_t& sparse_matrix_handler() const{return m_handler;} //sparse matrix handler

    vector<int_t>::iterator columnsBegin(){return m_columns.begin();}
    vector<int_t>::const_iterator columnsBegin() const {return m_columns.cbegin();}
    vector<int_t>::iterator columnsEnd(){return m_columns.end();}
    vector<int_t>::const_iterator columnsEnd() const {return m_columns.cend();}

    vector<int_t>::iterator rowsStartBegin(){return m_rows_start.begin();}
    vector<int_t>::const_iterator rowsStartBegin() const{return m_rows_start.cbegin();}
    vector<int_t>::iterator rowsStartEnd(){return m_rows_start.end();}
    vector<int_t>::const_iterator rowsStartEnd() const{return m_rows_start.cend();}

    vector<int_t>::iterator rowsEndBegin(){return m_rows_end.begin();}
    vector<int_t>::const_iterator rowsEndBegin() const{return m_rows_end.cbegin();}
    vector<int_t>::iterator rowsEndEnd(){return m_rows_end.end();}
    vector<int_t>::const_iterator rowsEndEnd() const{return m_rows_end.cend();}

    //For 3 array variation
    vector<int_t>::iterator rowsIndexBegin(){return m_rowsIndex.begin();}
    vector<int_t>::const_iterator rowsIndexBegin() const{return m_rowsIndex.cbegin();}
    vector<int_t>::iterator rowsIndexEnd(){return m_rowsIndex.end();}
    vector<int_t>::const_iterator rowsIndexEnd() const{return m_rowsIndex.cend();}

    iterator valuesBegin(){return m_elems.begin();}
    const_iterator valuesBegin() const{return m_elems.cbegin();}
    iterator valuesEnd(){return m_elems.end();}
    const_iterator valuesEnd() const{return m_elems.cend();}

    int_t* columnsData(){return m_columns.data();}
    const int_t* columnsData() const {return m_columns.data();}

    int_t* rowsStartData(){return m_rows_start.data();}
    const int_t* rowsStartData() const{return m_rows_start.data();}

    int_t* rowsEndData(){return m_rows_end.data();}
    const int_t* rowsEndData() const{return m_rows_end.data();}

    //3 array variation
    int_t* rowsIndexData(){return m_rowsIndex.data();}
    const int_t* rowsIndexData() const{return m_rowsIndex.data();}

    T* valuesData(){return m_elems.data();}
    const T* valuesData() const{return m_elems.data();}
};
#endif // SPARSE_MATRIX_H
