#include "matrix.h"

///*template<typename T>
//std::ostream& operator << (std::ostream& os,const Matrix<T,2> &m){
//    //std::ios_base::fmtflags ff = std::ios::scientific;
//    //ff |= std::ios::showpos;
//    //os.setf(ff);
//    for (size_t i = 0; i < m.rows(); ++i){
//        for (size_t j = 0; j < m.columns(); ++j)
//            os << m(i,j) << '\t' ;
//        os << '\n';
//    }
//    //os.unsetf(ff);
//    return os;
//}*/
//template<typename T>
//std::ostream& operator << (std::ostream& os,const Matrix<T,1> &m){
//    //std::ios_base::fmtflags ff = std::ios::scientific;
//    //ff |= std::ios::showpos;
//    //os.setf(ff);
//    for (size_t i = 0; i < m.size(); ++i){
//        os << m(i) << '\n';
//    }
//    //os.unsetf(ff);
//    return os;
//}
//template<typename T>
//std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,2> &m){
//    if (ofs.is_open()){
//        std::ios_base::fmtflags ff = std::ios::scientific;
//        ff |= std::ios::showpos;
//        ofs.setf(ff);
//        for (size_t i = 0; i < m.rows(); ++i){
//            for (size_t j = 0; j < m.cols(); ++j)
//                ofs << m(i,j) << '\t' ;
//            ofs << std::endl;
//        }
//        ofs.unsetf(ff);
//    }
//    return ofs;
//}
//template<typename T>
//std::ofstream& operator << (std::ofstream& ofs,const Matrix<T,1> &m){
//    if (ofs.is_open()){
//        std::ios_base::fmtflags ff = std::ios::scientific;
//        ff |= std::ios::showpos;
//        ofs.setf(ff);
//        for (size_t i = 0; i < m.size(); ++i){
//            ofs << i << "**" << m(i) << '\n';
//        }
//        ofs.unsetf(ff);
//    }
//    return ofs;
//}

//template<typename T>
//std::ostream& operator << (std::ostream& os,const Matrix<T,4> &m){
//    //if (ofs.is_open()){
//        std::ios_base::fmtflags ff = std::ios::scientific;
//        ff |= std::ios::showpos;
//        os.setf(ff);
//        for (size_t i = 0; i < m.extent(3); ++i){
//            for (size_t j = 0; j < m.extent(2); ++j){
//                os << "(:,:," << j << ", " << i << ") \n";
//                for (size_t k = 0; k < m.extent(0); ++k){
//                    for (size_t l = 0; l < m.extent(1); ++l){
//                        os << m(k,l,j,i) << '\t' ;
//                    }
//                    os << "\n";
//                }
//            }
//        }
//        os.unsetf(ff);
//        return os;
//    //}
//}
