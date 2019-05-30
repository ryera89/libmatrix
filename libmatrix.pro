#-------------------------------------------------
#
# Project created by QtCreator 2018-11-22T11:01:11
#
#-------------------------------------------------

QT       -= core gui

TARGET = libmatrix
TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt


QMAKE_CXXFLAGS += -std=c++1z #-fopenmp
# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    ndimmatrix/matrix.cpp \
    main.cpp \
    ndimmatrix/ndmatrix.cpp

HEADERS += \
    ndimmatrix/herm_matrix.h \
    ndimmatrix/matrix.h \
    ndimmatrix/matrix_impl.h \
    ndimmatrix/matrix_ref.h \
    ndimmatrix/matrix_slice.h \
    ndimmatrix/ndmatrix.h \
    ndimmatrix/sparse_matrix.h \
    ndimmatrix/symm_matrix.h

unix {
    target.path = /usr/lib
    #INSTALLS += target

    INCLUDEPATH += /opt/intel/parallel_studio_xe_2019.4.070/compilers_and_libraries_2019/linux/mkl/include/
    INCLUDEPATH += /usr/include/x86_64-linux-gnu/c++/8/
    LIBS += -L/opt/intel/parallel_studio_xe_2019.4.070/compilers_and_libraries_2019/linux/mkl/lib/intel64/ \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
    -L/opt/intel/parallel_studio_xe_2019.4.070/compilers_and_libraries_2019/linux/compiler/lib/intel64/  \
    -liomp5 -lpthread -lm #-dl
}

