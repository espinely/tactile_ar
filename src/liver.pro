QT += widgets gui

LIBS += -lGLU -lglut -lCGAL -lobj -lgmp -lopencv_core -lopencv_imgproc -lopencv_videoio -ltet

LIBPATH += /usr/local/lib/ 

SOURCES += \
    main.cpp \
    mainWindow.cpp \
    trackBall.cpp \
    GLmodel.cpp \
#    GLtexture.cpp \
    Model.cpp \
    Solver.cpp \
    GLWidget.cpp \
    Utils.cpp \
    ModelsListWidget.cpp \
    ModelContoursListWidget.cpp \
    ImageContoursListWidget.cpp

HEADERS += \
    mainWindow.h \
    trackBall.h \
    GLmodel.h \
#    GLtexture.h \
    Model.h \
    Solver.h \
    SolverCUDA.h \
    cuda_batched_solver/inverse.h \
    cuda_batched_solver/solve.h \
    cuda_batched_solver/operations.h \
    GLWidget.h \
    Utils.h \
    ModelsListWidget.h \
    ModelContoursListWidget.h \
    ImageContoursListWidget.h

# CUDA settings <-- may change depending on your system
CUDA_SOURCES += SolverCUDA.cu ./cuda_batched_solver/inverse.cu ./cuda_batched_solver/solve.cu
CUDA_SDK = "/usr/local/cuda/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda/"            # Path to cuda toolkit install

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = x64         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math


# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./


# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart -lcublas -lcusolver

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -DFERMI -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -DFERMI -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

#DISTFILES += \
#    cuda_batched_solver/inverse.cu \
#    cuda_batched_solver/solve.cu

DISTFILES += \
    shaders/phongBRDF.frag \
    shaders/phongBRDF.vert \
    shaders/quad.frag \
    shaders/quad.vert \
    shaders/contour.frag \
    shaders/contour.vert
