# example build script on ARM with NEON support

export CXX=mpicxx
export CC=mpicc

cmake \
    -DCMAKE_CXX_FLAGS="-g -O3 -std=c++11 -fopenmp -march=armv8-a+fp+simd+crc"  \
    -DMG_USE_NEON=TRUE \
    -DQDPXX_DIR=${QDPXX_DIR} \
    -DEigen3_DIR=${Eigen3_DIR} \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
     ..
