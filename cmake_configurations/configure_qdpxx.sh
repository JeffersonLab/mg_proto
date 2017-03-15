
export CXX=mpiicpc
export CC=mpiicc

cmake \
    -G"Eclipse CDT4 - Unix Makefiles" \
    -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
    -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_C_COMPILER_ID=Intel \
    -DCMAKE_CXX_COMPILER_ID=Intel \
    -DCMAKE_CXX_FLAGS="-g -O3 -std=c++11 -qopenmp -xCORE-AVX2"  \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/Devel/MG/install/mg \
    -DQDPXX_DIR=~/package-3-6-17/avx/install/qdp++-double \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
     ../mg_proto
