export CXX=g++
export CC=gcc

cmake \
    -G"Eclipse CDT4 - Unix Makefiles" \
    -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j4 \
    -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_CXX_FLAGS="-g -Wall -Winline -O4 -std=c++11 -fopenmp -march=native"  \
    -DCMAKE_INSTALL_PREFIX=/Users/bjoo/Devel/MG/install/mg \
    -DQDPXX_DIR=~/package-2-5-16-qudamg/scalar/install/qdp++-double-scalar \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
     /Users/bjoo/Git/mg
