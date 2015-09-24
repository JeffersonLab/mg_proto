export CXX=mpiicpc
export CC=mpiicc

cmake \
     -G"Eclipse CDT4 - Unix Makefiles" \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
     -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_CXX_FLAGS="-g -O2 -xAVX -std=c++11 -openmp"  \
    -DQMP_DIR="/home/bjoo/package-8-7-15/avx/install/qmp" \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/MG/install \
    -DMG_DEFAULT_LOGLEVEL=DEBUG3 \
     ../MGProject
