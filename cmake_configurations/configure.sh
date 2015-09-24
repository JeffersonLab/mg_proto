export CXX=mpiicpc
export CC=mpiicc

cmake \
    -DCMAKE_CXX_FLAGS="-g -O2 -xAVX -std=c++11 -openmp"  \
    -DQMP_DIR="/home/bjoo/package-8-7-15/avx/install/qmp" \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/MG/install \
    -DMG_DEFAULT_LOGLEVEL=DEBUG3 \
     ../MGProject
