
export CXX=mpicxx
export CC=mpicc
#export MPICH_CXX=nvcc_wrapper

/dist/cmake-3.3.0-Linux-x86_64/bin/cmake \
    -G"Eclipse CDT4 - Unix Makefiles" \
    -DKOKKOS_ENABLE_CUDA=TRUE \
    -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
    -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/Devel/MG/install \
    -DKOKKOS_HOST_ARCH="SNB" \
    -DKOKKOS_GPU_ARCH="Kepler37" \
    -DQDPXX_DIR=/home/bjoo/package-4-23-17/avx/install/qdp++-double \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
     ../mg_proto_kokkos



