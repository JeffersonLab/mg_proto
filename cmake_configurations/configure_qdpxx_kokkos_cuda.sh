source ./env.sh

export OMPI_CXX=$HOME/bin/nvcc_wrapper
CXX="${PK_CXX}" CC="${PK_CC}" CXXFLAGS="${PK_CXXFLAGS} -lineinfo -Xptxas=-v" cmake \
    -G"Eclipse CDT4 - Unix Makefiles" \
    -DMG_USE_KOKKOS=TRUE \
    -DMG_USE_QPHIX=FALSE \
    -DKOKKOS_ENABLE_CUDA=TRUE \
    -DKOKKOS_ENABLE_CUDA_LAMBDA=TRUE \
    -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
    -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/MGKokkosGPUTest/install_omp/mg \
    -DKOKKOS_HOST_ARCH="Power8" \
    -DKOKKOS_GPU_ARCH="Pascal60" \
    -DQDPXX_DIR=$HOME/package-6-21-17-v2/quda/install/sm_60_omp/qdp++-double \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
    -DCMAKE_BUILD_TYPE=DEBUG \
     ../mg_proto_vec3



