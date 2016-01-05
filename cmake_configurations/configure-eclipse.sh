export CXX=icpc
export CC=icc

# -xCORE-AVX2 -fma for Haswell
#
cmake \
     -G"Eclipse CDT4 - Unix Makefiles" \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j8 \
     -DCMAKE_ECLIPSE_VERSION=4.5.0 \
    -DCMAKE_CXX_FLAGS="-g -O3 -xCORE-AVX2 -fma -std=c++11 -openmp -qopt-report=5 -opt-report-phase=vec"  \
    -DCMAKE_INSTALL_PREFIX=/home/bjoo/MG/install \
    -DMG_DEFAULT_LOGLEVEL=DEBUG \
     ../MGProject/mg
