##### 
# SET UP ENVIRONMENT

module load gcc/5.4.0
module unload spectrum_mpi
module load cuda/8.0.61-1
module load cmake
module list 
export OMPI_CXX=${HOME}/bin/nvcc_wrapper

SM=sm_60     # Kepler Gaming
OMP="yes"

# The directory containing the build scripts, this script and the src/ tree
TOPDIR=`pwd`

# Install directory
INSTALLDIR=${TOPDIR}/install/${SM}

# Source directory
SRCDIR=${TOPDIR}/../src

# Build directory
BUILDDIR=${TOPDIR}/build


### ENV VARS for CUDA/MPI
# These are used by the configure script to make make.inc
PK_CUDA_HOME=/sw/summitdev/cuda/8.0.54
PK_MPI_HOME=$HOME/openmpi_gcc5.4.0
PK_GPU_ARCH=${SM}

export PATH=${PK_MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${PK_MPI_HOME}/lib:$LD_LIBRARY_PATH

### OpenMP
# Open MP enabled
if [ "x${OMP}x" == "xyesx" ]; 
then 
 OMPFLAGS="-fopenmp -D_REENTRANT "
 OMPENABLE="--enable-openmp"
 INSTALLDIR=${INSTALLDIR}_omp
else
 OMPFLAGS=""
 OMPENABLE=""
fi

if [ ! -d ${INSTALLDIR} ];
then
  mkdir -p ${INSTALLDIR}
fi
### COMPILER FLAGS
PK_CXXFLAGS=${OMPFLAGS}" -g -O3 -std=c++11 "

PK_CFLAGS=${OMPFLAGS}" -g -O3 -std=gnu99"

### Make
MAKE="make -j 10"

### MPI
PK_CC=mpicc
PK_CXX=mpicxx
#PK_NVCCFLAGS="NVCCFLAGS=\"--keep --keep-dir=/scratch/bjoo/tmp\""
