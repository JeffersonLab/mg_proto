#ifndef VOL_AND_BLOCK_ARGS_H
#define VOL_AND_BLOCK_ARGS_H

#include <cstdlib>
#include <cstring>
#include "lattice/constants.h"
#include "utils/print_utils.h"

namespace MGTesting { 

class VolAndBlockArgs { 
public:
 VolAndBlockArgs(const IndexArray& _ldims, 
		 const IndexArray& _bdims, 
		 int _nvec, 
		 int _bthreads,
		 int _iter) 
   : ldims(_ldims), bdims(_bdims), nvec(_nvec), bthreads(_bthreads), iter(_iter) {}
  void ProcessArgs(int argc, char *argv[]) 
  {
    int i=0; 
    while( i < argc ) {
      if( std::strcmp( argv[i], "-ldims") == 0 ) { 
	ldims[0]=std::atoi(argv[i+1]);
	ldims[1]=std::atoi(argv[i+2]);
	ldims[2]=std::atoi(argv[i+3]);
	ldims[3]=std::atoi(argv[i+4]);
	i+=5;
      }
      else if( std::strcmp( argv[i], "-bdims") == 0 ) { 
	bdims[0]=std::atoi(argv[i+1]);
	bdims[1]=std::atoi(argv[i+2]);
	bdims[2]=std::atoi(argv[i+3]);
	bdims[3]=std::atoi(argv[i+4]);
	i+=5;
      }
      else if ( std::strcmp( argv[i], "-nvec") == 0 ) { 
	nvec = std::atoi(argv[i+1]);
	i+=2;
      }
      else if ( std::strcmp( argv[i], "-bthreads") == 0 ) { 
	bthreads=std::atoi(argv[i+1]);
	i+=2;
      }
      else if ( std::strcmp( argv[i], "-iter") == 0 ) { 
	iter = std::atoi(argv[i+1]);
	i+=2;
      }
      else{
	i++;
      }
    }
  }

  void Dump() { 
    IndexArray coarse_dims;
    int coarse_sites = 1;
    for(int mu=0; mu < 4; ++mu) { 
      coarse_dims[mu] = ldims[mu]/bdims[mu];
      coarse_sites *= coarse_dims[mu];
    }

    MasterLog(INFO, "Local Lattice Size = %d %d %d %d", ldims[0],ldims[1],ldims[2],ldims[3]);
    MasterLog(INFO, "Block Size         = %d %d %d %d", bdims[0],bdims[1],bdims[2],bdims[3]);
    MasterLog(INFO, "Coarse Latt.  Size = %d %d %d %d", coarse_dims[0], coarse_dims[1],coarse_dims[2],coarse_dims[3]);
    MasterLog(INFO, "Num Blocks         = %d", coarse_sites);
    MasterLog(INFO, "Num. Vectors       = %d", nvec);
    MasterLog(INFO, "Threads per block  = %d", bthreads);
    MasterLog(INFO, "Iterations         = %d", iter);
  }

  IndexArray ldims;
  IndexArray bdims;
  int nvec;
  int bthreads;
  int iter;
};
} // Namespace
#endif
