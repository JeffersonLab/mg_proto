/*
 * geom_utils.h
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#include "lattice/qphix/qphix_types.h"
#include "utils/initialize.h"
#include "utils/print_utils.h"
#include <memory>
namespace MG {
  namespace MGQPhiX {

    // This should be initialized to NULL
    std::unique_ptr<Geom> _theGeom;
    std::unique_ptr<GeomF> _theGeomF;
    //! Check whether geometry is set
    // is initialized. We assume it is initialized only
    // once as we will have only 1 fine lattice in QPhiX
    // The coarse lattices will be kokkos

    bool IsGeomInitialized(void) {
       return _theGeom ? true : false;
    }

    //! Initialize the Geometry
    void InitializeGeom(const LatticeInfo& info)
    {
      const IndexArray& latdims = info.GetLatticeDimensions();
      int lattSize[4] = { latdims[0],
           latdims[1],
           latdims[2],
           latdims[3] };

      // Get QPhiX Command Line args
       QPhiX::QPhiXCLIArgs& CLI = MG::getQPhiXCLIArgs();

      _theGeom.reset(new Geom(lattSize,
          CLI.getBy(),
          CLI.getBz(),
          CLI.getNCores(),
          CLI.getSy(),
          CLI.getSz(),
          CLI.getPxy(),
          CLI.getPxyz(),
          CLI.getMinCt(),
          true));

      _theGeomF.reset(new GeomF(lattSize,
                CLI.getBy(),
                CLI.getBz(),
                CLI.getNCores(),
                CLI.getSy(),
                CLI.getSz(),
                CLI.getPxy(),
                CLI.getPxyz(),
                CLI.getMinCt(),
                true));

    }

    //! Get back a geometry if it is initialized
    //  otherwise barf
    template<>
    Geom& GetGeom<double>(void)
    {
      if ( ! _theGeom ) {
        MasterLog(ERROR,"QPhiX Geometry is uninitialized");
      }
      return *_theGeom;
    }

    //! Get back a geometry if it is initialized
     //  otherwise barf
     template<>
     GeomF& GetGeom<float>(void)
     {
       if ( ! _theGeom ) {
         MasterLog(ERROR,"QPhiX Geometry is uninitialized");
       }
       return *_theGeomF;
     }
  }

}
