/*
 * qphix_types.h
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_TYPES_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_TYPES_H_

#include <lattice/lattice_info.h>
#include <lattice/qphix/qphix_veclen.h>

#include <qphix/qphix_config.h>
#include <qphix/full_spinor.h>
#include <qphix/abs_solver.h>
#include <qphix/abs_solver.h>
#include <qphix/invbicgstab.h>
#include <qphix/clover.h>
#include <qphix/invbicgstab.h>
#include <qphix/qdp_packer.h>
#include <qphix/unprec_solver_wrapper.h>
#include <utils/initialize.h>
#include <memory>

namespace MG {

using Geom = QPhiX::Geometry<double, VECLEN_DP, QPHIX_SOALEN, false>;


using QPhiXCBSpinor = QPhiX::FourSpinorHandle<double,VECLEN_DP,QPHIX_SOALEN,false>;
using QPhiXCBGauge = QPhiX::GaugeHandle<double,VECLEN_DP,QPHIX_SOALEN, false>;
using QPhiXCBClover = QPhiX::CloverHandle<double, VECLEN_DP,QPHIX_SOALEN, false>;



using QPhiXFullSpinor = QPhiX::FullSpinor<double,VECLEN_DP,QPHIX_SOALEN,false>;

using ClovOp = QPhiX::EvenOddCloverOperator<double, VECLEN_DP, QPHIX_SOALEN, false>;
using BiCGStab = QPhiX::InvBiCGStab<double,VECLEN_DP,QPHIX_SOALEN,false>;
using EOPrecOp = QPhiX::EvenOddLinearOperator<double,VECLEN_DP,QPHIX_SOALEN,false>;
using QPhiXUnprecSolver = QPhiX::UnprecSolverWrapper<double,VECLEN_DP,
                                                    QPHIX_SOALEN,false,EOPrecOp>;
// Basic Geometry Utilizies
namespace MGQPhiX {
  bool IsGeomInitialized();
  void InitializeGeom(const LatticeInfo& info);
  Geom& GetGeom();
}

class QPhiXSpinor {
public:
    QPhiXSpinor(const LatticeInfo& info) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }
      _data.reset(new QPhiXFullSpinor(MGQPhiX::GetGeom()));
    }

    ~QPhiXSpinor() {}

    QPhiXFullSpinor& get() {
       return *_data;
    }

    const QPhiXFullSpinor& get() const {
      return *_data;
    }

    QPhiXCBSpinor& getCB(int cb) {
      return _data->getCB(cb);
    }

    const QPhiXCBSpinor& getCB(int cb) const {
      return _data->getCB(cb);
    }

    const Geom& getGeom() const {
      return MGQPhiX::GetGeom();
    }

    const LatticeInfo& GetInfo() const {
      return _info;
    }
private:
    const LatticeInfo& _info;

    std::unique_ptr<QPhiXFullSpinor> _data;
};

class QPhiXGauge {
public:
    QPhiXGauge(const LatticeInfo& info) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }

      for(int cb=0; cb < 2; ++cb) {
        _data[cb].reset(new QPhiXCBGauge(MGQPhiX::GetGeom()));
      }
    }

    ~QPhiXGauge() {}

     QPhiXCBGauge& getCB(int cb) {
       return *(_data[cb]);
     }

     const QPhiXCBGauge& getCB(int cb) const {
       return *(_data[cb]);
     }

     const Geom& getGeom() const {
         return MGQPhiX::GetGeom();
       }

     const LatticeInfo& GetInfo() const {
       return _info;
     }

private:
     const LatticeInfo& _info;
     std::unique_ptr<QPhiXCBGauge> _data[2];
};

class QPhiXClover {
public:
    QPhiXClover(const LatticeInfo& info) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }

      for(int cb=0; cb < 2; ++cb) {
        _data[cb].reset(new QPhiXCBClover(MGQPhiX::GetGeom()));
      }

      // Store the inverse
      _inv.reset(new QPhiXCBClover(MGQPhiX::GetGeom()));
    }

    ~QPhiXClover() {}

     QPhiXCBClover& getCB(int cb) {
       return *(_data[cb]);
     }

     const QPhiXCBClover& getCB(int cb) const {
       return *(_data[cb]);
     }

     QPhiXCBClover& getInv() {
       return *_inv;
     }

     const QPhiXCBClover& getInv() const {
       return *_inv;
     }

     const Geom& getGeom() const {
         return MGQPhiX::GetGeom();
       }

     const LatticeInfo& GetInfo() const {
       return _info;
     }
private:
   const LatticeInfo& _info;
   std::unique_ptr<QPhiXCBClover> _data[2];
   std::unique_ptr<QPhiXCBClover> _inv;
};

}




#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_TYPES_H_ */
