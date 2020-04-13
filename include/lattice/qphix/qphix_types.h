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

#include <qphix/invbicgstab.h>
#include <qphix/invmr.h>
#include <qphix/clover.h>
#include <qphix/invbicgstab.h>
#include <qphix/qdp_packer.h>
#include <qphix/unprec_solver_wrapper.h>
#include <utils/initialize.h>
#include <memory>

namespace MG {

template<typename FT>
using QPhiXGeomT = typename QPhiX::Geometry<FT,get_veclen<FT>(), QPHIX_SOALEN, QPHIX_COMPRESS12>;
using Geom = QPhiXGeomT<double>;
using GeomF = QPhiXGeomT<float>;

template<typename FT>
using QPhiXCBSpinorT = typename QPhiX::FourSpinorHandle<FT,get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXCBGaugeT = typename QPhiX::GaugeHandle<FT,get_veclen<FT>(),QPHIX_SOALEN, QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXCBCloverT = typename QPhiX::CloverHandle<FT, get_veclen<FT>(),QPHIX_SOALEN, QPHIX_COMPRESS12>;


template<typename FT>
using QPhiXFullSpinorT = typename QPhiX::FullSpinor<FT,get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXClovOpT = typename QPhiX::EvenOddCloverOperator<FT,get_veclen<FT>(), QPHIX_SOALEN, QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXBiCGStabT = typename QPhiX::InvBiCGStab<FT, get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXMRSolverT = typename QPhiX::InvMR<FT,get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXMRSmootherT = typename QPhiX::InvMRSmoother<FT,get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXEOPrecOpT = typename QPhiX::EvenOddLinearOperator<FT,get_veclen<FT>(),QPHIX_SOALEN,QPHIX_COMPRESS12>;

template<typename FT>
using QPhiXUnprecSolverT = typename QPhiX::UnprecSolverWrapper<FT, get_veclen<FT>(), QPHIX_SOALEN,QPHIX_COMPRESS12, QPhiXEOPrecOpT<FT>>;


using QPhiXCBSpinor = QPhiXCBSpinorT<double>;
using QPhiXCBGauge = QPhiXCBGaugeT<double>;
using QPhiXCBClover = QPhiXCBCloverT<double>;
using QPhiXFullSpinor = QPhiXFullSpinorT<double>;
using ClovOp = QPhiXClovOpT<double>;
using BiCGStab = QPhiXBiCGStabT<double>;
using QPhiXMRSmoother = QPhiXMRSmootherT<double>;
using EOPrecOp = QPhiXEOPrecOpT<double>;
using QPhiXUnprecSolver = QPhiXUnprecSolverT<double>;

using QPhiXCPSpinorF = QPhiXCBSpinorT<float>;
using QPhiXCBGaugeF = QPhiXCBGaugeT<float>;
using QPhiXCBCloverF = QPhiXCBCloverT<float>;
using QPhiXFullSpinorF = QPhiXFullSpinorT<float>;
using ClovOpF = QPhiXClovOpT<float>;
using BiCGStabF = QPhiXBiCGStabT<float>;
using QPhiXMRSmootherF = QPhiXMRSmootherT<float>;
using EOPrecOpF = QPhiXEOPrecOpT<float>;
using QPhiXUnprecSolverF = QPhiXUnprecSolverT<float>;

// Basic Geometry Utilizies
namespace MGQPhiX {
  bool IsGeomInitialized();
  void InitializeGeom(const LatticeInfo& info);
  template<typename FT>
  QPhiXGeomT<FT>& GetGeom();

  template<>
  Geom& GetGeom<double>();

  template<>
  GeomF& GetGeom<float>();
}

template<typename FT>
class QPhiXSpinorT {
public:

    using GeomT = QPhiXGeomT<FT>;

    QPhiXSpinorT(const LatticeInfo& info, IndexType ncol=1) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }
      for (IndexType col=0; col<ncol; ++col)
        _data.emplace_back(new QPhiXFullSpinorT<FT>(MGQPhiX::GetGeom<FT>()));
    }

    ~QPhiXSpinorT() {}

    inline IndexType GetNCol() const { return _data.size(); }

    inline
    QPhiXFullSpinorT<FT>& get(IndexType col) {
       return *_data[col];
    }

    inline
    const QPhiXFullSpinorT<FT>& get(IndexType col) const {
      return *_data[col];
    }

    inline
    QPhiXCBSpinorT<FT>& getCB(IndexType col, int cb) {
      return _data[col]->getCB(cb);
    }

    inline
    const QPhiXCBSpinorT<FT>& getCB(IndexType col, int cb) const {
      return _data[col]->getCB(cb);
    }

    inline
    const GeomT& getGeom() const {
      return static_cast<const QPhiXGeomT<FT>&>(MGQPhiX::GetGeom<FT>());
    }
#if 0
    GeomT& getGeom()  {
         return MGQPhiX::GetGeom<FT>();
    }
#endif

    inline
    const LatticeInfo& GetInfo() const {
      return _info;
    }

    inline
    FT& operator()(int col, int cb, int site, int spin, int color, int cmpx)
    {
      int osite = site / QPHIX_SOALEN;
      int isite = site % QPHIX_SOALEN;

      return (_data[col].get()->getCBData(cb))[osite][color][spin][cmpx][isite];
    }

    inline
    const FT& operator()(int col, int cb, int site, int spin, int color, int cmpx) const
     {
       int osite = site / QPHIX_SOALEN;
       int isite = site % QPHIX_SOALEN;

       return (_data[col].get()->getCBData(cb))[osite][color][spin][cmpx][isite];
     }
private:
    const LatticeInfo& _info;

    std::vector<std::unique_ptr<QPhiXFullSpinorT<FT>>> _data;
};

using QPhiXSpinor = QPhiXSpinorT<double>;
using QPhiXSpinorF = QPhiXSpinorT<float>;

template<typename FT>
class QPhiXGaugeT {
public:

    using GeomT = QPhiXGeomT<FT>;

    QPhiXGaugeT(const LatticeInfo& info) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }

      for(int cb=0; cb < 2; ++cb) {
        _data[cb].reset(new QPhiXCBGaugeT<FT>(MGQPhiX::GetGeom<FT>()));
      }
    }

    ~QPhiXGaugeT() {}

     QPhiXCBGaugeT<FT>& getCB(int cb) {
       return *(_data[cb]);
     }

     const QPhiXCBGaugeT<FT>& getCB(int cb) const {
       return *(_data[cb]);
     }

     const GeomT& getGeom() const {
         return MGQPhiX::GetGeom<FT>();
       }
#if 0
     GeomT& getGeom()  {
       return MGQPhiX::GetGeom<FT>();
     }
#endif
     const LatticeInfo& GetInfo() const {
       return _info;
     }

private:
     const LatticeInfo& _info;
     std::unique_ptr<QPhiXCBGaugeT<FT>> _data[2];
};

using QPhiXGauge = QPhiXGaugeT<double>;
using QPhiXGaugeF = QPhiXGaugeT<float>;

template<typename FT>
class QPhiXCloverT {
public:

    using GeomT = QPhiXGeomT<FT>;
    QPhiXCloverT(const LatticeInfo& info) : _info(info)
    {
      if( ! MGQPhiX::IsGeomInitialized() ) {
        MGQPhiX::InitializeGeom(info);
      }
      else {
        // check the info?
      }

      for(int cb=0; cb < 2; ++cb) {
        _data[cb].reset(new QPhiXCBCloverT<FT>(MGQPhiX::GetGeom<FT>()));
      }

      // Store the inverse
      _inv.reset(new QPhiXCBCloverT<FT>(MGQPhiX::GetGeom<FT>()));
    }

    ~QPhiXCloverT() {}

     QPhiXCBCloverT<FT>& getCB(int cb) {
       return *(_data[cb]);
     }

     const QPhiXCBCloverT<FT>& getCB(int cb) const {
       return *(_data[cb]);
     }

     QPhiXCBCloverT<FT>& getInv() {
       return *_inv;
     }

     const QPhiXCBCloverT<FT>& getInv() const {
       return *_inv;
     }

     const GeomT& getGeom() const {
         return MGQPhiX::GetGeom<FT>();
       }

#if 0
     GeomT& getGeom()  {
       return MGQPhiX::GetGeom<FT>();
     }
#endif

     const LatticeInfo& GetInfo() const {
       return _info;
     }
private:
   const LatticeInfo& _info;
   std::unique_ptr<QPhiXCBCloverT<FT>> _data[2];
   std::unique_ptr<QPhiXCBCloverT<FT>> _inv;
};


using QPhiXClover = QPhiXCloverT<double>;
using QPhiXCloverF = QPhiXCloverT<float>;
}




#endif /* INCLUDE_LATTICE_QPHIX_QPHIX_TYPES_H_ */
