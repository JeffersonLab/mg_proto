/*
 * qphix_testutils.cpp
 *
 *  Created on: Oct 17, 2017
 *      Author: bjoo
 */

#include <gtest/gtest.h>
#include "qphix_testutils.h"
#include "utils/print_utils.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include <cmath>
using namespace QDP;
using namespace MG;

namespace MGTesting
{

template<typename QPhiXT>
inline
void DiffCBSpinorT(const LatticeFermion& s1, const QPhiXT& qphix_spinor, int cb, double tol,
    bool persite_tol=false)
{

  LatticeFermion s2;

  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  LatticeFermion diff=zero;
  diff[rb[cb]]=s1-s2;
  double vol_cb=static_cast<double>(Layout::vol())/2;


  double r_norm_cb = toDouble(sqrt(norm2(diff,rb[cb])));
  double r_norm_cb_per_site = r_norm_cb / vol_cb;
  MasterLog(INFO, "CB %d : || r || = %16.8e  || r ||/site = %16.8e",cb, r_norm_cb, r_norm_cb_per_site);

  if( persite_tol ) {
    ASSERT_LT( r_norm_cb_per_site, tol);
  }
  else {
    ASSERT_LT( r_norm_cb, tol);
  }
}

void DiffCBSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, int cb, double tol,
    bool persite_tol)
{
  DiffCBSpinorT(s1,qphix_spinor,cb,tol,persite_tol);

}

void DiffCBSpinor(const LatticeFermion& s1, const QPhiXSpinorF& qphix_spinor, int cb, double tol,
    bool persite_tol)
{
  DiffCBSpinorT(s1,qphix_spinor,cb,tol,persite_tol);
}

template<typename QPhiXT>
inline
void DiffSpinorT(const LatticeFermion& s1, const QPhiXT& qphix_spinor, double tol, bool persite_tol=false)
{
  LatticeFermion s2;
  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  LatticeFermion diff=s1-s2;
  double vol=static_cast<double>(Layout::vol());
  double vol_cb=vol/2;

  double r_norm_cb0 = toDouble(sqrt(norm2(diff,rb[0])));
  double r_norm_cb1 = toDouble(sqrt(norm2(diff,rb[1])));
  double r_norm_cb0_per_site = r_norm_cb0 / vol_cb;
  double r_norm_cb1_per_site = r_norm_cb1 / vol_cb;
  MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb0, r_norm_cb0_per_site);
  MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb1, r_norm_cb1_per_site);

  double r_norm = toDouble(sqrt(norm2(diff)));
  double r_norm_per_site = r_norm /vol;
  MasterLog(INFO, "Full: || r || = %16.8e || r ||/site = %16.8e", r_norm, r_norm_per_site);

  if( persite_tol) {
    ASSERT_LT(r_norm_per_site,tol);
  }
  else {
    ASSERT_LT( r_norm, tol);
  }
}




void DiffSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, double tol, bool persite_tol)
{
  DiffSpinorT(s1,qphix_spinor,tol,persite_tol);
}

void DiffSpinor(const LatticeFermion& s1, const QPhiXSpinorF& qphix_spinor, double tol, bool persite_tol)
{
  DiffSpinorT(s1,qphix_spinor,tol,persite_tol);
}


void DiffSpinor(const LatticeFermion& s1, const LatticeFermion& s2, double tol,
    bool persite_tol)
{
  LatticeFermion diff=s1-s2;
  double vol=static_cast<double>(Layout::vol());
  double vol_cb=vol/2;

  double r_norm_cb0 = toDouble(sqrt(norm2(diff,rb[0])));
  double r_norm_cb1 = toDouble(sqrt(norm2(diff,rb[1])));
  double r_norm_cb0_per_site = r_norm_cb0 / vol_cb;
  double r_norm_cb1_per_site = r_norm_cb1 / vol_cb;
  MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb0, r_norm_cb0_per_site);
  MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb1, r_norm_cb1_per_site);

  double r_norm = toDouble(sqrt(norm2(diff)));
  double r_norm_per_site = r_norm /vol;
  MasterLog(INFO, "Full: || r || = %16.8e || r ||/site = %16.8e", r_norm, r_norm_per_site);
  if( persite_tol ) {
    ASSERT_LT(r_norm_per_site,tol);
  }
  else {
    ASSERT_LT( r_norm, tol);
  }
}

void DiffSpinorRelative(const LatticeFermion& b, const LatticeFermion& Ax, double tol)
{
  LatticeFermion r=b-Ax;

  double r_norm_cb0 = toDouble(sqrt(norm2(r,rb[0])));
  double r_norm_cb1 = toDouble(sqrt(norm2(r,rb[1])));

  double b_norm_cb0 = toDouble(sqrt(norm2(b,rb[0])));
  double b_norm_cb1 = toDouble(sqrt(norm2(b,rb[1])));

  double rel_norm_cb0 = r_norm_cb0 / b_norm_cb0;
  double rel_norm_cb1 = r_norm_cb1 / b_norm_cb1;

  MasterLog(INFO, "CB 0 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb0, rel_norm_cb0);
  MasterLog(INFO, "CB 1 : || r || = %16.8e  || r ||/site = %16.8e", r_norm_cb1, rel_norm_cb1);

  double norm_r = toDouble(sqrt(norm2(r)));
  double norm_b = toDouble(sqrt(norm2(b)));
  double rel_norm_r = norm_r / norm_b;
  MasterLog(INFO, "Full: || r || = %16.8e || r ||/ || b || = %16.8e", norm_r, rel_norm_r);
  ASSERT_LT(  toDouble(rel_norm_r), tol);

}

template<typename QDPT, typename QPT>
inline
void DiffCBSpinorPerSiteT(const QDPT& s1, const QPT& qphix_spinor, int cb, double tol)
{
  QDPT s2;
  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  const int cbsites = rb[cb].numSiteTable();

  for(int site=0; site < cbsites; ++site) {
    for(int spin=0; spin < 4; ++spin) {
      for(int color=0; color < 3; ++color) {
        double diff_re = s2.elem(site).elem(spin).elem(color).real() - s1.elem(site).elem(spin).elem(color).real();
        ASSERT_LT( std::abs(diff_re), tol);
        double diff_im = s2.elem(site).elem(spin).elem(color).imag() - s1.elem(site).elem(spin).elem(color).imag();
        ASSERT_LT( std::abs(diff_im), tol);
      }
    }
  }

}


void DiffCBSpinorPerSite(const QDP::LatticeFermion& s1, const MG::QPhiXSpinor& qphix_spinor, int cb, double tol)
{
  DiffCBSpinorPerSiteT(s1,qphix_spinor,cb,tol);
}
void DiffCBSpinorPerSite(const QDP::LatticeFermion& s1, const MG::QPhiXSpinorF& qphix_spinor, int cb, double tol)
{
  DiffCBSpinorPerSiteT(s1,qphix_spinor,cb,tol);
}

void DiffSpinorPerSite(const QDP::LatticeFermion& s1, const MG::QPhiXSpinor& qphix_spinor, double tol)
{
  for(int cb=0; cb < 2; ++cb) {
    DiffCBSpinorPerSite(s1,qphix_spinor,cb,tol);
  }
}
void DiffSpinorPerSite(const QDP::LatticeFermion& s1, const MG::QPhiXSpinorF& qphix_spinor, double tol)
{
  for(int cb=0; cb < 2; ++cb) {
    DiffCBSpinorPerSite(s1,qphix_spinor,cb,tol);
  }
}
} // namespace


