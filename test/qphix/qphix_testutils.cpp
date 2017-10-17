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

using namespace QDP;
using namespace MG;

namespace MGTesting
{


void DiffCBSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, int cb, double tol)
{

  LatticeFermion s2;

  QPhiXSpinorToQDPSpinor(qphix_spinor,s2);

  LatticeFermion diff=zero;
  diff[rb[cb]]=s1-s2;
  double vol_cb=static_cast<double>(Layout::vol())/2;


  double r_norm_cb = toDouble(sqrt(norm2(diff,rb[cb])));
  double r_norm_cb_per_site = r_norm_cb / vol_cb;
  MasterLog(INFO, "CB %d : || r || = %16.8e  || r ||/site = %16.8e",cb, r_norm_cb, r_norm_cb_per_site);

  ASSERT_LT( r_norm_cb, tol);

}

void DiffSpinor(const LatticeFermion& s1, const QPhiXSpinor& qphix_spinor, double tol)
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
  ASSERT_LT( r_norm, tol);

}

void DiffSpinor(const LatticeFermion& s1, const LatticeFermion& s2, double tol)
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
  ASSERT_LT( r_norm, tol);

}

void DiffSpinorRelative(const LatticeFermion& b, const LatticeFermion& Ax, double tol)
{
  LatticeFermion r=b-Ax;
  double norm_r = toDouble(sqrt(norm2(r)));
  double norm_b = toDouble(sqrt(norm2(b)));
  double rel_norm_r = norm_r / norm_b;
  MasterLog(INFO, "Full: || r || = %16.8e || r ||/ || b || = %16.8e", norm_r, rel_norm_r);
  ASSERT_LT(  toDouble(rel_norm_r), tol);

}
} // namespace


