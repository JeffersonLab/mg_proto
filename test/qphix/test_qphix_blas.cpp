/*
 * test_qphix_interface.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#include <gtest/gtest.h>
#include "../test_env.h"
#include "lattice/qphix/qphix_types.h"
#include "lattice/qphix/qphix_qdp_utils.h"
#include "./qphix_testutils.h"
#include "lattice/qphix/qphix_blas_wrappers.h"

#include "lattice/constants.h"
#include "../qdpxx/qdpxx_latticeinit.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"

#include <cmath>
using namespace QDP;
using namespace MG;
using namespace MGTesting;


TEST(TESTQPhiXBLAS, TestXmYNorm2Vec)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  LatticeFermion x; gaussian(x);
  LatticeFermion y; gaussian(y);

  LatticeInfo info(latdims);
  QPhiXSpinor q_x(info);
  QPhiXSpinor q_y(info);

  QDPSpinorToQPhiXSpinor(x,q_x,0);
  QDPSpinorToQPhiXSpinor(y,q_y,0);

  x -= y;
  Double n = norm2(x);

  double q_norm2 = XmyNorm2Vec(q_x,q_y)[0];
  DiffSpinor(x,q_x,1.0e-14);
  DiffSpinor(y,q_y,1.0e-14);
  double absdiff = std::abs(q_norm2 - toDouble(n));
  ASSERT_LT( absdiff, 1.0e-9);


}

TEST(TESTQPhiXBLAS, TestNorm2Vec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);

     LatticeInfo info(latdims);
     QPhiXSpinor qphix_x(info);

     QDPSpinorToQPhiXSpinor(x,qphix_x,0);

     Double n = norm2(x);
     double qphix_norm = Norm2Vec(qphix_x)[0];
     double absdiff = std::abs(qphix_norm - toDouble(n));
     ASSERT_LT( absdiff, 1.0e-9);


}

TEST(TESTQPhiXBLAS, TestInnerProductVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);

     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     DComplex iprod = QDP::innerProduct(x,y);

     std::complex<double> q_iprod = InnerProductVec(q_x,q_y)[0];

     double realnorm = std::abs ( std::real(q_iprod) - toDouble(real(iprod)) );
     double imnorm = std::abs ( std::imag(q_iprod) - toDouble(imag(iprod)) );

     ASSERT_LT( realnorm, 1.0e-9);
     ASSERT_LT( imnorm, 1.0e-9);

}


TEST(TESTQPhiXBLAS, TestZeroVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeFermion z = zero;

     LatticeInfo info(latdims);
     QPhiXSpinor q_z(latdims);
     ZeroVec(q_z);
     DiffSpinor(z,q_z,1.0e-14);

}

TEST(TESTQPhiXBLAS, TestCopyVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion in; gaussian(in);

     LatticeInfo info(latdims);
     QPhiXSpinor q_in(info);
     QPhiXSpinor q_copy(info);
     QDPSpinorToQPhiXSpinor(in, q_in, 0);

     CopyVec(q_copy,q_in);
     DiffSpinor(in,q_copy, 1.0e-14);

}

TEST(TESTQPhiXBLAS, TestZAxpyVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Complex a=Real(1.5);
     a += timesI(Real(-0.261));

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<std::complex<double>> alpha(1, std::complex<double>(toDouble(real(a)), toDouble(imag(a))));

     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,1.0e-13);
     DiffSpinor(y,q_y,1.0e-13);



}


TEST(TESTQPhiXBLAS, TestSAxpyVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Real a=Real(1.5);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<double> alpha(1, toDouble(a));


     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,5.0e-6);
     DiffSpinor(y,q_y,5.0e-6);

}
TEST(TESTQPhiXBLAS, TestCAxpyVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Complex a=Real(1.5);
     a += timesI(Real(-0.261));

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<std::complex<double>> alpha(1, std::complex<double>(toDouble(real(a)), toDouble(imag(a))));

     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,5.0e-6);
     DiffSpinor(y,q_y,5.0e-6);

}

TEST(TESTQPhiXBLAS, TestGaussianVec)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     QDP::Seed seed;
     QDP::RNG::savern(seed);
     LatticeFermion x; gaussian(x);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QDP::RNG::setrn(seed);

     Gaussian(q_x);

     DiffSpinor(x,q_x,1.0e-14);
}

TEST(TESTQPhiXBLAS, TestYPeqX)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     YpeqXVec(q_x,q_y);
     y += x;
     DiffSpinorPerSite(x,q_x,1.0e-14);
     DiffSpinorPerSite(y,q_y,1.0e-14);
}

TEST(TESTQPhiXBLAS, TestYMeqX)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinor q_x(info);
     QPhiXSpinor q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     YmeqXVec(q_x,q_y);
     y -= x;
     DiffSpinorPerSite(x,q_x,1.0e-14);
     DiffSpinorPerSite(y,q_y,1.0e-14);
}

/*** SP Tests ***/
TEST(TESTQPhiXBLAS, TestXmYNorm2VecF)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);

  LatticeFermion x; gaussian(x);
  LatticeFermion y; gaussian(y);

  LatticeInfo info(latdims);
  QPhiXSpinorF q_x(info);
  QPhiXSpinorF q_y(info);

  QDPSpinorToQPhiXSpinor(x,q_x,0);
  QDPSpinorToQPhiXSpinor(y,q_y,0);

  x -= y;
  Double n = norm2(x);

  double q_norm2 = XmyNorm2Vec(q_x,q_y)[0];
  DiffSpinor(x,q_x,5.0e-5);
  DiffSpinor(y,q_y,5.0e-5);
  double absdiff = std::abs(q_norm2 - toDouble(n));
  ASSERT_LT( absdiff, 6.0e-5);


}



TEST(TESTQPhiXBLAS, TestNorm2VecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);

     LatticeInfo info(latdims);
     QPhiXSpinorF qphix_x(info);

     QDPSpinorToQPhiXSpinor(x,qphix_x,0);

     Double n = norm2(x);
     double qphix_norm = Norm2Vec(qphix_x)[0];
     double absdiff = std::abs(qphix_norm - toDouble(n));
     ASSERT_LT( absdiff, 5.0e-5);


}


TEST(TESTQPhiXBLAS, TestInnerProductVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);

     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     DComplex iprod = QDP::innerProduct(x,y);

     std::complex<double> q_iprod = InnerProductVec(q_x,q_y)[0];

     double realnorm = std::abs ( std::real(q_iprod) - toDouble(real(iprod)) );
     double imnorm = std::abs ( std::imag(q_iprod) - toDouble(imag(iprod)) );

     ASSERT_LT( realnorm, 5.0e-5);
     ASSERT_LT( imnorm, 5.0e-5);

}


TEST(TESTQPhiXBLAS, TestZeroVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     LatticeFermion z = zero;

     LatticeInfo info(latdims);
     QPhiXSpinorF q_z(latdims);
     ZeroVec(q_z);
     DiffSpinor(z,q_z,1.0e-14);

}


TEST(TESTQPhiXBLAS, TestCopyVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion in; gaussian(in);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_in(info);
     QPhiXSpinorF q_copy(info);
     QDPSpinorToQPhiXSpinor(in, q_in, 0);

     CopyVec(q_copy,q_in);
     DiffSpinor(in,q_copy, 5.0e-5);

}


TEST(TESTQPhiXBLAS, TestZAxpyVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Complex a=Real(1.5);
     a += timesI(Real(-0.261));

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<std::complex<double>> alpha(1, std::complex<double>(toDouble(real(a)), toDouble(imag(a))));

     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,5.0e-5);
     DiffSpinor(y,q_y,5.0e-5);



}


TEST(TESTQPhiXBLAS, TestSAxpyVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Real a=Real(1.5);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<double> alpha(1, toDouble(a));


     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,5.0e-5);
     DiffSpinor(y,q_y,5.0e-5);

}
TEST(TESTQPhiXBLAS, TestCAxpyVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);
     Complex a=Real(1.5);
     a += timesI(Real(-0.261));

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);
     std::vector<std::complex<double>> alpha(1, std::complex<double>(toDouble(real(a)), toDouble(imag(a))));

     // AXPY
     y += a*x;

     // QPHix
     AxpyVec(alpha, q_x, q_y);

     DiffSpinor(x,q_x,5.0e-5);
     DiffSpinor(y,q_y,5.0e-5);

}

TEST(TESTQPhiXBLAS, TestAxVec)
{
  // Init the lattice
      IndexArray latdims={{8,8,8,8}};
      initQDPXXLattice(latdims);
      LatticeFermion x; gaussian(x);
      LatticeInfo info(latdims);
      QPhiXSpinor q_x(info);
      QDPSpinorToQPhiXSpinor(x,q_x,0);

      std::vector<double> alpha(1, 3.2);
      AxVec( alpha, q_x);
      x *= Real(alpha[0]);
      DiffSpinorPerSite(x,q_x,5.0e-14);


}

TEST(TESTQPhiXBLAS, TestAxVecF)
{
  // Init the lattice
      IndexArray latdims={{8,8,8,8}};
      initQDPXXLattice(latdims);
      LatticeFermion x; gaussian(x);
      LatticeInfo info(latdims);
      QPhiXSpinorF q_x(info);
      QDPSpinorToQPhiXSpinor(x,q_x,0);

      std::vector<double> alpha(1, 3.2);
      AxVec( alpha, q_x);
      x *= Real(alpha[0]);
      DiffSpinorPerSite(x,q_x,1.0e-6);


}
TEST(TESTQPhiXBLAS, TestGaussianVecF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     QDP::Seed seed;
     QDP::RNG::savern(seed);
     LatticeFermion x; gaussian(x);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QDP::RNG::setrn(seed);

     Gaussian(q_x);

     DiffSpinor(x,q_x,5.0e-5);
}

TEST(TESTQPhiXBLAS, TestYPeqXF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     YpeqXVec(q_x,q_y);
     y += x;
     DiffSpinorPerSite(x,q_x,1.0e-6);
     DiffSpinorPerSite(y,q_y,1.0e-6);
}

TEST(TESTQPhiXBLAS, TestYMeqXF)
{
  // Init the lattice
     IndexArray latdims={{8,8,8,8}};
     initQDPXXLattice(latdims);

     LatticeFermion x; gaussian(x);
     LatticeFermion y; gaussian(y);

     LatticeInfo info(latdims);
     QPhiXSpinorF q_x(info);
     QPhiXSpinorF q_y(info);
     QDPSpinorToQPhiXSpinor(x,q_x,0);
     QDPSpinorToQPhiXSpinor(y,q_y,0);

     YmeqXVec(q_x,q_y);
     y -= x;
     DiffSpinorPerSite(x,q_x,1.0e-6);
     DiffSpinorPerSite(y,q_y,1.0e-6);
}


TEST(TESTQPhiXBLAS, TestConvertRoutines)
{
  // Init the lattice
  IndexArray latdims={{8,8,8,8}};
  initQDPXXLattice(latdims);
  LatticeInfo info(latdims);

  LatticeFermion x; gaussian(x);
  QPhiXSpinor spinor_d(info);
  QDPSpinorToQPhiXSpinor(x,spinor_d,0);

  QPhiXSpinorF spinor_f(info);
  ConvertSpinor(spinor_d,spinor_f);
  DiffSpinor(x,spinor_f,5.0e-5);

  QPhiXSpinor spinor_back(info);
  ConvertSpinor(spinor_f,spinor_back);
  DiffSpinor(x,spinor_back,5.0e-5);
}

int main(int argc, char *argv[])
{
  return ::MGTesting::TestMain(&argc, argv);
}
