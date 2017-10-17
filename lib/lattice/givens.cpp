/*
 * givens.cpp
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */

#include "utils/print_utils.h"
#include "lattice/givens.h"
#include <complex>
namespace MG {

namespace FGMRESGeneric {


   // Givens rotation.
   //   There are a variety of ways to choose the rotations
   //   which can do the job. I employ the method given by Saad
   //   in Iterative Methods (Sec. 6.5.9 (eq. 6.80)
   //
   //  [  conj(c) conj(s)  ] [ h_jj    ] = [ r ]
   //  [    -s       c     ] [ h_j+1,j ]   [ 0 ]
   //
   //  We know that h_j+1,j is a vector norm so imag(h_{j+1,j} = 0)
   //
   //  we have: s = h_{j+1,j} / t
   //           c = h_jj / t
   //
   //   t=sqrt( norm2(h_jj) + h_{j+1,j}^2 ) is real and nonnegative
   //
   //  so in this case s, is REAL and nonnegative (since t is and h_{j+1,j} is
   //  but  c is in general complex valued.
   //
   //
   //  using this we find r = conj(c) h_jj + conj(s) h_{j+1,j}
   //                        = (1/t)* [  conj(h_jj)*h_jj + h_{j+1,j}*h_{j+1,h} ]
   //                        = (1/t)* [  norm2(h_jj) + h_{j+1,j}^2 ]
   //                        = (1/t)* [ t^2 ] = t
   //
   //  Applying this to a general 2 vector
   //
   //   [ conj(c) conj(s) ] [ a ] = [ r_1 ]
   //   [   -s      c     ] [ b ]   [ r_2 ]
   //
   //   we have r_1 = conj(c)*a + conj(s)*b  = conj(c)*a + s*b  since s is real, nonnegative
   //      and  r_2 = -s*a + c*b
   //
   //  NB: In this setup we choose the sine 's' to be real in the rotation.
   //      This is in contradistinction from LAPACK which typically chooses the cosine 'c' to be real
   //
   //
   // There are some special cases:
   //   if  h_jj and h_{j+1,j} are both zero we can choose any s and c as long as |c|^2 + |s|^2 =1
   //   Keeping with the notation that s is real and nonnegative we choose
   //   if    h_jj != 0 and h_{j+1,h) == 0 => c = sgn(conj(h_jj)), s = 0, r = | h_jj |
   //   if    h_jj == 0 and h_{j+1,j} == 0 => c = 0, s = 1,  r = 0 = h_{j+1,j}
   //   if    h_jj == 0 and h_{j+1,j} != 0 => c = 0, s = 1,  r = h_{j+1,j} = h_{j+1,j}
   //   else the formulae we computed.

   /*! Given  a marix H, construct the rotator so that H(row,col) = r and H(row+1,col) = 0
    *
    *  \param col  the column Input
    *  \param  H   the Matrix Input
    */

   Givens::Givens(int col, const Array2d<std::complex<double>>& H) : col_(col)
 {
     std::complex<double> f = H(col_,col_);
     std::complex<double> g = H(col_,col_+1);

     if(  real(f) == 0 && imag(f) == 0  ) {

       // h_jj is 0
       c_ = std::complex<double>(0,0);
       s_ = std::complex<double>(1,0);
       r_ = g;  // Handles the case when g is also zero
     }
     else {
       if( real(g) == 0 && imag(g) == 0  ) {
         s_ = std::complex<double>(0,0);

         // NB: in std::complex norm is what QDP++ calls norm2
         c_ = conj(f)/sqrt( norm(f) ); //   sgn( conj(f) ) = conj(f) / | conj(f) |  = conj(f) / | f |
         r_ = std::complex<double>( abs(f), 0 );
       }
       else {
         // Revisit this with
         double t = sqrt( norm(f) + norm(g) );
         r_ = std::complex<double>(t,0);
         c_  = f/t;
         s_  = g/t;
       }
     }
 }

   /*! Apply the rotation to column col of the matrix H. The
    *  routine affects col and col+1.
    *
    *  \param col  the columm
    *  \param  H   the matrix
    */

   void
   Givens::operator()(int col,  Array2d<std::complex<double>>& H) {
     if ( col == col_ ) {
       // We've already done this column and know the answer
       H(col_,col_) = r_;
       H(col_,col_+1) = 0;
     }
     else {
       int row = col_; // The row on which the rotation was defined
       std::complex<double> a = H(col,row);
       std::complex<double> b = H(col,row+1);
       H(col,row) = conj(c_)*a + conj(s_)*b;
       H(col,row+1) = -s_*a + c_*b;
     }
   }

   /*! Apply rotation to Column Vector v */
   void
   Givens::operator()(std::vector<std::complex<double>>& v) {
     std::complex<double> a =  v[col_];
     std::complex<double> b =  v[col_+1];
     v[col_] = conj(c_)*a + conj(s_)*b;
     v[col_+1] =  -s_*a + c_*b;

   }

}
}

