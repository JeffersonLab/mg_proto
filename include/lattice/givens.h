/*
 * givens.h
 *
 *  Created on: Oct 13, 2017
 *      Author: bjoo
 */
#pragma once
#ifndef INCLUDE_LATTICE_GIVENS_H_
#define INCLUDE_LATTICE_GIVENS_H_
#include "lattice/array2d.h"
#include <complex>
namespace MG {
namespace FGMRES {

  class Givens {
  public:


    Givens(int col, const Array2d<std::complex<double>>& H);
    /*! Apply the rotation to column col of the matrix H. The
     *  routine affects col and col+1.
     *
     *  \param col  the columm
     *  \param  H   the matrix
     */

    void operator()(int col,  Array2d<std::complex<double>>& H);

    /*! Apply rotation to Column Vector v */
    void operator()(std::vector<std::complex<double>>& v);

  private:
    int col_;
    std::complex<double> s_;
    std::complex<double> c_;
    std::complex<double> r_;
  };



  } // Namespace FGMRES
} // Namespace MG



#endif /* INCLUDE_LATTICE_GIVENS_H_ */
