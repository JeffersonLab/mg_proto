/*
 * block_blas.h
 *
 *  Created on: Nov 2, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_BLOCK_BLAS_H_
#define INCLUDE_LATTICE_BLOCK_BLAS_H_

#include "lattice/constants.h"
#include "utils/print_utils.h"
#include <complex>


using namespace MGUtils;

namespace MGGeometry {

	// This is a base case that sums over contiguous data
	template<typename T>
	void InnerProduct(std::complex<double>& result, const T* data_left, const T* data_right, IndexType len_in_complex)
	{
		result = std::complex<double>(0,0);
		const std::complex<T> *cdata_left  = reinterpret_cast<const std::complex<T>*>(data_left);
		const std::complex<T> *cdata_right = reinterpret_cast<const std::complex<T>*>(data_right);

#pragma omp for  reduction(+:result)
		for(IndexType elem=0; elem < len_in_complex; ++elem) {
			result += conj(cdata_left[elem])*cdata_right[elem];
		}

	}

	template<typename T>
	void Norm2(double &result, const T* data, IndexType len_in_complex)
	{
		result = 0;

#pragma omp for simd aligned(data: MG_DEFAULT_ALIGNMENT)  reduction(+:result)
		for(IndexType elem=0; elem < n_complex*len_in_complex; ++elem) {
			result += static_cast<double>(data[elem]*data[elem]);
		}


	}

}



#endif /* INCLUDE_LATTICE_BLOCK_BLAS_H_ */
