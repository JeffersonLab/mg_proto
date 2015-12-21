#ifndef INCLUDE_LATTICE_VIRTUAL_NODE_H
#define INCLUDE_LATTICE_VIRTUAL_NODE_H

namespace MGGeometry {

template<typename T, const unsigned int Nd>
struct VNGeneral {
	typedef T value_type;
	static const unsigned int n_dim;
	static const unsigned int n_sites;
	static const unsigned int mask;

};

template<typename T, const unsigned Nd>
const unsigned int VNGeneral<T,Nd>::n_dim = Nd;

template<typename T, const unsigned Nd>
const unsigned int VNGeneral<T,Nd>::n_sites = (1<<Nd);

template<typename T, const unsigned Nd>
const unsigned int VNGeneral<T,Nd>::mask = (1<<Nd)-1;

/** Here are the specializations of the virtual nodes */

template<typename T>
using VN_SSE = VNGeneral<T, sizeof(T)==sizeof(float) ? 2 : 1 >;

template<typename T>
using VN_AVX = VNGeneral<T, sizeof(T)==sizeof(float) ? 3 : 2 >;

template<typename T>
using VN_MIC = VNGeneral<T, sizeof(T)==sizeof(float) ? 4 : 3 >;

template<typename T>
using VN_AVX512 = VNGeneral<T, sizeof(T) == sizeof(float) ? 4 : 3>;

/* Special Scalar Vnode = no vectorization */
template<typename T>
using VN_Scalar = VNGeneral<T, 0>;

}
#endif
