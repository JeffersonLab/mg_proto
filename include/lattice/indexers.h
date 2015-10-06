/*
 * indexers.h
 *
 *  Created on: Oct 2, 2015
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_INDEXERS_H_
#define INCLUDE_LATTICE_INDEXERS_H_

namespace MGGeometry {


class LatticeCartesianSiteIndexer {
public:

	// Note I am using C++11 style brace initializer for face_strides
	//
	LatticeCartesianSiteIndexer(const unsigned int Nx,
								const unsigned int Ny,
								const unsigned int Nz,
								const unsigned int Nt) : _Nx(Nx), _Ny(Ny),_Nz(Nz), _Nt(Nt),
								face_strides{ {Ny,Nz}, {Nx,Nz}, {Nx,Ny}, {Nx, Ny} } {



	}

	~LatticeCartesianSiteIndexer() {}

	inline
	unsigned int Index(unsigned int x, unsigned int y, unsigned int z, unsigned int t)
	{
		return x + _Nx*(y + _Ny*(z + _Nz*t)) ;
	}

	inline
	void Coords(const unsigned int index,
				unsigned int& x,
				unsigned int& y,
				unsigned int& z,
				unsigned int& t )
	{
		unsigned int t_yzt = index / _Nx;   // t1 is ( y + _Ny*(z + _Nz*t) )
		x = index - t_yzt * _Nx;

		unsigned int t_zt = t_yzt / _Ny;      // t2 is ( z + _Nz*t)
		y = t_yzt - t_zt * _Ny;

		t = t_zt / _Nz;
		z  =t_zt - t * _Nz;

	}


	inline
	void Coords3D(const unsigned int dim,
				  const unsigned int index,
				  unsigned int& coord1,
				  unsigned int& coord2,
				  unsigned int& coord3) {

		unsigned int ndim1 = face_strides[dim][0];
		unsigned int ndim2 = face_strides[dim][1];

		unsigned int t1=index / ndim1;
		coord1 = index - t1*ndim1;
		coord3 = t1 / ndim2;
		coord2 = t1 - coord3*ndim2;
		return;

	}
private:
	const unsigned int _Nx;
	const unsigned int _Ny;
	const unsigned int _Nz;
	const unsigned int _Nt;
	const unsigned int face_strides[n_dim][2];


};


}



#endif /* INCLUDE_LATTICE_INDEXERS_H_ */
