/*
 * array2d.h
 *
 *  Created on: Mar 1, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_ARRAY2D_H_
#define INCLUDE_LATTICE_ARRAY2D_H_

#include <vector>
namespace MG
{

template<typename T>
class Array2d {
public:
	Array2d(): _Ncol(0), _Nrow(0)
	{
		_data.resize(0);
	}

	~Array2d()
	{
		_Ncol = 0;
		_Nrow = 0;
		_data.resize(0);
	}

	Array2d(int Ncol, int Nrow) : _Ncol(Ncol), _Nrow(Nrow)
	{
		_data.resize(Ncol*Nrow);
	}


	T& operator()(int row, int col) {
		return _data[ row + _Nrow*col ];
	}

	const T& operator()(int row, int col) const {
		return _data[ row + _Nrow*col];
	}

	void resize(int Ncol, int Nrow) {
		_Ncol = Ncol;
		_Nrow = Nrow;
		_data.resize(Ncol*Nrow);
	}

private:
	int _Ncol;
	int _Nrow;
	std::vector<T> _data;
};


}




#endif /* INCLUDE_LATTICE_ARRAY2D_H_ */
