/*
 * spinor_halo.h
 *
 *  Created on: Mar 7, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_SPINOR_HALO_SINGLE_H_
#define INCLUDE_LATTICE_SPINOR_HALO_SINGLE_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"

using namespace MG;

namespace MGTesting {


class SpinorHaloCB {
public:
	SpinorHaloCB(const LatticeInfo& info){}
	~SpinorHaloCB(){}

	bool LocalDir(int mu ) { return true; }
	bool AmIPtMin() { return true; }
	bool AmIPtMax() { return true; }

	int NumNonLocalDirs() { return 0; }

	void StartSendToDir(int mu) { }
	void FinishSendToDir(int mu) { }

	void StartRecvFromDir(int mu) { }
	void FinishRecvFromDir(int mu) { }


	void StartAllSends() {}
	void FinishAllSends(){}
	void StartAllRecvs() {}
	void FinishAllRecvs() {}

	void ProgressComms(){}


	float* GetSendToDirBuf(int mu) { return nullptr; }
	float* GetRecvFromDirBuf(int mu) { return nullptr; }

}; // SpinorHalo class

} // MG Testing Namespace




#endif /* INCLUDE_LATTICE_SPINOR_HALO_QMP_H_ */
