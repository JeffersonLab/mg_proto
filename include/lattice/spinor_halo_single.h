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

template <HaloType type>
class HaloCB {
public:
	HaloCB(const LatticeInfo& info): _info(info){

		switch( type ) {
				case COARSE_SPINOR:
					_datatype_size = sizeof(float)*_latt_info.GetNumColorSpins()*n_complex;
					break;
				case COARSE_GAUGE:
					_datatype_size = sizeof(float)*_latt_info.GetNumColorSpins()*_latt_info.GetNumColorSpins()*n_complex;
					break;
				default:
					MasterLog(ERROR, "Unknown Halo Type %d", (int)type);
					break;
				}
	}
	~HaloCB(){}

	bool
	LocalDir(int mu ) const { return true; }
	bool AmIPtMin() const { return true; }
	bool AmIPtMax() const { return true; }

	int NumNonLocalDirs() const { return 0; }

	void StartSendToDir(int mu) { }
	void FinishSendToDir(int mu) { }

	void StartRecvFromDir(int mu) { }
	void FinishRecvFromDir(int mu) { }


	void StartAllSends() {}
	void FinishAllSends(){}
	void StartAllRecvs() {}
	void FinishAllRecvs() {}

	void ProgressComms(){}


	float* GetSendToDirBuf(int mu)  { return nullptr; }
	float* GetRecvFromDirBuf(int mu)  { return nullptr; }
	const float* GetSendToDirBuf(int mu) const { return nullptr; }
	const float* GetRecvFromDirBuf(int mu) const { return nullptr; }

	int NumSitesInFace(int mu) const { return 0; }
	// FIXME: Ist his wrong?
	// We can still have sites in the face just because there is no comms

	const LatticeInfo& GetInfo() const {
		return _info;
	}

	size_t GetDataTypeSize() {
		return _datatype_size;
	}

	HaloType GetHaloType() {
		return type;
	}
private:
	const LatticeInfo& _info;
	size_t _datatype_size;

}; // Halo class

using SpinorHaloCB = HaloCB<COARSE_SPINOR>;
using CoarseGaugeHaloCB = HaloCB<COARSE_GAUGE>;

} // MG Testing Namespace




#endif /* INCLUDE_LATTICE_SPINOR_HALO_QMP_H_ */
