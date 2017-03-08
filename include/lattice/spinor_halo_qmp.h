/*
 * spinor_halo.h
 *
 *  Created on: Mar 7, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_SPINOR_HALO_QMP_H_
#define INCLUDE_LATTICE_SPINOR_HALO_QMP_H_

#include "utils/memory.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"

#include <qmp.h>

using namespace MG;

namespace MGTesting {


class SpinorHaloCB {
public:
	SpinorHaloCB(const LatticeInfo& info);
	~SpinorHaloCB();

	bool LocalDir(int mu );
	bool AmIPtMin();
	bool AmIPtMax();
	int NumNonLocalDirs();

	void StartSendToDir(int mu);
	void FinishSendToDir(int mu);

	void StartRecvFromDir(int mu);
	void FinishRecvFromDir(int mu);


	void StartAllSends();
	void FinishAllSends();
	void StartAllRecvs();
	void FinishAllRecvs();

	void ProgressComms();

	float* GetSendToDirBuf(int mu) { return _send_to_dir[mu]; }
	float* GetRecvFromDirBuf(int mu) { return _recv_from_dir[mu]; }


private:

	const LatticeInfo& _latt_info;
	const NodeInfo& _node_info;

	int _n_face_dir[4];
	bool _local_dir[4];
	size_t _face_in_bytes[4];

	float* _send_to_dir[8]; // Send buffers. SP for now
	float* _recv_from_dir[8]; // Receive buffers

    QMP_msgmem_t _msgmem_send_to_dir[8];
    QMP_msgmem_t _msgmem_recv_from_dir[8];
    QMP_msghandle_t _mh_send_to_dir[8];
    QMP_msghandle_t _mh_recv_from_dir[8];

    QMP_msghandle_t _mh_recv_all_dir[8];
    QMP_msghandle_t _mh_send_all_dir[8];

    QMP_msghandle_t _mh_send_all;
    QMP_msghandle_t _mh_recv_all;



    int _num_nonlocal_dir;
    int _nonlocal_dir[4];

    bool _am_i_pt_min;
    bool _am_i_pt_max;


}; // SpinorHalo class

} // MG Testing Namespace




#endif /* INCLUDE_LATTICE_SPINOR_HALO_QMP_H_ */
