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
#include "lattice/coarse/coarse_types.h"
#include "utils/print_utils.h"
#include <qmp.h>
#include <mpi.h>

using namespace MG;

namespace MG {


template<typename T>
class HaloContainer {
public:
	HaloContainer(const LatticeInfo& info) : _latt_info(info),
	_node_info(info.GetNodeInfo()), _datatype_size(haloDatumSize<T>(info)),
	_n_cols(0)
	{
		create(1);
	}

	~HaloContainer() { destroy(); }

private:
	void create(IndexType n_cols) {
		_n_cols = n_cols;
		MasterLog(INFO, "Creating HaloCB");
		const IndexArray& latt_size = _latt_info.GetLatticeDimensions();

		_n_face_dir[X_DIR] = (latt_size[Y_DIR]*latt_size[Z_DIR]*latt_size[T_DIR])/2;
		_n_face_dir[Y_DIR] = (latt_size[X_DIR]*latt_size[Z_DIR]*latt_size[T_DIR])/2;
		_n_face_dir[Z_DIR] = (latt_size[X_DIR]*latt_size[Y_DIR]*latt_size[T_DIR])/2;
		_n_face_dir[T_DIR] = (latt_size[X_DIR]*latt_size[Y_DIR]*latt_size[Z_DIR])/2;

		// We have QMP
		// Decide which directions are local by appealing to
		// QMP geometry
		const int* machine_size = QMP_get_logical_dimensions();
		if( QMP_get_logical_number_of_dimensions() != 4 ) {
			QMP_error("Number of QMP logical dimensions must be 4");
			QMP_abort(1);
		}

		// Find local vs non-local dirs
		for(int mu = 0; mu < 4; mu ++) {
			if( machine_size[mu] > 1 ){
				_local_dir[mu] = false;
			}
			else {
				_local_dir[mu] = true;
			}


		}


		// Count faces in the non-local-dims
		for(int mu=0; mu < n_dim; ++mu) {
			if( ! _local_dir[mu] ) {
				_face_in_bytes[mu] = _n_face_dir[mu]*_datatype_size*n_cols*sizeof(float);
			}
			else {
				_face_in_bytes[mu] = 0; // Local
			}
		}


		// Init pointers to null

		for(int mu=0; mu < 2*n_dim; ++mu ) {		// Buffers
			_send_to_dir[mu] = nullptr;
			_recv_from_dir[mu] = nullptr;

			// Msg Mem handles
			_msgmem_send_to_dir[mu] = nullptr;
			_msgmem_recv_from_dir[mu]= nullptr;

			// Send and receive handles in the directions
			_mh_send_to_dir[mu] = nullptr;
			_mh_recv_from_dir[mu] = nullptr;

				// Temporaries for the combined handles
			_mh_recv_all_dir[mu] = nullptr;
			_mh_send_all_dir[mu] = nullptr;
		}

			_mh_send_all = nullptr;
			_mh_recv_all = nullptr;


		// Declare memory for sends and receives in all the individual directions.
		for(int mu=0; mu < n_dim; ++mu) {

			if( ! _local_dir[mu] ) {
				// Code will crash if allocations fail: MemoryAllocate will print error and exit
				// So I am not checking returned pointers here.
				_send_to_dir[2*mu+MG_BACKWARD] = (float *)MemoryAllocate(_face_in_bytes[mu]);
				_send_to_dir[2*mu+MG_FORWARD] = (float *)MemoryAllocate(_face_in_bytes[mu]);
				_recv_from_dir[2*mu+MG_BACKWARD] = (float *)MemoryAllocate(_face_in_bytes[mu]);
				_recv_from_dir[2*mu+MG_FORWARD] = (float *)MemoryAllocate(_face_in_bytes[mu]);

				_msgmem_send_to_dir[2*mu+MG_BACKWARD] = QMP_declare_msgmem(_send_to_dir[2*mu+MG_BACKWARD], _face_in_bytes[mu]);
				_msgmem_send_to_dir[2*mu+MG_FORWARD] = QMP_declare_msgmem(_send_to_dir[2*mu+MG_FORWARD], _face_in_bytes[mu]);
				_msgmem_recv_from_dir[2*mu + MG_BACKWARD] = QMP_declare_msgmem(_recv_from_dir[2*mu+MG_BACKWARD], _face_in_bytes[mu]);
				_msgmem_recv_from_dir[2*mu+ MG_FORWARD] = QMP_declare_msgmem(_recv_from_dir[2*mu+MG_FORWARD], _face_in_bytes[mu]);

				_mh_recv_from_dir[2*mu+MG_BACKWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_BACKWARD],mu,-1,0);

				_mh_recv_from_dir[2*mu+MG_FORWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_FORWARD],mu,1, 0);

				_mh_send_to_dir[2*mu+MG_BACKWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_BACKWARD],mu,-1, 0);


				_mh_send_to_dir[2*mu+MG_FORWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_FORWARD],mu,1,0);

			}

		} // Loop over directions

		// Now declare new handles which will get collapsed.
		// these have to be 'compact'
		_num_nonlocal_dir = 0;
		for(int mu=0; mu < n_dim; ++mu) {

			if( ! _local_dir[mu] ) {
				_nonlocal_dir[ _num_nonlocal_dir ] = mu;

				_mh_recv_all_dir[2*_num_nonlocal_dir+MG_BACKWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_BACKWARD],
						mu, -1,0 );

				_mh_send_all_dir[2*_num_nonlocal_dir+MG_FORWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_FORWARD],
									mu,1,0);

				_mh_recv_all_dir[2*_num_nonlocal_dir+MG_FORWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_FORWARD],
						mu,1, 0);

				_mh_send_all_dir[2*_num_nonlocal_dir+MG_BACKWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_BACKWARD],
						mu,-1, 0);



			_num_nonlocal_dir++;
			}


		}



		if ( _num_nonlocal_dir > 0 ) {
			 _mh_send_all = QMP_declare_multiple(_mh_send_all_dir,2*_num_nonlocal_dir);
			 _mh_recv_all = QMP_declare_multiple(_mh_recv_all_dir,2*_num_nonlocal_dir);

		}

		const IndexArray& node_coords = _node_info.NodeCoords();
		const IndexArray& node_dims = _node_info.NodeDims();

		_am_i_pt_min = (node_coords[T_DIR]==0);
		_am_i_pt_max = (node_coords[T_DIR]==(node_dims[T_DIR]-1));


	}// Function

	void destroy() {
		_n_cols = 0;

		// Free the combined
		if( _mh_send_all ) { QMP_free_msghandle( _mh_send_all ); _mh_send_all=nullptr; }
		if( _mh_recv_all ) { QMP_free_msghandle( _mh_recv_all ); _mh_recv_all=nullptr; }
		// Free the underlying individuals

	#if 0
		// Freeing the multiple should free these?
		int last = 2*_num_nonlocal_dir-1;

		while( last >= 0 ) {

			if ( _mh_send_all_dir[last] ) QMP_free_msghandle( _mh_send_all_dir[last]); _mh_send_all_dir[last]=nullptr;
			if ( _mh_recv_all_dir[last] ) QMP_free_msghandle( _mh_recv_all_dir[last]); _mh_recv_all_dir[last]=nullptr;

			last--;
		}
	#endif

		// free theindividual directions.
		for(int mu = 0; mu < 8; mu++) {
			int dir = mu / 2;
			if ( ! _local_dir[dir] ) {



				if ( _mh_send_to_dir[mu] ) QMP_free_msghandle(_mh_send_to_dir[mu]);
				_mh_send_to_dir[mu] = nullptr;

				if ( _mh_recv_from_dir[mu]) QMP_free_msghandle(_mh_recv_from_dir[mu]);
				_mh_recv_from_dir[mu] = nullptr;

				if( _msgmem_send_to_dir[mu] ) QMP_free_msgmem( _msgmem_send_to_dir[mu]);
				_msgmem_send_to_dir[ mu ] = nullptr;

				if( _msgmem_recv_from_dir[mu] ) QMP_free_msgmem( _msgmem_recv_from_dir[mu]);
				_msgmem_recv_from_dir[mu] = nullptr;

				if( _send_to_dir[mu]) MemoryFree(_send_to_dir[mu]);
				_send_to_dir[mu] = nullptr;

				if( _recv_from_dir[mu]) MemoryFree(_recv_from_dir[mu]);
				_recv_from_dir[mu] = nullptr;
			}
		}
	}

public:

	void setNCols(IndexType n_cols) {
		if (_n_cols != n_cols) {
			destroy();
			create(n_cols);
		}
	}

	bool
	LocalDir(int d) const {
		return _local_dir[d];
	}


	bool
	AmIPtMin()  const {
		return _am_i_pt_min;
	}

	bool
	AmIPtMax() const {
		return _am_i_pt_max;
	}

	int
	NumNonLocalDirs() const {
		return _num_nonlocal_dir;
	}


	void StartSendToDir(int mu)
	{
		if( QMP_start(_mh_send_to_dir[mu]) != QMP_SUCCESS ) {
			QMP_error("Failed to start send\n");
			QMP_abort(1);
		}
	}

	void FinishSendToDir(int mu)
	{
		if( QMP_wait(_mh_send_to_dir[mu]) != QMP_SUCCESS ) {
			QMP_error("Failed to finish send\n");
			QMP_abort(1);
		}
	}

	void StartRecvFromDir(int mu)
	{
		if( QMP_start(_mh_recv_from_dir[mu]) != QMP_SUCCESS ) {
			QMP_error("Failed to start recv\n");
			QMP_abort(1);
		}
	}

	void FinishRecvFromDir(int mu)
	{
		if( QMP_wait(_mh_recv_from_dir[mu]) != QMP_SUCCESS ) {
			QMP_error("Failed to finish recv dir\n");
			QMP_abort(1);
		}
	}



	void StartAllSends()
	{
		if( QMP_start(_mh_send_all) != QMP_SUCCESS ) {
				QMP_error("Failed to start send\n");
				QMP_abort(1);
			}
	}

	void FinishAllSends()
	{
		if( QMP_wait(_mh_send_all) != QMP_SUCCESS ) {
				QMP_error("Failed to start send\n");
				QMP_abort(1);
			}
	}

	void StartAllRecvs()
	{
		if( QMP_start(_mh_recv_all) != QMP_SUCCESS ) {
				QMP_error("Failed to start send\n");
				QMP_abort(1);
			}
	}

	void FinishAllRecvs()
	{
		if( QMP_wait(_mh_recv_all) != QMP_SUCCESS ) {
				QMP_error("Failed to start send\n");
				QMP_abort(1);
			}
	}




	void ProgressComms()
	{
		int flag = 0;
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
	}


	float* GetSendToDirBuf(int mu) { return _send_to_dir[mu]; }
	float* GetRecvFromDirBuf(int mu) { return _recv_from_dir[mu]; }

	const float* GetSendToDirBuf(int mu) const { return _send_to_dir[mu]; }
	const float* GetRecvFromDirBuf(int mu) const { return _recv_from_dir[mu]; }

	int    NumSitesInFace(int mu) const { return _n_face_dir[mu]; }

	const LatticeInfo& GetInfo() const {
		return _latt_info;
	}

	inline
	size_t GetDataTypeSize() const
	{
		return _datatype_size*_n_cols;

	}

private:

	const LatticeInfo _latt_info;
	const NodeInfo& _node_info;
	const size_t _datatype_size;
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

	IndexType _n_cols;


}; // Halo class




} // MG Testing Namespace




#endif /* INCLUDE_LATTICE_SPINOR_HALO_QMP_H_ */
