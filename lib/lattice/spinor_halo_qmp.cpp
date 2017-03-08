/*
 * spinor_halo.cpp
 *
 *  Created on: Mar 7, 2017
 *      Author: bjoo
 */


#include "lattice/spinor_halo_qmp.h"
#include "utils/print_utils.h"
#include <mpi.h>

namespace MGTesting
{

SpinorHaloCB::SpinorHaloCB(const LatticeInfo& info) : _latt_info(info),
_node_info(info.GetNodeInfo())
{
	MasterLog(INFO, "Creating SpinorHaloCB");
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
			_face_in_bytes[mu] = _n_face_dir[mu]*sizeof(float)*_latt_info.GetNumColorSpins()*n_complex;
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

			_mh_recv_from_dir[2*mu+MG_BACKWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_BACKWARD],mu,0,0);

			_mh_recv_from_dir[2*mu+MG_FORWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_FORWARD],mu,1, 0);

			_mh_send_to_dir[2*mu+MG_BACKWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_BACKWARD],mu,0, 0);


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
					mu, 0,0 );

			_mh_send_all_dir[2*_num_nonlocal_dir+MG_FORWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_FORWARD],
								mu,1,0);

			_mh_recv_all_dir[2*_num_nonlocal_dir+MG_FORWARD] = QMP_declare_receive_relative(_msgmem_recv_from_dir[2*mu+MG_FORWARD],
					mu,1, 0);

			_mh_send_all_dir[2*_num_nonlocal_dir+MG_BACKWARD] = QMP_declare_send_relative(_msgmem_send_to_dir[2*mu+MG_BACKWARD],
					mu,0, 0);



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

SpinorHaloCB::~SpinorHaloCB()
{

	// Free the combined
	if( _mh_send_all ) QMP_free_msghandle( _mh_send_all ); _mh_send_all=nullptr;
	if( _mh_recv_all ) QMP_free_msghandle( _mh_recv_all ); _mh_recv_all=nullptr;
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


void
SpinorHaloCB::StartSendToDir(int mu)
{
	if( QMP_start(_mh_send_to_dir[mu]) != QMP_SUCCESS ) {
		QMP_error("Failed to start send\n");
		QMP_abort(1);
	}
}

void
SpinorHaloCB::FinishSendToDir(int mu)
{
	if( QMP_wait(_mh_send_to_dir[mu]) != QMP_SUCCESS ) {
		QMP_error("Failed to finish send\n");
		QMP_abort(1);
	}
}

void
SpinorHaloCB::StartRecvFromDir(int mu)
{
	if( QMP_start(_mh_recv_from_dir[mu]) != QMP_SUCCESS ) {
		QMP_error("Failed to start recv\n");
		QMP_abort(1);
	}
}

void
SpinorHaloCB::FinishRecvFromDir(int mu)
{
	if( QMP_wait(_mh_recv_from_dir[mu]) != QMP_SUCCESS ) {
		QMP_error("Failed to finish recv dir\n");
		QMP_abort(1);
	}
}

void
SpinorHaloCB::StartAllSends()
{
	if( QMP_start(_mh_send_all) != QMP_SUCCESS ) {
			QMP_error("Failed to start send\n");
			QMP_abort(1);
		}
}

void
SpinorHaloCB::FinishAllSends()
{
	if( QMP_wait(_mh_send_all) != QMP_SUCCESS ) {
			QMP_error("Failed to start send\n");
			QMP_abort(1);
		}
}

void
SpinorHaloCB::StartAllRecvs()
{
	if( QMP_start(_mh_recv_all) != QMP_SUCCESS ) {
			QMP_error("Failed to start send\n");
			QMP_abort(1);
		}
}

void
SpinorHaloCB::FinishAllRecvs()
{
	if( QMP_wait(_mh_recv_all) != QMP_SUCCESS ) {
			QMP_error("Failed to start send\n");
			QMP_abort(1);
		}
}

void
SpinorHaloCB::ProgressComms()
{
	int flag = 0;
	MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
}


}// Namespace;
