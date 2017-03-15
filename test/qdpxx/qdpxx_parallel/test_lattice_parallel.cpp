#include "gtest/gtest.h"
#include "../../test_env.h"
#include "../qdpxx_utils.h"

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"
#include "lattice/geometry_utils.h"
#include <vector>
#include <random>

#include "lattice/spinor_halo.h"
#include "lattice/coarse/coarse_op.h"
using namespace MG; 
using namespace MGTesting;


TEST(TestLatticeParallel, TestSpinorHaloCreate)
{
	 IndexArray latdims={{4,4,4,4}};
	  NodeInfo node;
	  LatticeInfo info(latdims, 4, 3, node);
	  initQDPXXLattice(latdims);
	  SpinorHaloCB halo(info); // Create then Destroy.
}

TEST(TestSpinorHalo, TestDirectionShift)
{
	IndexArray latdims={{4,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,4,node);
	initQDPXXLattice(latdims);
	SpinorHaloCB halo(info);

	// My Node ID
	int my_node = node.NodeID();

	for(int dir=0; dir < n_dim; ++dir) {
		if( ! halo.LocalDir(dir) ) {


			int num_sites = halo.NumSitesInFace(dir);
			int num_elem = num_sites*info.GetNumColorSpins();

			// Forward send: 8xmy_node + MG_FORWARD
			float *buffer = halo.GetSendToDirBuf(2*dir + MG_FORWARD);
			float value = (float)(8*my_node + 2*dir + MG_FORWARD);

			MasterLog(INFO, "Filling Forward T-DIR buffer with %lf", value);
#pragma omp parallel for
			for(int idx=0; idx < num_elem; ++idx) {
				buffer[idx] = value;
			}

			// Backward send: 8xmy_node + 2*dir + MG_BACKWARD
			buffer = halo.GetSendToDirBuf(2*dir + MG_BACKWARD);
			value  = (float)(8*my_node + 2*dir + MG_BACKWARD);
			MasterLog(INFO, "Filling Backward T-DIR buffer with %lf", value);
#pragma omp parallel for
			for(int idx=0; idx < num_elem; ++idx) {
				buffer[idx] = value;
			}


		} // ! halo.LocalDIr()
		else {
			// Assert buffer is null for local directions
			MasterLog(INFO,"Asserting buffers in dir %d are NULL", dir);
			ASSERT_EQ( halo.GetSendToDirBuf( 2*dir + MG_BACKWARD), nullptr) ;
			ASSERT_EQ( halo.GetSendToDirBuf( 2*dir + MG_FORWARD), nullptr) ;
		}
	} // dir


	for(int dir=0; dir < n_dim; ++dir ) {
		if( ! halo.LocalDir(dir) ) {

			// This test is executed in a grid of 1x1x1x2 so
			MasterLog(INFO,"Staring Backward Recvs in dir %d",dir);
			halo.StartRecvFromDir(2*dir+MG_BACKWARD);

			MasterLog(INFO, "Starting Forward T Recvs in dir %d",dir);
			halo.StartRecvFromDir(2*dir+MG_FORWARD);

			MasterLog(INFO, "Starting Forward T sends in dir %d",dir);
			halo.StartSendToDir(2*dir + MG_FORWARD);

			MasterLog(INFO, "Starting Backward T sends in dir %d",dir);
			halo.StartSendToDir(2*dir + MG_BACKWARD);

			MasterLog(INFO, "FInishing Forward T Sends in dir %d",dir);
			halo.FinishSendToDir(2*dir + MG_FORWARD);

			MasterLog(INFO, "Finishing BAckward T Sends in dir %d", dir);
			halo.FinishSendToDir(2*dir + MG_BACKWARD);

			MasterLog(INFO, "Finishing receives from T backward in dir %d", dir);
			halo.FinishRecvFromDir(2*dir + MG_BACKWARD);

			MasterLog(INFO, "Finishing receives from T Forward in dir %d", dir);
			halo.FinishRecvFromDir(2*dir+ MG_FORWARD);
		}
	}

	for(int dir = 0; dir < n_dim; ++dir) {
		if ( ! halo.LocalDir(dir) ) {

			// Comms should be finished.
			// Check that we got the right data.
			int forward_node = node.NeighborNode(dir,MG_FORWARD);
			int back_node = node.NeighborNode(dir, MG_BACKWARD);

			// Forward neighbor will have sent backwards.
			float forw_value = (float)(8*forward_node + 2*dir + MG_BACKWARD);

			// Backward neighbor will have sent his data forward
			float back_value = (float)(8*back_node + 2*dir + MG_FORWARD);

			float *buffer = halo.GetRecvFromDirBuf(2*dir + MG_FORWARD);
			int num_elem = info.GetNumColorSpins()*halo.NumSitesInFace(dir);

			MasterLog(INFO, "Checking Forward buffer contains %lf (expected)", forw_value);
			for(int i=0; i < num_elem; ++i) {
				ASSERT_FLOAT_EQ(buffer[i], forw_value);
			}

			MasterLog(INFO, "Checking Backward buffer contains %lf (expected)", back_value);
			buffer = halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD);
			for(int i=0; i < num_elem; ++i) {
				ASSERT_FLOAT_EQ(buffer[i], back_value);

			}


		}
		else {
			// Buffers are null for nonlocal directions
			ASSERT_EQ( halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD), nullptr );
			ASSERT_EQ( halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD), nullptr) ;

		}
	}

}

TEST(TestSpinorHalo, TestCommAll)
{
	IndexArray latdims={{4,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,4,node);

	initQDPXXLattice(latdims);
	SpinorHaloCB halo(info);

	// My Node ID
	int my_node = node.NodeID();

	for(int dir=0; dir < n_dim; ++dir) {
		if( ! halo.LocalDir(dir) ) {


			int num_sites = halo.NumSitesInFace(dir);
			int num_elem = num_sites*info.GetNumColorSpins();

			// Forward send: 8xmy_node + MG_FORWARD
			float *buffer = halo.GetSendToDirBuf(2*dir + MG_FORWARD);
			float value = (float)(8*my_node + 2*dir + MG_FORWARD);

			MasterLog(INFO, "Filling Forward  buffer in dir  %d with %lf", dir,value);
#pragma omp parallel for
			for(int idx=0; idx < num_elem; ++idx) {
				buffer[idx] = value;
			}

			// Backward send: 8xmy_node + 2*dir + MG_BACKWARD
			buffer = halo.GetSendToDirBuf(2*dir + MG_BACKWARD);
			value  = (float)(8*my_node + 2*dir + MG_BACKWARD);
			MasterLog(INFO, "Filling Backward buffer in dir %d with %lf",dir, value);
#pragma omp parallel for
			for(int idx=0; idx < num_elem; ++idx) {
				buffer[idx] = value;
			}


		} // ! halo.LocalDIr()
		else {
			// Assert buffer is null for local directions
			MasterLog(INFO,"Asserting buffers in dir %d are NULL", dir);
			ASSERT_EQ( halo.GetSendToDirBuf( 2*dir + MG_BACKWARD), nullptr) ;
			ASSERT_EQ( halo.GetSendToDirBuf( 2*dir + MG_FORWARD), nullptr) ;
		}
	} // dir




			halo.StartAllRecvs();
			halo.StartAllSends();
			halo.FinishAllSends();
			halo.FinishAllRecvs();

	for(int dir = 0; dir < n_dim; ++dir) {
		if ( ! halo.LocalDir(dir) ) {

			// Comms should be finished.
			// Check that we got the right data.
			int forward_node = node.NeighborNode(dir,MG_FORWARD);
			int back_node = node.NeighborNode(dir, MG_BACKWARD);

			// Forward neighbor will have sent backwards.
			float forw_value = (float)(8*forward_node + 2*dir + MG_BACKWARD);

			// Backward neighbor will have sent his data forward
			float back_value = (float)(8*back_node + 2*dir + MG_FORWARD);

			float *buffer = halo.GetRecvFromDirBuf(2*dir + MG_FORWARD);
			int num_elem = info.GetNumColorSpins()*halo.NumSitesInFace(dir);

			MasterLog(INFO, "Checking Forward buffer in dir %d contains %lf (expected)",dir, forw_value);
			for(int i=0; i < num_elem; ++i) {
				ASSERT_FLOAT_EQ(buffer[i], forw_value);
			}

			MasterLog(INFO, "Checking Backward buffer in dir %d contains %lf (expected)",dir, back_value);
			buffer = halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD);
			for(int i=0; i < num_elem; ++i) {
				ASSERT_FLOAT_EQ(buffer[i], back_value);

			}


		}
		else {
			// Buffers are null for nonlocal directions
			ASSERT_EQ( halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD), nullptr );
			ASSERT_EQ( halo.GetRecvFromDirBuf(2*dir + MG_BACKWARD), nullptr) ;

		}
	}

}

TEST(TestLatticeParallel, CoarseDiracHaloCreate)
{
	// Check the Halo is initialized properly in a coarse Dirac Op
	IndexArray latdims={{4,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,4,node);

	initQDPXXLattice(latdims);


	// Coarse Dirac Operator
	CoarseDiracOp D_op_coarse(info);
	SpinorHaloCB& my_halo = D_op_coarse.GetSpinorHalo();

	const LatticeInfo& halo_info = my_halo.GetInfo();
	const NodeInfo& halo_node = halo_info.GetNodeInfo();

	ASSERT_EQ( node.NodeID(), halo_node.NodeID() );
	ASSERT_EQ( node.NumNodes(), halo_node.NumNodes() );
	for(int mu=0; mu < n_dim; ++mu) {
		ASSERT_EQ( node.NodeDims()[mu], halo_node.NodeDims()[mu] );
		ASSERT_EQ( halo_info.GetLatticeDimensions()[mu], info.GetLatticeDimensions()[mu]);
		ASSERT_EQ( halo_info.GetLatticeOrigin()[mu], info.GetLatticeOrigin()[mu]);
	}
	for(int mu=0; mu < n_dim; ++mu) {
		for(int fwb = MG_BACKWARD; fwb <= MG_FORWARD; ++fwb ) {
			ASSERT_EQ( halo_node.NeighborNode(mu,fwb), node.NeighborNode(mu,fwb));
		}
	}

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

