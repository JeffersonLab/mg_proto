/*
 * block_ortho_profile.cpp
 *
 *  Created on: Feb 13, 2018
 *      Author: bjoo
 */


/*
 * recursive_vcycle_profile.cpp
 *
 *  Created on: Nov 8, 2017
 *      Author: bjoo
 */
#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"

#include <lattice/qphix/invfgmres_qphix.h>

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/coarse_types.h"


#include "lattice/fine_qdpxx/mg_params_qdpxx.h"
#include "lattice/qphix/mg_level_qphix.h"
#include "lattice/qphix/vcycle_recursive_qphix.h"

#include <ittnotify.h>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(QPhiXTestRecursiveVCycle, TestLevelSetup2Level)
{
	IndexArray latdims={{16,16,16,16}};



	initQDPXXLattice(latdims);

	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	IndexArray node_orig=NodeInfo().NodeCoords();
	for(int mu=0; mu < n_dim; ++mu) node_orig[mu]*=latdims[mu];

	float m_q = 0.01;
	float c_sw = 1.25;

	int t_bc=-1; // Antiperiodic t BCs


	multi1d<LatticeColorMatrix> u(Nd);
	for(int mu=0; mu < Nd; ++mu) {
		//u[mu]=1;
		gaussian(u[mu]);
		reunit(u[mu]);
	}
	// Move to QPhiX space:
	LatticeInfo fine_info(node_orig,latdims,4,3,NodeInfo());

	MasterLog(INFO,"Creating M");

	// Create float prec linop for preconditioner
	std::shared_ptr<QPhiXWilsonCloverLinearOperatorF> M_f=
			std::make_shared<QPhiXWilsonCloverLinearOperatorF>(fine_info, m_q, c_sw, t_bc,u);

	// Create full prec linop for solver
	QPhiXWilsonCloverLinearOperator M(fine_info,m_q, c_sw, t_bc,u);

	SetupParams level_setup_params = {
			3,       // Number of levels
			{24,24},   // Null vecs on L0, L1
			{
					{4,4,4,4},  // Block Size from L0->L1
					{2,2,2,2}   // Block Size from L1->L2
			},
			{500,500},          // Max Nullspace Iters
			{5e-6,5e-6},        // Nullspace Target Resid
			{false,false}
	};


	QPhiXMultigridLevels mg_levels;
	MasterLog(INFO,"Calling Level setup");
	SetupQPhiXMGLevels(level_setup_params, mg_levels, M_f);
}


int main(int argc, char *argv[])
{
	return MGTesting::TestMain(&argc, argv);
}






