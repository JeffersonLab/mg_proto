
#ifndef INCLUDE_LATTICE_COARSE_PRIMME_H_
#define INCLUDE_LATTICE_COARSE_PRIMME_H_

#include <complex>
#include <stdexcept>

#include "lattice/coarse/coarse_op.h"
#include "lattice/cmat_mult.h"
#include "utils/memory.h"
#include "utils/print_utils.h"
#include "MG_config.h"
#include "lattice/geometry_utils.h"
#include "lattice/eigs_common.h"

#ifdef MG_USE_PRIMME

#include "primme.h"

namespace MG {

	namespace {
		// Auxiliary function for global sum reduction
		void globalSumDouble(void *sendBuf, void *recvBuf, int *count, 
				primme_params *primme, int *ierr)
		{
			double *x = (double*)sendBuf, *y = (double*)recvBuf;
			if (x != y) {
				for (int i=0, count_=*count; i<count_; i++) y[i] = x[i];
			}
#ifdef MG_QMP_COMMS
			if (*count > 0) QMP_sum_double_array(y, *count);
#endif
			*ierr = 0;
		}

		// Auxiliary function for the matvec
		template<typename Spinor, typename LinOpT> struct PrimmeMatrixMatvec {
			static void fun(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy,
					int *blockSize, primme_params *primme, int *ierr)
			{
				// If this routine exits before reaching the end, notify it as an error by default
				*ierr = -1;

				// Quick return
				if (*blockSize <= 0) {*ierr = 0; return; }

				LinOpT *M = (LinOpT*)primme->matrix;

				std::shared_ptr<Spinor> xs = M->tmp(M->GetInfo(), *blockSize);
				std::shared_ptr<Spinor> ys = M->tmp(M->GetInfo(), *blockSize);

				PutColumns((const float*)x, *ldx*2, *xs, M->GetSubset());
				ZeroVec(*ys, M->GetSubset());
				(*M)(*ys, *xs);
				Gamma5Vec(*ys, M->GetSubset());
				GetColumns(*ys, M->GetSubset(), (float*)y, *ldy*2);

				// We're good!
				*ierr = 0;
			}
		};


	}

	/*
	 * Compute an approximate invariant subspace of a \gamma_5-Hermitian operator
	 *
	 * \param info: lattice info
	 * \param M: operator to compute the invariant subspace
	 * \param eigs_params: parameters to control the eigensolver
	 * \param V: returned eigenvectors, V[i]
	 * \param values: returned eigenvalues, values[i]
	 *
	 * The routine compute the largest eigenvalues/singular values of the
	 * \gamma_5-Hermitian operator M. The returned V and lambda satisfy:
	 *
	 *    ||\gamma_5 * M * V[i] - V[i] * values[i]||_2 <= eigs_params.RsdTarget * ||M||_2.
	 *
	 * The returned V is also the right singular vectors of M, \gamma_5*V[i]*sign(values[i])
	 * are the left singular vectors, and |values[i]| are the singular value.
	 */

  	template<typename Spinor, typename LinOpT>
	void computeDeflation(const LatticeInfo& info, const LinOpT& M, EigsParams eigs_params, std::shared_ptr<Spinor>& V, std::vector<float>& values)
	{
		const CBSubset& cbsubset = M.GetSubset();
		size_t nLocal = (size_t)info.GetNumColorSpins()*info.GetNumCBSites()*(cbsubset.end - cbsubset.start);
		IndexType nEv = eigs_params.MaxNumEvals;

		// Initialize PRIMME configuration
		primme_params primme;
		primme_initialize(&primme);

		// Set global sum reduction
		primme.numProcs = info.GetNodeInfo().NumNodes();
		primme.procID = info.GetNodeInfo().NodeID();
		primme.globalSumReal = globalSumDouble;
		primme.globalSumReal_type = primme_op_double;

		// Determine local and global matrix dimension
		double n = nLocal; QMP_sum_double_array(&n, 1);
		primme.n = n;             /* set global problem dimension */
		primme.nLocal = nLocal;   /* set local problem dimension */

		primme.numEvals = nEv;   /* Number of wanted eigenpairs */
		primme.eps = eigs_params.RsdTarget;      /* ||r|| <= eps * ||matrix|| */
		if (eigs_params.MaxRestartSize > 0)
			primme.maxBasisSize = eigs_params.MaxRestartSize;
		if (eigs_params.MaxIter > 0)
			primme.maxMatvecs = eigs_params.MaxIter;

		// Create evals. For twisted operators, gamma * operator is not Hermitian, but normal.
		values.resize(nEv);
		float *evals = values.data();

		// Create residual norms
		float *rnorms = new float[nEv];

		// Create eigenvectors
		// NOTE: BLAS/LAPACK obsession with the Fortran's way of passing arrays is contagious, and eigenvectors passed to
		// PRIMME, library which relays on BLAS/LAPACK heavily, follow the same convention.
		std::complex<float> *evecs = new std::complex<float>[nLocal * nEv];

		// Seek for the largest eigenvalue in magnitude for using the inverse operator
		primme.target = primme_largest_abs;
		double zero = 0;
		primme.targetShifts = &zero;
		primme.numTargetShifts = 1;

		// Set operator
		primme.matrixMatvec = PrimmeMatrixMatvec<Spinor,LinOpT>::fun;
		primme.matrix = (void*)&M;

		// primme.locking = 0;
		// primme.minRestartSize = primme.numEvals + 32;
		// primme.maxBasisSize = primme.numEvals + 64;

		// Set advanced options. If the operator is an inverter, configure PRIMME to minimize the
		// inverter applications. Otherwise use an strategy that minimizes orthogonalization time.
		primme_set_method(PRIMME_DEFAULT_MIN_MATVECS, &primme);

		primme.printLevel = eigs_params.VerboseP ? 3 : 1;

		// Call primme
		// Display PRIMME configuration struct (optional)
		if (primme.procID == 0 && primme.printLevel > 1)
			primme_display_params(primme);

		if (nEv > 0) {
			int ret = cprimme(evals, evecs, rnorms, &primme);

			if (1) {
				MasterLog(INFO, "Converged pairs       using PRIMME  = %d\n", primme.initSize);
				MasterLog(INFO, "Time to solve problem               = %e\n", primme.stats.elapsedTime);
				MasterLog(INFO, "Time spent in matVec                = %e  %.1f%%\n", primme.stats.timeMatvec,
						100 * primme.stats.timeMatvec / primme.stats.elapsedTime);
				MasterLog(INFO, "Time spent in orthogonalization     = %e  %.1f%% (%.1f GFLOPS)\n", primme.stats.timeOrtho,
						100 * primme.stats.timeOrtho / primme.stats.elapsedTime,
						primme.stats.numOrthoInnerProds * primme.n / primme.stats.timeOrtho / 1e9);
				MasterLog(INFO, "Time spent in dense operations      = %e  %.1f%% (%.1f GFLOPS)\n", primme.stats.timeDense,
						100 * primme.stats.timeDense / primme.stats.elapsedTime, primme.stats.flopsDense / primme.stats.timeDense / 1e9);
				double timeComm = primme.stats.timeGlobalSum + primme.stats.timeBroadcast;
				MasterLog(INFO, "Time spent in communications        = %e  %.1f%%\n", timeComm,
						100 * timeComm / primme.stats.elapsedTime);
			}
		}

		// Copy evecs to V
		V = std::make_shared<Spinor>(info, primme.initSize);
		ZeroVec(*V);
		PutColumns((const float*)evecs, nLocal*2, *V, cbsubset);

		// Resize evals
		values.resize(primme.initSize);

		// Local clean-up
		delete [] rnorms;
		delete evecs;
		primme_free(&primme);

		// Check solution
		if (1) {
			unsigned int nEv = V->GetNCol();
			std::vector<double> Vnorms2 = Norm2Vec(*V, cbsubset);
			for (unsigned int i=0; i<nEv; i++)
				assert(fabs(std::sqrt(Vnorms2[i]) - 1.0) <= 1e-5);
			std::shared_ptr<Spinor> MV = M.tmp(info, nEv);
			ZeroVec(*MV);
			M(*MV, *V);
			Gamma5Vec(*MV, cbsubset);
			std::shared_ptr<Spinor> Vvalues = M.tmp(info, nEv);
			ZeroVec(*Vvalues);
			AxpyVec(values, *V, *Vvalues, cbsubset);
			std::vector<double> resnorms2 = XmyNorm2Vec(*MV, *Vvalues, cbsubset);
			for (unsigned int i=0; i<nEv; i++)
				assert(std::sqrt(resnorms2[i]) <= fabs(values[0]) * eigs_params.RsdTarget * 6);
		}
	}
}

#else

namespace MG {
  	template<typename Spinor, typename LinOpT>
	void computeDeflation(const LatticeInfo& info, const LinOpT& M, EigsParams eigs_params, std::shared_ptr<Spinor>& V, std::vector<float>& values) {
		(void)info;
		(void)M;
		(void)eigs_params;
		(void)V;
		(void)values;

		throw std::runtime_error("deflation is not available: mg_proto was built without PRIMME");
	}
}

#endif // MG_USE_PRIMME


#endif // INCLUDE_LATTICE_COARSE_PRIMME_H_
