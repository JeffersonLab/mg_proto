/*
 * qphix_mgdeflation.h
 *
 *  Created on: May 29, 2020
 *      Author: Eloy Romero <eloy@cs.wm.edu>
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_


#include <MG_config.h>
#include <cassert>
#include <memory>
#include "lattice/qphix/qphix_eo_clover_linear_operator.h" // QPhiXWilsonCloverEOLinearOperatorF
#include "lattice/eigs_common.h" // EigsParams
#include "lattice/fine_qdpxx/mg_params_qdpxx.h" // SetupParams, FGMRESParams
#include "lattice/qphix/mg_level_qphix.h" // QPhiXMultigridLevels
#include "lattice/coarse/invfgmres_coarse.h" // UnprecFGMRESSolverCoarseWrapper
#include "lattice/qphix/qphix_types.h" // QPhiXSpinorF
#include "lattice/coarse/coarse_transfer.h" // CoarseTransfer
#include "lattice/qphix/qphix_transfer.h" // QPhiXTransfer
#include "lattice/coarse/coarse_deflation.h" // computeDeflation

namespace MG {

	/*
	 * Deflate an approximate invariant subspace of the multigrid prolongator
	 *
	 * If V is a subset of the multigrid prolongator at the coarsest level on a
	 * linear operator A, then MGDeflation::operator()(out, in) does:
	 *
	 *   out = V * inv(V^H * A * V) * V^H * A * in,
	 *
	 * where V^H * A * V is numerically close to a diagonal matrix. 
	 */

	class MGDeflation : public AuxiliarySpinors<QPhiXSpinorF>, public AuxiliarySpinors<CoarseSpinor>
	{
		using AuxQ = AuxiliarySpinors<QPhiXSpinorF>;
		using AuxC = AuxiliarySpinors<CoarseSpinor>;

		public:

			/*
			 * Constructor
			 *
			 * \param p: Multigrid parameters
			 * \param M_fine: original operator
			 * \param rank: rank of the deflated space
			 * \param gap: lower bound of the expected reduction of the conditioning number;
			 *        ie., lower bound of lambda_{rank-1}/lambda_0 where lambda_i are the largest eigenvalues
			 *        of the inverse of the coarse operator.
			 */

			MGDeflation(const std::shared_ptr<LatticeInfo> info, const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& M_fine, SetupParams p, FGMRESParams solver_params, EigsParams eigs_params)
				: _info(info), _M_fine(M_fine)
			{
				// Setup multigrid
				SetupQPhiXMGLevels(p, _mg_levels, _M_fine);
				if (_mg_levels.coarse_levels.size() <= 0)
					return;
				int n_levels = _mg_levels.coarse_levels.size()+2;

				// Generate the prolongators and restrictiors
				_Transfer_coarse_level.resize(_mg_levels.coarse_levels.size() - 1);
				for(int coarse_idx=n_levels-2; coarse_idx >= 0; --coarse_idx) {
					_Transfer_coarse_level[coarse_idx] = std::make_shared<CoarseTransfer>(_mg_levels.coarse_levels[coarse_idx].blocklist, _mg_levels.coarse_levels[coarse_idx].null_vecs);
				}
				_Transfer_fine_level = std::make_shared<QPhiXTransfer<QPhiXSpinorF>>(_mg_levels.fine_level.blocklist, _mg_levels.fine_level.null_vecs);

				// Create the solver for the coarsest level
				auto this_level_linop = _mg_levels.coarse_levels.back().M;
				solver_params.RsdTarget = std::min(solver_params.RsdTarget, eigs_params.RsdTarget*0.6);
				UnprecFGMRESSolverCoarseWrapper solver(this_level_linop, solver_params, nullptr);
				//FGMRESSolverCoarse solver(this_level_linop, solver_params, nullptr);

				// Compute the left singular vectors, the right singular are \gamma_5 * left singular vectors
				MasterLog(INFO, "Computing deflation for level %d", _mg_levels.coarse_levels.size());
				computeDeflation(*_mg_levels.coarse_levels.back().info, solver, eigs_params, _eigenvectors, _eigenvalues);
			}

			/*
			 * Apply oblique deflation on vectors 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * It applies the deflation on the input vectors and return the results on 'out':
			 *
			 *   out = V * inv(V^H * A * V) * V^H * A * in,
			 */

			void VVA(QPhiXSpinor& out, const QPhiXSpinor& in) const {
				apply(out, in, true);
			}

			/*
			 * Apply oblique deflation on vectors 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * It applies the deflation on the input vectors and return the results on 'out':
			 *
			 *   out = A * V * inv(V^H * A * V) * V^H * in,
			 */

			void AVV(QPhiXSpinor& out, const QPhiXSpinor& in) const {
				apply(out, in, false);
			}

			/*
			 * Apply oblique deflation on vectors 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * It applies the deflation on the input vectors and return the results on 'out':
			 *
			 *   out = V * inv(V^H * A * V) * V^H * A * in,
			 */

			void operator()(QPhiXSpinor& out, const QPhiXSpinor& in) const {
				VVA(out, in);
			}

			/*
			 * Apply oblique deflation on vectors 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 * \param do_VVA: if true, 
			 *          out = V * inv(V^H * A * V) * V^H * A * in,
			 *        otherwise
			 *          out = A * V * inv(V^H * A * V) * V^H * in.
			 *
			 * It applies the deflation on the input vectors and return the results on 'out'.
			 */

			void apply(QPhiXSpinor& out, const QPhiXSpinor& in, bool do_VVA=true) const {
				assert(out.GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				// Convert to float
				std::shared_ptr<QPhiXSpinorF> in_f = AuxQ::tmp(*_mg_levels.fine_level.info, ncol);
				ZeroVec(*in_f, SUBSET_ALL);
				ConvertSpinor(in,*in_f);

				// Ain = A * in if do_VVA else in
				std::shared_ptr<QPhiXSpinorF> Ain_f;
				if (do_VVA) {
					Ain_f = AuxQ::tmp(*_mg_levels.fine_level.info, ncol);
					(*_M_fine)(*Ain_f, *in_f, LINOP_OP);
				} else {
					Ain_f = in_f;
				}
				in_f.reset();

				// Transfer to first coarse level
				std::shared_ptr<CoarseSpinor> coarse_in = AuxC::tmp(*_mg_levels.coarse_levels[0].info, ncol);
				_Transfer_fine_level->R(*Ain_f, *coarse_in);
				Ain_f.reset();

				// Transfer to the deepest level
				for(int coarse_idx=1; coarse_idx < _mg_levels.coarse_levels.size(); coarse_idx++) {
					std::shared_ptr<CoarseSpinor> in_level = AuxC::tmp(*_mg_levels.coarse_levels[coarse_idx].info, ncol);
					_Transfer_coarse_level[coarse_idx]->R(*coarse_in, *in_level);
					coarse_in = in_level;
				}

				// Apply gamma_5
				Gamma5Vec(*coarse_in);

				// Apply deflation at the coarsest level
				std::shared_ptr<CoarseSpinor> coarse_out = AuxC::tmp(*_mg_levels.coarse_levels.back().info, ncol);
				project_coarse_level(*coarse_out, *coarse_in);
				coarse_in.reset();

				// Transfer to the first coarse level
				for(int coarse_idx=_mg_levels.coarse_levels.size()-2; coarse_idx >= 0; coarse_idx--) {
					std::shared_ptr<CoarseSpinor> out_level = AuxC::tmp(*_mg_levels.coarse_levels[coarse_idx].info, ncol);
					_Transfer_coarse_level[coarse_idx]->P(*coarse_out, *out_level);
					coarse_out = out_level;
				}

				// Transfer to the fine level
				std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels.fine_level.info, ncol);
				_Transfer_fine_level->P(*coarse_out, *out_f);

				// Aout_f = out_f if do_VVA else A*out_f
				std::shared_ptr<QPhiXSpinorF> Aout_f;
				if (do_VVA) {
					Aout_f = out_f;
				} else {
					Aout_f = AuxQ::tmp(*_mg_levels.fine_level.info, ncol);
					(*_M_fine)(*Aout_f, *out_f, LINOP_OP);
				}
				out_f.reset();

				// Convert back to double
				ConvertSpinor(*Aout_f, out);
			}

			/*
			 * Return the columns of V starting from i-th column.
			 *
			 * \param i: index of the first column to return
			 * \param out: output vectors
			 */

			void V(unsigned int i, QPhiXSpinor& out) const {
				assert(out.GetNCol() + i <= _eigenvectors->GetNCol());
				IndexType ncol = out.GetNCol();

				// Apply deflation at the coarsest level
				std::shared_ptr<CoarseSpinor> coarse_out = AuxC::tmp(*_mg_levels.coarse_levels.back().info, ncol);
				CopyVec(*coarse_out, 0, ncol, *_eigenvectors, i);

				// Transfer to the first coarse level
				for(int coarse_idx=_mg_levels.coarse_levels.size()-2; coarse_idx >= 0; coarse_idx--) {
					std::shared_ptr<CoarseSpinor> out_level = AuxC::tmp(*_mg_levels.coarse_levels[coarse_idx].info, ncol);
					_Transfer_coarse_level[coarse_idx]->P(*coarse_out, *out_level);
					coarse_out = out_level;
				}

				// Transfer to the fine level
				std::shared_ptr<QPhiXSpinorF> out_f = AuxQ::tmp(*_mg_levels.fine_level.info, ncol);
				_Transfer_fine_level->P(*coarse_out, *out_f);

				// Convert back to double
				ConvertSpinor(*out_f, out);
			}


			/*
			 * Return the diagonal of inv(X'*P' * A * P*X)
			 *
			 * If P is the multigrid prolongator at the deepest coarse level on a
			 * linear operator A. Then MGDeflation::operator()(out, in) does:
			 *
			 *   out = in - P*X * inv(X'*P' * \gamma_5 * A * P*X) * X'*P' * \gamma_5 * A * in,
			 *
			 * where X are the eigenvectors of the largest eigenvalues on the coarse operator,
			 * P' * \gamma_5 * A * P.
			 */

			std::vector<std::complex<double>> GetDiagInvCoarse() const {
				// Compute \gamma_5 * _eigenvectors	
				std::shared_ptr<CoarseSpinor> g5_eigenvectors = AuxC::tmp(*_mg_levels.coarse_levels.back().info, _eigenvectors->GetNCol());
				CopyVec(*_eigenvectors, *g5_eigenvectors);
				Gamma5Vec(*g5_eigenvectors);

				// Compute d .* _eigenvalues
				std::vector<std::complex<double>> d = InnerProductVec(*_eigenvectors, *g5_eigenvectors);
				for (unsigned int i=0; i<d.size(); i++)
					d[i] *= _eigenvalues[i];

				return d;
			}

			const LatticeInfo& GetInfo() { return *_info; }

			unsigned int GetRank() { return _eigenvalues.size(); }

		private:

			/*
			 * Multiply by the restricted coarsest operator.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * Return the :
			 *
			 *   out = X * \Lambda * X' * in,
			 */

			void project_coarse_level(CoarseSpinor& out, CoarseSpinor& in) const {
				assert(out.GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				// Do ip = X' * in
				unsigned int nEv = _eigenvectors->GetNCol();
				std::vector<std::complex<double>> ip = InnerProductMat(*_eigenvectors, in);

				// Do ip = \Lambda * ip
				for (unsigned int j=0; j<ncol; ++j)
					for (unsigned int i=0; i<nEv; ++i)
						ip[i + j*nEv] *= _eigenvalues[i];

				// out = X * ip
				UpdateVecs(*_eigenvectors, ip, out);
			}

			const std::shared_ptr<LatticeInfo> _info;
			const std::shared_ptr<QPhiXWilsonCloverEOLinearOperatorF>& _M_fine;
			QPhiXMultigridLevelsEO _mg_levels;
			std::shared_ptr<CoarseSpinor> _eigenvectors; // eigenvectors of \gamma_5 * inv(A)
			std::vector<float> _eigenvalues; // eigenvalues of \gamma_5 * inv(A)
			std::shared_ptr<QPhiXTransfer<QPhiXSpinorF>> _Transfer_fine_level;
			std::vector<std::shared_ptr<CoarseTransfer>> _Transfer_coarse_level;
	};

}

#endif // INCLUDE_LATTICE_QPHIX_QPHIX_DISCO_H_
