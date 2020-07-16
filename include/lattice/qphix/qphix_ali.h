/*
 * qphix_ali.h
 *
 *  Created on: July 11, 2020
 *      Author: Eloy Romero <eloy@cs.wm.edu>
 */

#ifndef INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
#define INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_


#include <MG_config.h>
#include <cassert>
#include <memory>
#include <stdexcept>
#include "lattice/qphix/qphix_mgdeflation.h" // MGDeflation
#include "lattice/qphix/qphix_eo_clover_linear_operator.h" // QPhiXWilsonCloverEOLinearOperatorF
#include "lattice/eigs_common.h" // EigsParams
#include "lattice/fine_qdpxx/mg_params_qdpxx.h" // SetupParams
#include "lattice/qphix/mg_level_qphix.h" // QPhiXMultigridLevels
#include "lattice/coarse/invfgmres_coarse.h" // UnprecFGMRESSolverCoarseWrapper
#include "lattice/qphix/qphix_types.h" // QPhiXSpinorF
#include "lattice/coarse/coarse_transfer.h" // CoarseTransfer
#include "lattice/qphix/qphix_transfer.h" // QPhiXTransfer
#include "lattice/coarse/coarse_deflation.h" // computeDeflation
#include "lattice/qphix/vcycle_recursive_qphix.h" // VCycleRecursiveQPhiXEO2
#include "lattice/coloring.h" // Coloring

namespace MG {

	/*
	 * Solve a linear system using the Approximate Lattice Inverse as a preconditioner
	 *
	 * If K is the ALI preconditioner and P is an approximate projector on the lower part of A's spectrum,
	 * then the linear system A*x = b is solved as x = y + A^{-1}*P*b where K*A*y = K*(I-P)*b. The preconditioner
	 * K approximates the links of A^{-1} for near neighbor sites. The approach is effective if |[(I-P)*A^{-1}]_ij|
	 * decays quickly as i and j are further apart sites.
	 *
	 * The projector is built using multigrid deflation (see MGDeflation) and K is reconstructed with probing based
	 * on coloring the graph lattice.
	 */

	class ALIPrec : public LinearSolver<QPhiXSpinor, QPhiXGauge>,
			public AuxiliarySpinors<QPhiXSpinorF>, public AuxiliarySpinors<CoarseSpinor>
	{
		using AuxQ = AuxiliarySpinors<QPhiXSpinor>;
		using AuxQF = AuxiliarySpinors<QPhiXSpinorF>;
		using AuxC = AuxiliarySpinors<CoarseSpinor>;

		public:

			/*
			 * Constructor
			 *
			 * \param info: lattice info
			 * \param M_fine: linear system operator (A)
			 * \param defl_p: Multigrid parameters used to build the multgrid deflation
			 * \param defl_solver_params: linear system parameters to build the multigrid deflation
			 * \param defl_eigs_params: eigensolver parameters to build the multigrid deflation
			 * \param prec_p: Multigrid parameters used to build the preconditioner
			 * \param K_distance: maximum distance of the approximated links
			 * \param probing_distance: maximum distance for probing
			 *
			 * The parameters defl_p, defl_solver_params and defl_eigs_params are passed to MGDeflation to build
			 * the projector P. The interesting values of (I-P)*A^{-1} are reconstructed with a probing scheme
			 * that remove contributions from up to 'probing_distance' sites.
			 */

			ALIPrec(const std::shared_ptr<LatticeInfo> info, const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> M_fine,
			        SetupParams defl_p, LinearSolverParamsBase defl_solver_params, EigsParams defl_eigs_params, SetupParams prec_p,
			        std::vector<MG::VCycleParams> prec_vcycle_params, LinearSolverParamsBase prec_solver_params,
			        unsigned int K_distance, unsigned int probing_distance, const CBSubset subset) :
				_info(info), _M_fine(M_fine), _K_distance(K_distance), _op(*info), _subset(subset)
			{
				// Create projector
				_mg_deflation = std::make_shared<MGDeflation>(_info, _M_fine, defl_p, defl_solver_params, defl_eigs_params);

				// Build K
				build_K(prec_p, prec_vcycle_params, prec_solver_params, K_distance, probing_distance);
			}

			/*
			 * Apply the preconditioner onto 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * It applies the deflation on the input vectors and return the results on 'out'.
			 *
			 * out = A^{-1}*P*in + K*(I-P)*in
			 */

			std::vector<LinearSolverResults> operator()(QPhiXSpinor& out, const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE) const override {
				(void)resid_type;

				assert(out.GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				// in_minus_Pin = in - P*in
				std::shared_ptr<QPhiXSpinor> P_in = AuxQ::tmp(*_info, ncol);
				_mg_deflation->AVV(*P_in, in);
				std::shared_ptr<QPhiXSpinor> in_minus_P_in = AuxQ::tmp(*_info, ncol);
				CopyVec(*in_minus_P_in, in);
				YmeqXVec(*P_in, *in_minus_P_in);
				P_in.reset();
			
				// out = K * in_minus_P_in
				applyK(out, *in_minus_P_in);
				in_minus_P_in.reset();

				// invAP_in = A^{-1} * in
				std::shared_ptr<QPhiXSpinor> invAP_in = AuxQ::tmp(*_info, ncol);
				_mg_deflation->VV(*invAP_in, in);

				// out += invAP_in
				YpeqXVec(*invAP_in, out);

				return std::vector<LinearSolverResults>(ncol, LinearSolverResults());
			}

			const LatticeInfo& GetInfo() const override { return *_info; }
			const CBSubset& GetSubset() const override { return SUBSET_ALL; }

			const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> GetM() const { return _M_fine; }


		private:

			void build_K(SetupParams p, std::vector<MG::VCycleParams> vcycle_params, LinearSolverParamsBase solver_params,
			             unsigned int K_distance, unsigned int probing_distance, unsigned int blocking=32)
			{
				if (K_distance == 0) return;
				if (K_distance > 1)
					throw std::runtime_error("Not implemented 'K_distance' > 1");

				IndexType num_colorspin = _info->GetNumColorSpins();

				QPhiXMultigridLevelsEO mg_levels;
				SetupQPhiXMGLevels(p, mg_levels, _M_fine);
				VCycleRecursiveQPhiXEO2 vcycle(vcycle_params, mg_levels);
				FGMRESSolverQPhiXF eo_solver(*_M_fine, solver_params, &vcycle);

				Coloring coloring(_info, probing_distance);
				unsigned int num_probing_vecs = coloring.GetNumSpinColorColors();
				for (unsigned int col=0, nc=std::min(num_probing_vecs, blocking); col < num_probing_vecs; col+=nc, nc=std::min(num_probing_vecs - col, blocking)) {

					// p(i) is the probing vector for the color col+i
					std::shared_ptr<QPhiXSpinorF> p = AuxQF::tmp(*_info, nc);
					coloring.GetProbingVectors(*p, col);

					// Pp = P * p
					std::shared_ptr<QPhiXSpinorF> Pp = AuxQF::tmp(*_info, nc);
					_mg_deflation->AVV(*Pp, *p);

					// p_minus_Pp = p - Pp
					std::shared_ptr<QPhiXSpinorF> p_minus_Pp = AuxQF::tmp(*_info, nc);
					CopyVec(*p_minus_Pp, *p);
					YmeqXVec(*Pp, *p_minus_Pp);
					p.reset();
					Pp.reset();
					
					// sol = inv(M_fine) * p_minus_Pp
					std::shared_ptr<QPhiXSpinorF> sol = AuxQF::tmp(*_info, nc);
					std::vector<MG::LinearSolverResults> res = eo_solver(*sol, *p_minus_Pp, RELATIVE);
					p_minus_Pp.reset();

					// Update K from sol
					update_K_from_probing_vecs(coloring, col, sol);
				}
			}

			void update_K_from_probing_vecs(const Coloring& coloring, unsigned int c0, const std::shared_ptr<QPhiXSpinorF> sol) {
				if (!_K_vals) _K_vals = std::make_shared<CoarseGauge>(*_info);

				// Local lattice size and its origin
				const IndexArray& lattice_dims = _info->GetLatticeDimensions();
				const IndexType cborig = _info->GetCBOrigin();
				IndexArray global_lattice_dims;
				_info->LocalDimsToGlobalDims(global_lattice_dims, lattice_dims);

				IndexType num_cbsites = _info->GetNumCBSites();
				IndexType num_colorspin = _info->GetNumColorSpins();
				IndexType num_color = _info->GetNumColors();
				IndexType num_spin = _info->GetNumSpins();
				IndexType ncol = sol->GetNCol();
				CBSubset subset = SUBSET_ALL;

				// Loop over the sites and sum up the norm
#pragma omp parallel for collapse(3) schedule(static)
				for(int cb=subset.start; cb < subset.end; ++cb ) {
					for(int cbsite = 0; cbsite < num_cbsites; ++cbsite ) {
						for(int col = 0; col < ncol; ++col ) {
							// Decompose the color into the node's color and the spin-color components
							IndexType col_spin, col_color, node_color;
							coloring.SpinColorColorComponents(c0 + col, node_color, col_spin, col_color);
							unsigned int colorj = coloring.GetColorCBIndex(cb, cbsite);

							// Process this site if its color is the same as the color of the probing vector
							if (colorj != node_color) continue;

							// Get diag
							for(int color=0; color < num_color; ++color) {
								for(int spin=0; spin < num_spin; ++spin) {
									_K_vals->GetSiteDiagData(cb,cbsite,col_spin,col_color,spin,color,RE) = (*sol)(col,cb,cbsite,spin,color,0);
									_K_vals->GetSiteDiagData(cb,cbsite,col_spin,col_color,spin,color,IM) = (*sol)(col,cb,cbsite,spin,color,1);
								}
							}
						}
					}
				}
			}

			/*
			 * Apply K. out = K * in.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 */

			void applyK(QPhiXSpinor& out, const QPhiXSpinor& in) const {
				assert(out.GetNCol() == in.GetNCol());
				int ncol = in.GetNCol();

				if (_K_distance == 0) {
					// If no K, copy 'in' into 'out'
					CopyVec(out, in);

				} else if (_K_distance == 1) {
					// Apply the diagonal of K
					std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(*_info, ncol);
					std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(*_info, ncol);
					ConvertSpinor(in, *in_c);
#pragma omp parallel
					{
						int tid = omp_get_thread_num();
						for(int cb = _subset.start; cb < _subset.end; ++cb) {
							_op.M_diag(*out_c,*_K_vals,*in_c,cb,LINOP_OP,tid);
						}
					}
					ConvertSpinor(*out_c, out);

				} else if (_K_distance == 2) {
					// Apply the whole operator
					std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(*_info, ncol);
					std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(*_info, ncol);
					ConvertSpinor(in, *in_c);
#pragma omp parallel
					{
						int tid = omp_get_thread_num();
						for(int cb = _subset.start; cb < _subset.end; ++cb) {
							_op.unprecOp(*out_c,*_K_vals,*in_c,cb,LINOP_OP,tid);
						}
					}
					ConvertSpinor(*out_c, out);

				} else {
					assert(false);
				}
			}
	
			const std::shared_ptr<LatticeInfo> _info;
			const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> _M_fine;
			std::shared_ptr<MGDeflation> _mg_deflation;
			std::shared_ptr<CoarseGauge> _K_vals;
			unsigned int _K_distance;
			const CoarseDiracOp _op;
			const CBSubset _subset;
	};
}
	
#endif // INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
