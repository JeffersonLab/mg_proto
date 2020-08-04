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
#include <numeric>
#include <stdexcept>
#include "lattice/coarse/coarse_op.h"
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

#ifdef MG_QMP_COMMS
#include <qmp.h>
#endif

namespace MG {

	namespace GlobalComm {

#ifdef MG_QMP_COMMS
		void GlobalSum(double& array) {
			QMP_sum_double_array(&array,1);
		}
#else
		void GlobalSum(double& array) {}
#endif
	}

	template<typename T>
	inline double sum(const std::vector<T>& v) {
		return std::accumulate(v.begin(), v.end(), 0.0);
	}

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
			        unsigned int K_distance, unsigned int probing_distance, const CBSubset subset, unsigned int mode=1) :
				_info(info), _M_fine(M_fine), _K_distance(K_distance), _op(*info), _subset(subset), _mode(mode)
			{
				AuxQF::subrogateTo(_M_fine.get());

				MasterLog(INFO,"ALI Solver constructor: mode= %d BEGIN", _mode);
				
				// Create projector
				_mg_deflation = std::make_shared<MGDeflation>(_info, _M_fine, defl_p, defl_solver_params, defl_eigs_params);

				// Create Multigrid preconditioner
				_mg_levels = std::make_shared<QPhiXMultigridLevelsEO>();
				SetupQPhiXMGLevels(prec_p, *_mg_levels, _M_fine);
				_vcycle = std::make_shared<VCycleRecursiveQPhiXEO2>(prec_vcycle_params, *_mg_levels);

				// Build K
				build_K(prec_p, prec_vcycle_params, prec_solver_params, K_distance, probing_distance);
				MasterLog(INFO,"ALI Solver constructor: END", _mode);

				// Hack vcycle
				set_smoother();

				AuxQ::clear();
				AuxQF::clear();
				AuxC::clear();
			}

			/*
			 * Apply the preconditioner onto 'in'.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 *
			 * It applies the deflation on the input vectors and return the results on 'out'.
			 *
			 *    out = [M^{-1}*Q + K*(I-Q)] * in,
			 *
			 * where Q = M_oo^{-1}*P*M_oo, P is a projector on M, and K approximates M^{-1}_oo*M_oo.
			 */

			std::vector<LinearSolverResults> operator()(QPhiXSpinor& out, const QPhiXSpinor& in, ResiduumType resid_type = RELATIVE) const override {
				(void)resid_type;
				//test_defl(in);
				//applyK(out, in);
				(*_vcycle)(out, in);
				IndexType ncol = out.GetNCol();
				return std::vector<LinearSolverResults>(ncol, LinearSolverResults());
			}

			template<typename Spinor>
			void apply_precon(Spinor& out, const Spinor& in, int recursive=0) const {
				// // TEMP!!!
				// double norm2_cb0 = sqrt(Norm2Vec(in, SUBSET_EVEN)[0]);
				// double norm2_cb1 = sqrt(Norm2Vec(in, SUBSET_ODD)[0]);
				// MasterLog(INFO,"MG Level 0: ALI Solver operator(): || v_e ||=%16.8e || v_o ||=%16.8e", norm2_cb0, norm2_cb1);

				assert(out.GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
				ZeroVec(*in_f);
				ConvertSpinor(in, *in_f, _subset);

				// I_Q_in = (I-Q)*in
				std::shared_ptr<QPhiXSpinorF> I_Q_in_f = AuxQF::tmp(*_info, ncol);
				std::shared_ptr<QPhiXSpinorF> VV_in_f = AuxQF::tmp(*_info, ncol);
				apply_complQ(*I_Q_in_f, *in_f, VV_in_f.get());
				in_f.reset();
		
				// M_oe*(M_ee^{-1}*M_eo*VV_in_o + VV_in_e)
				if (_mode == 0) {
					std::shared_ptr<QPhiXSpinorF> strange_VV_in_f = AuxQF::tmp(*_info, ncol);
					_M_fine->strangeOp(*strange_VV_in_f, *VV_in_f);
					if (recursive > 0) {
						std::shared_ptr<QPhiXSpinorF> Minv_oo_strange_VV_in_f = AuxQF::tmp(*_info, ncol);
						apply_precon(*Minv_oo_strange_VV_in_f, *strange_VV_in_f, recursive-1);
						YpeqXVec(*Minv_oo_strange_VV_in_f, *VV_in_f, _subset);
					} else {
						YpeqXVec(*strange_VV_in_f, *I_Q_in_f, _subset);
					}
				}
	
				// out_f = K * (I-Q)*in
				std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(*_info, ncol);
				applyK(*out_f, *I_Q_in_f);
				I_Q_in_f.reset();

				// out += VV_in
				YpeqXVec(*VV_in_f, *out_f, _subset);
				ZeroVec(out);
				ConvertSpinor(*out_f, out, _subset);
			}

			/**
			 * Return M^{-1}_oo * Q * in
			 *
			 * \param eo_solver: invertor on _M_fine
			 * \param out: (out) output vector
			 * \param in: input vector
			 */

			template<typename Spinor>
			void apply_invM_Q(Spinor& out, const Spinor& in) const {
				assert(in.GetNCol() == out.GetNCol());
				int ncol = in.GetNCol();

				Spinor in0(*_info, ncol);
				ZeroVec(in0);
				CopyVec(in0, in, _subset);
				_mg_deflation->VV(out, in0);

				// out_o + K * M_oe*(M_ee^{-1}*M_eo*out_o + out_e)
				if (_mode == 0 && _K_vals) {
					std::shared_ptr<QPhiXSpinorF> VV_in_f = AuxQF::tmp(*_info, ncol);
					ConvertSpinor(out, *VV_in_f);
					std::shared_ptr<QPhiXSpinorF> strange_VV_in_f = AuxQF::tmp(*_info, ncol);
					_M_fine->strangeOp(*strange_VV_in_f, *VV_in_f);
					std::shared_ptr<QPhiXSpinorF> K_f = AuxQF::tmp(*_info, ncol);
					applyK(*K_f, *strange_VV_in_f);
					YpeqXVec(*K_f, *VV_in_f, _subset);
					ConvertSpinor(*VV_in_f, out, _subset);
				}

				ZeroVec(out, _subset.complementary());
			}

			void test_defl(const QPhiXSpinor& in) const {
				IndexType ncol = in.GetNCol();

				// I_Q_in = (I-Q)*in = in - L^{-1} * P * in
				std::shared_ptr<QPhiXSpinor> I_Q_in = AuxQ::tmp(*_info, ncol);
				std::shared_ptr<QPhiXSpinor> invM_Q_in = AuxQ::tmp(*_info, ncol);
				apply_complQ(*I_Q_in, in);
				apply_invM_Q(*invM_Q_in, in);

				std::shared_ptr<QPhiXSpinorF> invM_Q_in_f = AuxQF::tmp(*_info, ncol);
				ZeroVec(*invM_Q_in_f);
				ConvertSpinor(*invM_Q_in, *invM_Q_in_f, _subset);
				std::shared_ptr<QPhiXSpinorF> MinvM_Q_in_f = AuxQF::tmp(*_info, ncol);
				(*_M_fine)(*MinvM_Q_in_f, *invM_Q_in_f);
				std::shared_ptr<QPhiXSpinor> MinvM_Q_in = AuxQ::tmp(*_info, ncol);
				ConvertSpinor(*MinvM_Q_in_f, *MinvM_Q_in, SUBSET_ODD);

				YpeqXVec(*MinvM_Q_in, *I_Q_in, SUBSET_ODD);
				YmeqXVec(in, *I_Q_in, SUBSET_ODD);
			 	std::vector<double> n_in = Norm2Vec(in);
			 	std::vector<double> n_diff = Norm2Vec(*I_Q_in);
				for (int col=0; col<ncol; col++)
			 		MasterLog(INFO,"MG Level 0: ALI Solver test_defl: error= %16.8e", sqrt(n_diff[col]/n_in[col]));
				
			}
	
			const LatticeInfo& GetInfo() const override { return *_info; }
			const CBSubset& GetSubset() const override { return SUBSET_ALL; }

			const std::shared_ptr<const QPhiXWilsonCloverEOLinearOperatorF> GetM() const { return _M_fine; }
			const std::shared_ptr<MGDeflation> GetMGDeflation() const { return _mg_deflation; }


		private:

			struct S : public Smoother<QPhiXSpinorF,QPhiXGaugeF>,
			public LinearSolver<QPhiXSpinorF,QPhiXGaugeF> {
				S(const ALIPrec& aliprec) : _aliprec(aliprec) {}
				std::vector<LinearSolverResults> operator()(QPhiXSpinorF& out, const QPhiXSpinorF& in, ResiduumType resid_type = RELATIVE) const override {
					(void)resid_type;
					_aliprec.apply_precon(out, in);
					return std::vector<LinearSolverResults>(in.GetNCol(), LinearSolverResults());
				}
				void operator()(QPhiXSpinorF& out, const QPhiXSpinorF& in) const override {
					_aliprec.apply_precon(out, in);
				}
				const CBSubset& GetSubset() const override { return SUBSET_ODD; };
				const LatticeInfo& GetInfo() const override { return _aliprec.GetInfo(); }
				const ALIPrec& _aliprec;
			};

			void set_smoother() {
				_antipostsmoother = std::make_shared<S>(*this);
				//_vcycle->SetAntePostSmoother(_antipostsmoother.get());
				_vcycle->GetPostSmoother()->setPrec(_antipostsmoother.get());
			}

			/**
			 * Return (I-Q) * in
			 *
			 * \param out: (out) output vector
			 * \param in: input vector
			 *
			 * NOTE: Assuming 'in' is properly zeroed
			 */

			void apply_complQ(QPhiXSpinor& out, const QPhiXSpinor& in) const {
				assert(out.GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
				ZeroVec(*in_f);
				ConvertSpinor(in, *in_f, _subset);
				std::shared_ptr<QPhiXSpinorF> out_f = AuxQF::tmp(*_info, ncol);
				apply_complQ(*out_f, *in_f);
				ZeroVec(out);
				ConvertSpinor(*out_f, out, _subset);
			}

			void apply_complQ(QPhiXSpinorF& out, const QPhiXSpinorF& in, QPhiXSpinorF* VVin=nullptr) const {
				assert(out.GetNCol() == in.GetNCol());
				assert(!VVin || VVin->GetNCol() == in.GetNCol());
				IndexType ncol = out.GetNCol();

				if (_mode == 0) {
					// Q = (I-P)
					std::shared_ptr<QPhiXSpinorF> VVin0;
					if (!VVin) {
						VVin0 = AuxQF::tmp(*_info, ncol);
						VVin = VVin0.get();
					}
					_mg_deflation->VV(*VVin, in);
					std::shared_ptr<QPhiXSpinorF> AVVin = AuxQF::tmp(*_info, ncol);
					_M_fine->unprecOp(*AVVin, *VVin);
					VVin0.reset(); VVin = nullptr;
					
					ZeroVec(out);
					CopyVec(out, in, _subset);
					YmeqXVec(*AVVin, out, _subset);
				} else if (_mode == 1) {
					// Q = (I-L^{-1}*P*L)
					std::shared_ptr<QPhiXSpinorF> VVin0;
					if (!VVin) {
						VVin0 = AuxQF::tmp(*_info, ncol);
						VVin = VVin0.get();
					}
					_mg_deflation->VV(*VVin, in);
					std::shared_ptr<QPhiXSpinorF> AVVin = AuxQF::tmp(*_info, ncol);
					_M_fine->unprecOp(*AVVin, *VVin);
					VVin0.reset(); VVin = nullptr;
					
					ZeroVec(out);
					CopyVec(out, in, _subset);
					YmeqXVec(*AVVin, out, _subset);
				} else {
					assert(false);
				}
			}

			/**
			 * Return M^{-1}_oo * (I-Q) * in
			 *
			 * \param eo_solver: invertor on _M_fine
			 * \param out: (out) output vector
			 * \param in: input vector
			 */

			void apply_invM_after_defl(const FGMRESSolverQPhiXF& eo_solver, QPhiXSpinorF& out, const QPhiXSpinorF& in) const {
				assert(in.GetNCol() == out.GetNCol());
				int ncol = in.GetNCol();
				
				// I_Q_in = (I-Q)*in = in - M_oo^{-1} * P * in
				std::shared_ptr<QPhiXSpinorF> in_f = AuxQF::tmp(*_info, ncol);
				ZeroVec(*in_f);
				ConvertSpinor(in, *in_f, _subset);
				std::shared_ptr<QPhiXSpinorF> I_Q_in = AuxQF::tmp(*_info, ncol);
				apply_complQ(*I_Q_in, *in_f);

				// out = M^{-1} * (I-Q) * in
				ZeroVec(out);
				std::vector<MG::LinearSolverResults> res = eo_solver(out, *I_Q_in, RELATIVE);
			}

			void build_K(SetupParams p, std::vector<MG::VCycleParams> vcycle_params, LinearSolverParamsBase solver_params,
			             unsigned int K_distance, unsigned int probing_distance, unsigned int blocking=32)
			{
				if (K_distance == 0) return;
				if (K_distance > 1)
					throw std::runtime_error("Not implemented 'K_distance' > 1");

				FGMRESSolverQPhiXF eo_solver(*_M_fine, solver_params, _vcycle.get());

				std::shared_ptr<Coloring> coloring = get_good_coloring(eo_solver, probing_distance, solver_params.RsdTarget * 2);
				unsigned int num_probing_vecs = coloring->GetNumSpinColorColors();
				for (unsigned int col=0, nc=std::min(num_probing_vecs, blocking); col < num_probing_vecs; col+=nc, nc=std::min(num_probing_vecs - col, blocking)) {

					// p(i) is the probing vector for the color col+i
					std::shared_ptr<QPhiXSpinorF> p = AuxQF::tmp(*_info, nc);
					coloring->GetProbingVectors(*p, col);

					// sol = inv(M_fine) * (I-P) * p
					std::shared_ptr<QPhiXSpinorF> sol = AuxQF::tmp(*_info, nc);
					apply_invM_after_defl(eo_solver, *sol, *p);
					p.reset();

					// Update K from sol
					update_K_from_probing_vecs(*coloring, col, sol);
				}

				test_K(eo_solver);
			}

			void update_K_from_probing_vecs(const Coloring& coloring, unsigned int c0, const std::shared_ptr<QPhiXSpinorF> sol) {
				if (!_K_vals) {
					_K_vals = std::make_shared<CoarseGauge>(*_info);
					ZeroGauge(*_K_vals);
				}

				IndexType num_cbsites = _info->GetNumCBSites();
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
									int g5 = (spin < num_spin/2 ? 1 : -1) * (col_spin < num_spin/2 ? 1 : -1);
									_K_vals->GetSiteDiagData(cb,cbsite,col_spin,col_color,spin,color,RE) += (*sol)(col,cb,cbsite,spin,color,0)/2;
									_K_vals->GetSiteDiagData(cb,cbsite,col_spin,col_color,spin,color,IM) += (*sol)(col,cb,cbsite,spin,color,1)/2;
									_K_vals->GetSiteDiagData(cb,cbsite,spin,color,col_spin,col_color,RE) += (*sol)(col,cb,cbsite,spin,color,0)/2*g5;
									_K_vals->GetSiteDiagData(cb,cbsite,spin,color,col_spin,col_color,IM) -= (*sol)(col,cb,cbsite,spin,color,1)/2*g5;
								}
							}
						}
					}
				}
			}

			std::shared_ptr<Coloring> get_good_coloring(const FGMRESSolverQPhiXF& eo_solver, unsigned int max_probing_distance, double tol) {
				// Returned coloring
				std::shared_ptr<Coloring> coloring;

				// Build probing vectors to get the exact first columns for ODD site 0
				std::shared_ptr<QPhiXSpinorF> e = AuxQF::tmp(*_info, _info->GetNumColorSpins());
				ZeroVec(*e);
				if (_info->GetNodeInfo().NodeID() == 0) {
					for(int color=0; color < _info->GetNumColors(); ++color) {
						for(int spin=0; spin < _info->GetNumSpins(); ++spin) {
							int sc = color * _info->GetNumSpins() + spin;
							(*e)(sc,ODD,0,spin,color,0) = 1.0;
						}
					}
				}

				// sol_e = inv(M_fine) * (I-P) * e
				std::shared_ptr<QPhiXSpinorF> sol_e = AuxQF::tmp(*_info, _info->GetNumColorSpins());
				apply_invM_after_defl(eo_solver, *sol_e, *e);

				unsigned int probing_distance = 0;
				while(probing_distance <= max_probing_distance) {
					// Create coloring
					coloring = std::make_shared<Coloring>(_info, probing_distance, SUBSET_ODD);

					// Get the probing vectors for the ODD site 0
					unsigned int color_node = coloring->GetColorCBIndex(ODD, 0);
					std::shared_ptr<QPhiXSpinorF> p = AuxQF::tmp(*_info, _info->GetNumColorSpins());
					coloring->GetProbingVectors(*p, coloring->GetSpinColorColor(color_node, 0, 0));

					// sol_p = inv(M_fine) * (I-P) * p
					std::shared_ptr<QPhiXSpinorF> sol_p = AuxQF::tmp(*_info, _info->GetNumColorSpins());
					apply_invM_after_defl(eo_solver, *sol_p, *p);

					// Compute ||sol[0] - K[0,0]||_F / ||sol[0]||_F
					double sol_F = 0.0, diff_F = 0.0;
					if (_info->GetNodeInfo().NodeID() == 0) {
						std::vector<std::complex<float>> sol_p_00(_info->GetNumColorSpins() * _info->GetNumColorSpins());
						for(int colorj=0; colorj < _info->GetNumColors(); ++colorj) {
							for(int spinj=0; spinj < _info->GetNumSpins(); ++spinj) {
								int sc_e_j = colorj * _info->GetNumSpins() + spinj;
								int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
								for(int color=0; color < _info->GetNumColors(); ++color) {
									for(int spin=0; spin < _info->GetNumSpins(); ++spin) {
										std::complex<float> sol_e_ij((*sol_e)(sc_e_j,ODD,0,spin,color,0), (*sol_e)(sc_e_j,ODD,0,spin,color,1));
										std::complex<float> sol_p_ij((*sol_p)(sc_p_j,ODD,0,spin,color,0), (*sol_p)(sc_p_j,ODD,0,spin,color,1));
										sol_F += (sol_e_ij * std::conj(sol_e_ij)).real();
										std::complex<float> diff_ij = sol_e_ij - sol_p_ij;
										diff_F += (diff_ij * std::conj(diff_ij)).real();

										int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
										sol_p_00[sc_p_j * _info->GetNumColorSpins() + sc_p_i] = sol_p_ij;
									}
								}
							}
						}

						ZeroVec(*sol_p, SUBSET_ALL);

						for(int colorj=0; colorj < _info->GetNumColors(); ++colorj) {
							for(int spinj=0; spinj < _info->GetNumSpins(); ++spinj) {
								int sc_p_j = coloring->GetSpinColorColor(0, spinj, colorj);
								for(int color=0; color < _info->GetNumColors(); ++color) {
									for(int spin=0; spin < _info->GetNumSpins(); ++spin) {
										int sc_p_i = coloring->GetSpinColorColor(0, spin, color);
										(*sol_p)(sc_p_j,ODD,0,spin,color,0) = sol_p_00[sc_p_j * _info->GetNumColorSpins() + sc_p_i].real();
										(*sol_p)(sc_p_j,ODD,0,spin,color,1) = sol_p_00[sc_p_j * _info->GetNumColorSpins() + sc_p_i].imag();
									}
								}
							}
						}
					} else {
						ZeroVec(*sol_p, SUBSET_ALL);
					}
					GlobalComm::GlobalSum(diff_F);
					GlobalComm::GlobalSum(sol_F);

					std::shared_ptr<QPhiXSpinorF> aux = AuxQF::tmp(*_info, _info->GetNumColorSpins());
					double norm_sol_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));
					CopyVec(*aux, *sol_e);
					double norm_diff = sqrt(sum(XmyNorm2Vec(*aux, *sol_p, SUBSET_ODD)));
					
					std::shared_ptr<QPhiXSpinorF> M_inv_oo_M_oo_P = AuxQF::tmp(*_info, _info->GetNumColorSpins());
					apply_invM_Q(*M_inv_oo_M_oo_P, *e);
					YpeqXVec(*M_inv_oo_M_oo_P, *sol_p, SUBSET_ODD);
					(*_M_fine)(*aux, *sol_p);
					YmeqXVec(*e, *aux, SUBSET_ODD);
					double norm_F = sqrt(sum(Norm2Vec(*aux, SUBSET_ODD)));
					
					MasterLog(INFO, "K probing error with %d distance coloring: ||M^{-1}_00-K_00||_F/||M^{-1}_00||_F= %g ||M^{-1}_0-K_0||_F/||M^{-1}_0||_F= %g   ||M*K-I||= %g",
							probing_distance, norm_diff/norm_sol_e, sqrt(diff_F / sol_F), norm_F);

					if (diff_F <= sol_F * tol * tol) break;

					probing_distance++;
				}

				return coloring;
			}

			void test_K(const FGMRESSolverQPhiXF& eo_solver) {
				// Build probing vectors to get the exact first columns for ODD site 0
				const int nc = _info->GetNumColorSpins();
				std::shared_ptr<QPhiXSpinorF> e = AuxQF::tmp(*_info, nc);
				ZeroVec(*e);
				if (_info->GetNodeInfo().NodeID() == 0) {
					for(int color=0; color < _info->GetNumColors(); ++color) {
						for(int spin=0; spin < _info->GetNumSpins(); ++spin) {
							int sc = color * _info->GetNumSpins() + spin;
							(*e)(sc,ODD,0,spin,color,0) = 1.0;
						}
					}
				}

				// sol_e = inv(M_fine) * (I-Q) * e
				std::shared_ptr<QPhiXSpinorF> sol_e = AuxQF::tmp(*_info, nc);
				//apply_invM_after_defl(eo_solver, *sol_e, *e);
				eo_solver(*sol_e, *e);

				// sol_p \approx inv(M_fine) * (I-Q) * e
				std::shared_ptr<QPhiXSpinorF> sol_p = AuxQF::tmp(*_info, nc);
				std::shared_ptr<QPhiXSpinor> e_d = AuxQ::tmp(*_info, nc);
				std::shared_ptr<QPhiXSpinor> sol_p_d = AuxQ::tmp(*_info, nc);
				ConvertSpinor(*e, *e_d);
				(*this)(*sol_p_d, *e_d);
				ConvertSpinor(*sol_p_d, *sol_p);

				double norm_e = sqrt(sum(Norm2Vec(*sol_e, SUBSET_ODD)));
				double norm_diff = sqrt(sum(XmyNorm2Vec(*sol_e, *sol_p, SUBSET_ODD)));
				MasterLog(INFO, "K probing error : ||M^{-1}-K||_F= %g", norm_diff / norm_e);
			}

			/*
			 * Apply K. out = K * in.
			 *
			 * \param out: returned vectors
			 * \param in: input vectors
			 */

			template<typename Spinor>
			void applyK(Spinor& out, const Spinor& in) const {
				assert(out.GetNCol() == in.GetNCol());
				int ncol = in.GetNCol();

				if (_K_distance == 0) {
					// If no K, copy 'in' into 'out'
					ZeroVec(out, _subset);

				} else if (_K_distance == 1) {
					// Apply the diagonal of K
					std::shared_ptr<CoarseSpinor> in_c = AuxC::tmp(*_info, ncol);
					std::shared_ptr<CoarseSpinor> out_c = AuxC::tmp(*_info, ncol);
					ConvertSpinor(in, *in_c, _subset);
#pragma omp parallel
					{
						int tid = omp_get_thread_num();
						for(int cb = _subset.start; cb < _subset.end; ++cb) {
							_op.M_diag(*out_c,*_K_vals,*in_c,cb,LINOP_OP,tid);
						}
					}
					ConvertSpinor(*out_c, out, _subset);

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
			const unsigned int _mode;
			std::shared_ptr<QPhiXMultigridLevelsEO> _mg_levels;
			std::shared_ptr<VCycleRecursiveQPhiXEO2> _vcycle;
			std::shared_ptr<S> _antipostsmoother;
	};
}
	
#endif // INCLUDE_LATTICE_QPHIX_QPHIX_ALI_H_
