/*
 * kokkos_matvec.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_VMATVEC_H_
#define TEST_KOKKOS_VMATVEC_H_

#include "kokkos_defaults.h"
#include "kokkos_constants.h"
#include "kokkos_types.h"
#include "kokkos_ops.h"
#include "kokkos_vectype.h"

namespace MG
{


	template<typename GT, typename VN, typename ST, int dir>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					ComplexCMadd(v_out(row,spin), gauge_in(i,dir,row,col), v_in(col,spin));
				}
			}
		}


	}

	// Permute Versions
	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor0_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					ComplexCMadd(v_out(row,spin), VN::permute0(gauge_in(i,0,row,col)), v_in(col,spin));
				}
			}
		}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor1_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					ComplexCMadd(v_out(row,spin), VN::permute1(gauge_in(i,1,row,col)), v_in(col,spin));
				}
			}
		}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor2_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					ComplexCMadd(v_out(row,spin), VN::permute2(gauge_in(i,2,row,col)), v_in(col,spin));
				}
			}
		}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_u_halfspinor3_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

		for(int row=0; row < 3; ++row) {

			for(int spin = 0; spin < 2; ++spin) {
				ComplexZero(v_out(row,spin));
			}

			for(int col=0; col < 3; ++col) {
				for(int spin = 0; spin < 2; ++spin) {
					ComplexCMadd(v_out(row,spin), VN::permute3(gauge_in(i,3,row,col)), v_in(col,spin));
				}
			}
		}


	}







	template<typename GT, typename VN, typename ST, int dir>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
				  for(int row=0; row < 3; ++row) {
				    for(int spin=0; spin < 2; ++spin) {
				      ComplexConjMadd(v_out(row,spin), gauge_in(i,dir,col,row), v_in(col,spin));
				    }
				  }
				}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor0_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
				  for(int row=0; row < 3; ++row) {
				    for(int spin=0; spin < 2; ++spin) {
				      ComplexConjMadd(v_out(row,spin), VN::permute0(gauge_in(i,0,col,row)), v_in(col,spin));
				    }
				  }
				}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor1_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
				  for(int row=0; row < 3; ++row) {
				    for(int spin=0; spin < 2; ++spin) {
				      ComplexConjMadd(v_out(row,spin), VN::permute1(gauge_in(i,1,col,row)), v_in(col,spin));
				    }
				  }
				}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor2_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
				  for(int row=0; row < 3; ++row) {
				    for(int spin=0; spin < 2; ++spin) {
				      ComplexConjMadd(v_out(row,spin), VN::permute2(gauge_in(i,2,col,row)), v_in(col,spin));
				    }
				  }
				}


	}

	template<typename GT, typename VN, typename ST>
	KOKKOS_FORCEINLINE_FUNCTION
	void mult_adj_u_halfspinor3_perm(const VGaugeView<GT,VN>& gauge_in,
			const HalfSpinorSiteView<ST>& v_in,
			HalfSpinorSiteView<ST>& v_out, int i)
	{

				for(int row=0; row < 3; ++row) {
					for(int spin = 0; spin < 2; ++spin) {

						ComplexZero(v_out(row,spin));
					}
				}

				for(int col=0; col < 3; ++col) {
				  for(int row=0; row < 3; ++row) {
				    for(int spin=0; spin < 2; ++spin) {
				      ComplexConjMadd(v_out(row,spin), VN::permute3(gauge_in(i,3,col,row)), v_in(col,spin));
				    }
				  }
				}


	}
	template<typename GT, typename VN, typename TGT, typename ST, typename TST>
	void KokkosMVLattice(const KokkosCBFineVGaugeField<GT,VN>& u_in,
			const KokkosCBFineVSpinor<ST,VN,2>& hspinor_in,
			int dir,
			     const KokkosCBFineVSpinor<ST,VN,2>& hspinor_out, int _sites_per_team = 2)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<ST> hspinor_in_view = hspinor_in.GetData();
		GaugeView<GT> u = u_in.GetData();
		HalfSpinorView<ST> hspinor_out_view = hspinor_out.GetData();

		const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {


				// Site local workspace...
				HalfSpinorSiteView<TST> site_in;
				GaugeSiteView<TGT>  gauge_in;

				for(int col=0; col <3; ++col) {
					for(int spin=0; spin < 2; ++spin) {
						Load(site_in(col,spin), hspinor_in_view(i,col,spin));
					}
				}


				HalfSpinorSiteView<TST> site_out;
				if( dir == 0 ) {
					mult_u_halfspinor<GT,VN,TST,0>(gauge_in, site_in, site_out);
				}
				if( dir == 1 ) {
					mult_u_halfspinor<GT,VN,TST,1>(gauge_in, site_in, site_out);
				}
				if( dir == 2 ) {
					mult_u_halfspinor<GT,VN,TST,2>(gauge_in, site_in, site_out);
				}
				if( dir == 3 ) {
					mult_u_halfspinor<GT,VN,TST,3>(gauge_in, site_in, site_out);

				}
				// Write out
				for(int col=0; col < 3; ++col) {
					for(int spin=0; spin < 2; ++spin ) {
						Store(hspinor_out_view(i,col,spin),site_out(col,spin));
					}
				}
		});
		  });
	}



	template<typename GT, typename TGT, typename ST, typename TST>
	void KokkosHVLattice(const KokkosCBFineGaugeField<GT>& u_in,
				  const KokkosCBFineSpinor<ST,2>& hspinor_in,
				  int dir,
			     const KokkosCBFineSpinor<ST,2>& hspinor_out,
			     int _sites_per_team = 2)

	{
		int num_sites = u_in.GetInfo().GetNumCBSites();
		HalfSpinorView<ST> hspinor_in_view = hspinor_in.GetData();
		HalfSpinorView<ST> hspinor_out_view = hspinor_out.GetData();
		GaugeView<GT> u = u_in.GetData();

		const MG::ThreadExecPolicy  policy(num_sites/_sites_per_team,Kokkos::AUTO(),Veclen<ST>::value);
		Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle&  team) {
		    const int start_idx = team.league_rank()*_sites_per_team;
		    const int end_idx = start_idx + _sites_per_team  < num_sites ? start_idx + _sites_per_team : num_sites;
		    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,start_idx,end_idx),[=](const int i) {


			// Site local workspace...
			HalfSpinorSiteView<TST> site_in;
			for(int col=0; col <3; ++col) {
				for(int spin=0; spin < 2; ++spin) {
					Load(site_in(col,spin), hspinor_in_view(i,col,spin));
				}
			}

			GaugeSiteView<TGT> gauge_in;
			for(int col=0; col <3; ++col) {
				for(int col2=0; col2 < 3; ++col2) {
				  Load(gauge_in(col,col2), u(i,dir,col,col2));
				}
			}

			HalfSpinorSiteView<TST> site_out;
			mult_adj_u_halfspinor<GT,TST>(gauge_in,site_in, site_out);

			// Write out
			for(int col=0; col < 3; ++col) {
				for(int spin=0; spin < 2; ++spin ) {
					Store(hspinor_out_view(i,col,spin), site_out(col,spin));
				}
			}
		});
	});
	}

}


#endif /* TEST_KOKKOS_MATVEC_H_ */
