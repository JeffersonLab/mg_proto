/*
 * kokkos_spinproj.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_SPINPROJ_H_
#define TEST_KOKKOS_KOKKOS_SPINPROJ_H_

#include "kokkos_constants.h"
#include "kokkos_types.h"
#include "kokkos_ops.h"
#include "kokkos_traits.h"
namespace MG {


template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir0(const SpinorView<T> in,
		HalfSpinorSiteView<T>& spinor_out, int i)
{
  using FType = typename BaseType<T>::Type;
  constexpr FType sign = static_cast<FType>(isign);

	/*                              ( 1  0  0 -i)  ( a0 )    ( a0 - i a3 )
	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -i  0)  ( a1 )  = ( a1 - i a2 )
	 *                    0         ( 0  i  1  0)  ( a2 )    ( a2 + i a1 )
	 *                              ( i  0  0  1)  ( a3 )    ( a3 + i a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a3i} + i{a0i - a3r} )
	 *      ( b1r + i b1i )     ( {a1r + a2i} + i{a1i - a2r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r - a1i} + i{a2i + a1r} )  =  ( - b1i + i b1r )
	 *      ( b3r + i b3i )     ( {a3r - a0i} + i{a3i + a0r} )     ( - b0i + i b0r )
	 */

	for(int color=0; color < 3; ++color) {
		//		spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_IM);
		//		spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,3,K_RE);
		A_add_sign_iB(spinor_out(color,0), in(i,color,0), sign, in(i,color,3) );

	}

	for(int color=0; color < 3; ++color) {
		//	spinor_out(color,1,K_RE) = in(i,color,1,K_RE)-sign*in(i,color,2,K_IM);
		//	spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_RE);
		A_add_sign_iB(spinor_out(color,1), in(i,color,1), sign, in(i,color,2));
	}
}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir1(const SpinorView<T> in,
		HalfSpinorSiteView<T>& spinor_out, int i)
{
	  using FType = typename BaseType<T>::Type;
	  constexpr FType sign = static_cast<FType>(isign);

	/*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
	 *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
	 *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
	 *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )

	 * Therefore the top components are

	 *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
	 *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
	 */
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,3,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)-sign*in(i,color,3,K_IM);
		A_add_sign_B(spinor_out(color,0),in(i,color,0),-sign,in(i,color,3));
	}
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B(spinor_out(color,1), in(i,color,1),sign,in(i,color,2));
	}
}


 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir2(const SpinorView<T> in,
		HalfSpinorSiteView<T>& spinor_out, int i)
{

  using FType =  typename BaseType<T>::Type;
  constexpr FType sign = static_cast<FType>(isign);

	/*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
	 *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
	 *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
	 *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
	 */

	for(int color=0; color < 3; ++color) {
		//spinor_out(color,0,K_RE) = in(i,color,0,K_RE)-sign*in(i,color,2,K_IM);
		//spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_RE);
		A_add_sign_iB(spinor_out(color,0),in(i,color,0),sign,in(i,color,2));
	}

	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_IM);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)-sign*in(i,color,3,K_RE);
		A_add_sign_iB(spinor_out(color,1), in(i,color,1), -sign,in(i,color,3));
	}
}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir3(const SpinorView<T> in,
		HalfSpinorSiteView<T>& spinor_out,
		int i)
{
	  using FType = typename BaseType<T>::Type;
	  constexpr FType sign = static_cast<FType>(isign);
	/*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	 *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	 *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	 *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	 */
	for(int color=0; color < 3; ++color) {
		// spinor_out(color,0,K_RE) = in(i,color,0,K_RE)+sign*in(i,color,2,K_RE);
		// spinor_out(color,0,K_IM) = in(i,color,0,K_IM)+sign*in(i,color,2,K_IM);
		A_add_sign_B(spinor_out(color,0), in(i,color,0), sign, in(i,color,2));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(color,1,K_RE) = in(i,color,1,K_RE)+sign*in(i,color,3,K_RE);
		// spinor_out(color,1,K_IM) = in(i,color,1,K_IM)+sign*in(i,color,3,K_IM);
		A_add_sign_B(spinor_out(color,1), in(i,color,1), sign, in(i,color,3));
	}
}


 template<typename T, int dir, int isign>
void KokkosProjectLattice(const KokkosCBFineSpinor<T,4>& kokkos_in,
		KokkosCBFineSpinor<T,2>& kokkos_hspinor_out)
{
	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
	const SpinorView<T>& spinor_in = kokkos_in.GetData();
	HalfSpinorView<T>& hspinor_out = kokkos_hspinor_out.GetData();


	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {
		HalfSpinorSiteView<T> res;

		if( dir == 0) {
		  //			KokkosProjectDir<T,0>(spinor_in,plus_minus,res,i);
		  KokkosProjectDir0<T,isign>(spinor_in, res, i);
		}
		else if (dir == 1) {
		  //			KokkosProjectDir<T,1>(spinor_in,plus_minus,res,i);
		  KokkosProjectDir1<T,isign>(spinor_in, res, i);
		}
		else if (dir == 2 ) {
		  KokkosProjectDir2<T,isign>(spinor_in, res,i);
		}
		else {
		  KokkosProjectDir3<T,isign>(spinor_in, res,i);
		}

		for(int color=0; color < 3; ++color) {
		  for(int spin=0; spin<2; ++spin) {

		    //hspinor_out(i,spin,color,reim) = res(spin,color,reim);
		    ComplexCopy(hspinor_out(i,color,spin), res(color,spin));
		  }
		}
		
	});


}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir0(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{

  using FType = typename BaseType<T>::Type;
  constexpr FType sign = static_cast<FType>(isign);
	/*                              ( 1  0  0 +i)  ( a0 )    ( a0 + i a3 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1 +i  0)  ( a1 )  = ( a1 + i a2 )
	 *                    0         ( 0 -i  1  0)  ( a2 )    ( a2 - i a1 )
	 *                              (-i  0  0  1)  ( a3 )    ( a3 - i a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a3i} + i{a0i + a3r} )
	 *      ( b1r + i b1i )     ( {a1r - a2i} + i{a1i + a2r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a1i} + i{a2i - a1r} )  =  ( b1i - i b1r )
	 *      ( b3r + i b3i )     ( {a3r + a0i} + i{a3i - a0r} )     ( b0i - i b0r )
	 */
  for(int color=0; color < 3; ++color) { 
    for(int spin=0; spin < 2; ++spin) { 
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }


	// Spin 2
	for(int color=0; color < 3; ++color ) {
		//	spinor_out(color,2).real() = sign*hspinor_in(color,1).imag();
		//	spinor_out(color,2).imag() = -sign*hspinor_in(color,1).real();
		A_peq_sign_miB(spinor_out(color,2), sign, hspinor_in(color,1));
	}

	for(int color=0; color < 3; ++color) {
		//	spinor_out(color,3).real() = sign*hspinor_in(color,0).imag();
		//	spinor_out(color,3).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB(spinor_out(color,3),sign, hspinor_in(color,0));
	}
}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir1(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
  using FType = typename BaseType<T>::Type;
  constexpr FType sign = static_cast<FType>(isign);
  /*                              ( 1  0  0 -1)  ( a0 )    ( a0 - a3 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  1  0)  ( a1 )  = ( a1 + a2 )
	 *                    1         ( 0  1  1  0)  ( a2 )    ( a2 + a1 )
	 *                              (-1  0  0  1)  ( a3 )    ( a3 - a0 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a3r} + i{a0i - a3i} )
	 *      ( b1r + i b1i )     ( {a1r + a2r} + i{a1i + a2i} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a1r} + i{a2i + a1i} )  =  (   b1r + i b1i )
	 *      ( b3r + i b3i )     ( {a3r - a0r} + i{a3i - a0i} )     ( - b0r - i b0i )
	 */
  for(int color=0; color < 3; ++color) { 
    for(int spin=0; spin < 2; ++spin) { 
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }

	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,1).real();
		// spinor_out(color,2).imag() = sign*hspinor_in(color,1).imag();
		A_peq_sign_B(spinor_out(color,2),sign,hspinor_in(color,1));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3).real() = -sign*hspinor_in(color,0).real();
		// spinor_out(color,3).imag() = -sign*hspinor_in(color,0).imag();
	  A_peq_sign_B(spinor_out(color,3),-sign,hspinor_in(color,0));
	}
}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir2(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
	  using FType = typename BaseType<T>::Type;
		  constexpr FType sign = static_cast<FType>(isign);
	/*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
	 *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
	 *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
	 *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a0i} + i{a2i - a0r} )  =  (   b0i - i b0r )
	 *      ( b3r + i b3i )     ( {a3r - a1i} + i{a3i + a1r} )     ( - b1i + i b1r )
	 */

  for(int color=0; color < 3; ++color) { 
    for(int spin=0; spin < 2; ++spin) { 
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }


	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2).real() = sign*hspinor_in(color,0).imag();
		// spinor_out(color,2).imag() = -sign*hspinor_in(color,0).real();
		A_peq_sign_miB(spinor_out(color,2), sign, hspinor_in(color,0));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = -sign*hspinor_in(color,1,K_IM);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_RE);
		A_peq_sign_miB(spinor_out(color,3), -sign, hspinor_in(color,1));
	}
}

 template<typename T, int isign>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir3(const HalfSpinorSiteView<T>& hspinor_in,
			SpinorSiteView<T>& spinor_out)
{
	  using FType = typename BaseType<T>::Type;
		  constexpr FType sign = static_cast<FType>(isign);

	/*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	 *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	 *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	 *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	 * Therefore the top components are
	 *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	 *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	 * The bottom components of be may be reconstructed using the formula
	 *      ( b2r + i b2i )  =  ( {a2r + a0r} + i{a2i + a0i} )  =  ( b0r + i b0i )
	 *      ( b3r + i b3i )     ( {a3r + a1r} + i{a3i + a1i} )     ( b1r + i b1i )
	 */
  for(int color=0; color < 3; ++color) { 
    for(int spin=0; spin < 2; ++spin) { 
      ComplexPeq(spinor_out(color,spin),hspinor_in(color,spin));
    }
  }

	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(color,2,K_RE) = sign*hspinor_in(color,0,K_RE);
		// spinor_out(color,2,K_IM) = sign*hspinor_in(color,0,K_IM);
		A_peq_sign_B(spinor_out(color,2),sign,hspinor_in(color,0));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(color,3,K_RE) = sign*hspinor_in(color,1,K_RE);
		// spinor_out(color,3,K_IM) = sign*hspinor_in(color,1,K_IM);
		A_peq_sign_B(spinor_out(color,3), sign, hspinor_in(color,1));
	}
}


 template<typename T, int dir, int isign>
void KokkosReconsLattice(const KokkosCBFineSpinor<T,2>& kokkos_hspinor_in,
		KokkosCBFineSpinor<T,4>& kokkos_spinor_out)
{
	const int num_sites = kokkos_hspinor_in.GetInfo().GetNumCBSites();
	SpinorView<T>& spinor_out = kokkos_spinor_out.GetData();
	const HalfSpinorView<T>& hspinor_in_view = kokkos_hspinor_in.GetData();

	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {




		HalfSpinorSiteView<T> hspinor_in;
		SpinorSiteView<T> res;

		// Stream in top 2 components.
		for(int color=0; color < 3; ++color) {
		  for(int spin=0; spin < 2; ++spin ) {
		    ComplexCopy(hspinor_in(color,spin),hspinor_in_view(i,color,spin));
		  }
		}

		for(int color=0; color < 3; ++color) {
		  for(int spin=0; spin < 4; ++spin ) {
		    ComplexZero(res(color,spin));
		  }
		}

		// Reconstruct into a SpinorSiteView
		if (dir == 0 ) {
		  KokkosRecons23Dir0<T,isign>(hspinor_in,
					     res);
		}
		else if (dir == 1 ) {
		  KokkosRecons23Dir1<T,isign>(hspinor_in,
						     res);

		}
		else if ( dir == 2 ) {
		  KokkosRecons23Dir2<T,isign>(hspinor_in,
						     res);
		}
		else {
		  KokkosRecons23Dir3<T,isign>(hspinor_in,
						     res);

		}

		// Stream out into a spinor
		for(int color=0; color < 3; ++color ) {
		  for(int spin=0; spin < 4; ++spin) {

		    ComplexCopy( spinor_out(i,color,spin), res(color,spin));

		  }


		}

	});


}



}



#endif /* TEST_KOKKOS_KOKKOS_SPINPROJ_H_ */
