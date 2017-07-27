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
namespace MG {




template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir0(const SpinorView<Kokkos::complex<T>> in,
						HalfSpinorSiteView<Kokkos::complex<T>>& spinor_out, const T& sign, int i)
{


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
//		spinor_out(0,color,K_RE) = in(i,0,color,K_RE)-sign*in(i,3,color,K_IM);
//		spinor_out(0,color,K_IM) = in(i,0,color,K_IM)+sign*in(i,3,color,K_RE);
		A_add_iB(spinor_out(0,color), in(i,0,color), sign, in(i,3,color) );

	}

	for(int color=0; color < 3; ++color) {
	//	spinor_out(1,color,K_RE) = in(i,1,color,K_RE)-sign*in(i,2,color,K_IM);
	//	spinor_out(1,color,K_IM) = in(i,1,color,K_IM)+sign*in(i,2,color,K_RE);
		A_add_iB(spinor_out(1,color), in(i,1,color), sign, in(i,2,color));
	}
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir1(const SpinorView<Kokkos::complex<T>> in,
						HalfSpinorSiteView<Kokkos::complex<T>>& spinor_out, const T sign, int i)
{

	  /*                              ( 1  0  0  1)  ( a0 )    ( a0 + a3 )
	   *  B  :=  ( 1 - Gamma  ) A  =  ( 0  1 -1  0)  ( a1 )  = ( a1 - a2 )
	   *                    1         ( 0 -1  1  0)  ( a2 )    ( a2 - a1 )
	   *                              ( 1  0  0  1)  ( a3 )    ( a3 + a0 )

	   * Therefore the top components are

	   *      ( b0r + i b0i )  =  ( {a0r + a3r} + i{a0i + a3i} )
	   *      ( b1r + i b1i )     ( {a1r - a2r} + i{a1i - a2i} )
	*/
	for(int color=0; color < 3; ++color) {
		// spinor_out(0,color,K_RE) = in(i,0,color,K_RE)-sign*in(i,3,color,K_RE);
		// spinor_out(0,color,K_IM) = in(i,0,color,K_IM)-sign*in(i,3,color,K_IM);
		A_sub_B(spinor_out(0,color),in(i,0,color),sign,in(i,3,color));
	}
	for(int color=0; color < 3; ++color) {
		// spinor_out(1,color,K_RE) = in(i,1,color,K_RE)+sign*in(i,2,color,K_RE);
		// spinor_out(1,color,K_IM) = in(i,1,color,K_IM)+sign*in(i,2,color,K_IM);
		A_add_B(spinor_out(1,color), in(i,1,color),sign,in(i,2,color));
	}
}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir2(const SpinorView<Kokkos::complex<T>> in,
						HalfSpinorSiteView<Kokkos::complex<T>>& spinor_out, const T sign, int i)
{


	  /*                              ( 1  0  i  0)  ( a0 )    ( a0 + i a2 )
	   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0 -i)  ( a1 )  = ( a1 - i a3 )
	   *                    2         (-i  0  1  0)  ( a2 )    ( a2 - i a0 )
	   *                              ( 0  i  0  1)  ( a3 )    ( a3 + i a1 )
	   * Therefore the top components are
	   *      ( b0r + i b0i )  =  ( {a0r - a2i} + i{a0i + a2r} )
	   *      ( b1r + i b1i )     ( {a1r + a3i} + i{a1i - a3r} )
	   */

	for(int color=0; color < 3; ++color) {
		//spinor_out(0,color,K_RE) = in(i,0,color,K_RE)-sign*in(i,2,color,K_IM);
		//spinor_out(0,color,K_IM) = in(i,0,color,K_IM)+sign*in(i,2,color,K_RE);
		A_add_iB(spinor_out(0,color),in(i,0,color),sign,in(i,2,color));
	}

	for(int color=0; color < 3; ++color ) {
 		// spinor_out(1,color,K_RE) = in(i,1,color,K_RE)+sign*in(i,3,color,K_IM);
		// spinor_out(1,color,K_IM) = in(i,1,color,K_IM)-sign*in(i,3,color,K_RE);
		A_sub_iB(spinor_out(1,color), in(i,1,color), sign,in(i,3,color));
	}
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir3(const SpinorView<Kokkos::complex<T>> in,
						HalfSpinorSiteView<Kokkos::complex<T>>& spinor_out,
						const T sign, int i)
{
	  /*                              ( 1  0  1  0)  ( a0 )    ( a0 + a2 )
	   *  B  :=  ( 1 + Gamma  ) A  =  ( 0  1  0  1)  ( a1 )  = ( a1 + a3 )
	   *                    3         ( 1  0  1  0)  ( a2 )    ( a2 + a0 )
	   *                              ( 0  1  0  1)  ( a3 )    ( a3 + a1 )
	   * Therefore the top components are
	   *      ( b0r + i b0i )  =  ( {a0r + a2r} + i{a0i + a2i} )
	   *      ( b1r + i b1i )     ( {a1r + a3r} + i{a1i + a3i} )
	   */
	for(int color=0; color < 3; ++color) {
		// spinor_out(0,color,K_RE) = in(i,0,color,K_RE)+sign*in(i,2,color,K_RE);
		// spinor_out(0,color,K_IM) = in(i,0,color,K_IM)+sign*in(i,2,color,K_IM);
		A_add_B(spinor_out(0,color), in(i,0,color), sign, in(i,2,color));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(1,color,K_RE) = in(i,1,color,K_RE)+sign*in(i,3,color,K_RE);
		// spinor_out(1,color,K_IM) = in(i,1,color,K_IM)+sign*in(i,3,color,K_IM);
		A_add_B(spinor_out(1,color), in(i,1,color), sign, in(i,3,color));
	}
}


template<typename T, const int dir>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosProjectDir( 	const SpinorView<Kokkos::complex<T>> kokkos_in,
						int plus_minus,
						HalfSpinorSiteView<Kokkos::complex<T>>& kokkos_hspinor_out,int i)
{
	T sign = static_cast<T>( plus_minus == 1 ? 1 : -1 );
	if( dir == 0 ) {
		KokkosProjectDir0(kokkos_in,kokkos_hspinor_out,sign,i);
	}
	else if ( dir == 1 ) {
		KokkosProjectDir1(kokkos_in,kokkos_hspinor_out,sign,i);
	}
	else if ( dir == 2 ) {
		KokkosProjectDir2(kokkos_in,kokkos_hspinor_out,sign,i);
	}
	else {
		KokkosProjectDir3(kokkos_in,kokkos_hspinor_out,sign,i);
	}

}

template<typename T>
void KokkosProjectLattice(const KokkosCBFineSpinor<Kokkos::complex<T>,4>& kokkos_in,
				   int dir,  int plus_minus,
				   KokkosCBFineSpinor<Kokkos::complex<T>,2>& kokkos_hspinor_out)
{
	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
	const SpinorView<Kokkos::complex<T>>& spinor_in = kokkos_in.GetData();
	HalfSpinorView<Kokkos::complex<T>>& hspinor_out = kokkos_hspinor_out.GetData();


	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {
				HalfSpinorSiteView<Kokkos::complex<T>> res;

				if( dir == 0) {
					KokkosProjectDir<T,0>(spinor_in,plus_minus,res,i);
				}
				else if (dir == 1) {
					KokkosProjectDir<T,1>(spinor_in,plus_minus,res,i);
				}
				else if (dir == 2 ) {
					KokkosProjectDir<T,2>(spinor_in,plus_minus, res,i);
				}
				else {
					KokkosProjectDir<T,3>(spinor_in,plus_minus, res,i);
				}
				for(int spin=0; spin<2; ++spin) {
					for(int color=0; color < 3; ++color) {

							//hspinor_out(i,spin,color,reim) = res(spin,color,reim);
							ComplexCopy(hspinor_out(i,spin,color), res(spin,color));
					}
				}

	});


}


template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir0(const HalfSpinorSiteView<Kokkos::complex<T>>& hspinor_in,
						SpinorSiteView<Kokkos::complex<T>>& spinor_out,
						const T sign)
{
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

	// Spin 2
	for(int color=0; color < 3; ++color ) {
	//	spinor_out(2,color).real() = sign*hspinor_in(1,color).imag();
	//	spinor_out(2,color).imag() = -sign*hspinor_in(1,color).real();
		A_eq_miB(spinor_out(2,color), sign, hspinor_in(1,color));
	}

	for(int color=0; color < 3; ++color) {
	//	spinor_out(3,color).real() = sign*hspinor_in(0,color).imag();
	//	spinor_out(3,color).imag() = -sign*hspinor_in(0,color).real();
		A_eq_miB(spinor_out(3,color),sign, hspinor_in(0,color));
	}
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir1(const HalfSpinorSiteView<Kokkos::complex<T>>& hspinor_in,
						SpinorSiteView<Kokkos::complex<T>>& spinor_out,
						const T sign)
{

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
	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(2,color).real() = sign*hspinor_in(1,color).real();
		// spinor_out(2,color).imag() = sign*hspinor_in(1,color).imag();
		A_eq_B(spinor_out(2,color),sign,hspinor_in(1,color));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(3,color).real() = -sign*hspinor_in(0,color).real();
		// spinor_out(3,color).imag() = -sign*hspinor_in(0,color).imag();
		A_eq_mB(spinor_out(3,color),sign,hspinor_in(0,color));
	}
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir2(const HalfSpinorSiteView<Kokkos::complex<T>>& hspinor_in,
						SpinorSiteView<Kokkos::complex<T>>& spinor_out,
						const T sign)
{
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

	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(2,color).real() = sign*hspinor_in(0,color).imag();
		// spinor_out(2,color).imag() = -sign*hspinor_in(0,color).real();
		A_eq_miB(spinor_out(2,color), sign, hspinor_in(0,color));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(3,color,K_RE) = -sign*hspinor_in(1,color,K_IM);
		// spinor_out(3,color,K_IM) = sign*hspinor_in(1,color,K_RE);
		A_eq_iB(spinor_out(3,color), sign, hspinor_in(1,color));
	}
}

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION
void KokkosRecons23Dir3(const HalfSpinorSiteView<Kokkos::complex<T>>& hspinor_in,
						SpinorSiteView<Kokkos::complex<T>>& spinor_out,
						const T sign)
{
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

	// Spin 2
	for(int color=0; color < 3; ++color ) {
		// spinor_out(2,color,K_RE) = sign*hspinor_in(0,color,K_RE);
		// spinor_out(2,color,K_IM) = sign*hspinor_in(0,color,K_IM);
		A_eq_B(spinor_out(2,color),sign,hspinor_in(0,color));
	}

	for(int color=0; color < 3; ++color) {
		// spinor_out(3,color,K_RE) = sign*hspinor_in(1,color,K_RE);
		// spinor_out(3,color,K_IM) = sign*hspinor_in(1,color,K_IM);
		A_eq_B(spinor_out(3,color), sign, hspinor_in(1,color));
	}
}

template<typename T, int dir, bool accum=false>
KOKKOS_FORCEINLINE_FUNCTION
void  KokkosReconsDir(const HalfSpinorSiteView<Kokkos::complex<T>>& hspinor_in,
			int plus_minus,
			SpinorView<Kokkos::complex<T>> spinor_out, int i)
{
	T sign = static_cast<T>( plus_minus );
	SpinorSiteView<Kokkos::complex<T>> res;

 	// Keep the first two spin components -- stream in
	for(int spin=0; spin < 2; ++spin ) {
		for(int color=0; color < 3; ++color) {

			//	res(spin,color,reim) = hspinor_in(spin,color,reim);
			ComplexCopy(res(spin,color),hspinor_in(spin,color));
		}
	}

	// Pick the right reconstruct
	if( dir == 0 ) {
		KokkosRecons23Dir0(hspinor_in, res, sign);
	}
	else if ( dir == 1 ) {
		KokkosRecons23Dir1(hspinor_in, res, sign);
	}
	else if ( dir == 2 ) {
		KokkosRecons23Dir2(hspinor_in, res, sign);
	}
	else {
		KokkosRecons23Dir3(hspinor_in, res, sign);
	}

#if 0
	if( i== 0) {
		for(int spin=0; spin < 4; ++spin) {
					for(int color=0; color < 3; ++color) {
						for(int reim=0; reim < 2 ; ++ reim ) {
							printf("spin=%d color=%d res = (%lf, %lf)\n", spin,color, res(spin,color,0), res(spin,color,1));
						}
					}
				}
	}
#endif
	// stream out
	// Reconstruct/Accumulate
	if( accum == false) {
		// Set the result from hopefully registerized 'res'
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {

					ComplexCopy( spinor_out(i,spin,color), res(spin,color));

			}
		}
	}
	else {
		// Accumulate result
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
					ComplexPeq(spinor_out(i,spin,color) , res(spin,color) );
			}
		}
	}
}

template<typename T>
void KokkosReconsLattice(const KokkosCBFineSpinor<Kokkos::complex<T>,2>& kokkos_hspinor_in,
				   int dir, int plus_minus,
				   KokkosCBFineSpinor<Kokkos::complex<T>,4>& kokkos_spinor_out)
{
	const int num_sites = kokkos_hspinor_in.GetInfo().GetNumCBSites();
	SpinorView<Kokkos::complex<T>>& spinor_out = kokkos_spinor_out.GetData();
	const HalfSpinorView<Kokkos::complex<T>>& hspinor_in_view = kokkos_hspinor_in.GetData();

	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {




				HalfSpinorSiteView<Kokkos::complex<T>> hspinor_in;

				for(int spin=0; spin < 2; ++spin) {
					for(int color=0; color < 3; ++color ) {

					ComplexCopy( hspinor_in(spin,color), hspinor_in_view(i,spin,color));

					}
				}

				// Do the do!
				if (dir == 0 ) {
					KokkosReconsDir<T,0,false>(hspinor_in,
											plus_minus,
											spinor_out,
											i);
				}
				else if (dir == 1 ) {
					KokkosReconsDir<T,1,false>(hspinor_in,
											plus_minus,
											spinor_out,i);
				}
				else if ( dir == 2 ) {
					KokkosReconsDir<T,2,false>(hspinor_in,
											plus_minus,
											spinor_out,i);
				}
				else {
					KokkosReconsDir<T,3,false>(hspinor_in,
															plus_minus,spinor_out,i);
				}

			});


}



}



#endif /* TEST_KOKKOS_KOKKOS_SPINPROJ_H_ */
