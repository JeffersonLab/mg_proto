/*
 * kokkos_spinproj.h
 *
 *  Created on: May 26, 2017
 *      Author: bjoo
 */

#ifndef TEST_KOKKOS_KOKKOS_SPINPROJ_H_
#define TEST_KOKKOS_KOKKOS_SPINPROJ_H_

#include "kokkos_types.h"

namespace MG {

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosProjectDir0(const Kokkos::View<T[4][3][2]>& spinor_in,
						Kokkos::View<T[2][3][2]>& spinor_out, const T sign)
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
		spinor_out(0,color,RE) = spinor_in(0,color,RE)-sign*spinor_in(3,color,IM);
		spinor_out(0,color,IM) = spinor_in(0,color,IM)+sign*spinor_in(3,color,RE);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(1,color,RE) = spinor_in(1,color,RE)-sign*spinor_in(2,color,IM);
		spinor_out(1,color,IM) = spinor_in(1,color,IM)+sign*spinor_in(2,color,RE);
	}
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosProjectDir1(const Kokkos::View<T[4][3][2]>& spinor_in,
						Kokkos::View<T[2][3][2]>& spinor_out, const T sign)
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
		spinor_out(0,color,RE) = spinor_in(0,color,RE)-sign*spinor_in(3,color,RE);
		spinor_out(0,color,IM) = spinor_in(0,color,IM)-sign*spinor_in(3,color,IM);
	}
	for(int color=0; color < 3; ++color) {
		spinor_out(1,color,RE) = spinor_in(1,color,RE)+sign*spinor_in(2,color,RE);
		spinor_out(1,color,IM) = spinor_in(1,color,IM)+sign*spinor_in(2,color,IM);
	}
}


template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosProjectDir2(const Kokkos::View<T[4][3][2]>& spinor_in,
						Kokkos::View<T[2][3][2]>& spinor_out, const T sign)
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
		spinor_out(0,color,RE) = spinor_in(0,color,RE)-sign*spinor_in(2,color,IM);
		spinor_out(0,color,IM) = spinor_in(0,color,IM)+sign*spinor_in(2,color,RE);
	}

	for(int color=0; color < 3; ++color ) {
 		spinor_out(1,color,RE) = spinor_in(1,color,RE)+sign*spinor_in(3,color,IM);
		spinor_out(1,color,IM) = spinor_in(1,color,IM)-sign*spinor_in(3,color,RE);
	}
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosProjectDir3(const Kokkos::View<T[4][3][2]>& spinor_in,
						Kokkos::View<T[2][3][2]>& spinor_out, const T sign)
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
		spinor_out(0,color,RE) = spinor_in(0,color,RE)+sign*spinor_in(2,color,RE);
		spinor_out(0,color,IM) = spinor_in(0,color,IM)+sign*spinor_in(2,color,IM);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(1,color,RE) = spinor_in(1,color,RE)+sign*spinor_in(3,color,RE);
		spinor_out(1,color,IM) = spinor_in(1,color,IM)+sign*spinor_in(3,color,IM);
	}
}


template<typename T, const int dir>
KOKKOS_INLINE_FUNCTION
void KokkosProjectDir( 	const Kokkos::View<T[4][3][2]>& kokkos_in,
		int plus_minus,
						Kokkos::View<T[2][3][2]>& kokkos_hspinor_out)
{
	T sign = static_cast<T>( plus_minus == 1 ? 1 : -1 );
	if( dir == 0 ) {
		KokkosProjectDir0(kokkos_in,kokkos_hspinor_out,sign);
	}
	else if ( dir == 1 ) {
		KokkosProjectDir1(kokkos_in,kokkos_hspinor_out,sign);
	}
	else if ( dir == 2 ) {
		KokkosProjectDir2(kokkos_in,kokkos_hspinor_out,sign);
	}
	else if ( dir == 3 ) {
		KokkosProjectDir3(kokkos_in,kokkos_hspinor_out,sign);
	}
	else {
		MasterLog(ERROR, "Direction %d not implemented", dir);
	}
}

template<typename T>
void KokkosProject(const KokkosCBFineSpinor<T,4>& kokkos_in,
				   int dir,  int plus_minus,
				   KokkosCBFineSpinor<T,2>& kokkos_hspinor_out)
{
	int num_sites = kokkos_in.GetInfo().GetNumCBSites();
	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {
				const Kokkos::View<T[4][3][2]>& site_in = Kokkos::subview(kokkos_in.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
				Kokkos::View<T[2][3][2]> site_out = Kokkos::subview(kokkos_hspinor_out.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				// Thread local result
				Kokkos::View<T[2][3][2]> res("res_spin_proj");
				if( dir == 0) {
					KokkosProjectDir<T,0>(site_in,plus_minus,res);
				}
				else if (dir == 1) {
					KokkosProjectDir<T,1>(site_in,plus_minus,res);
				}
				else if (dir == 2 ) {
					KokkosProjectDir<T,2>(site_in,plus_minus, res);
				}
				else {
					KokkosProjectDir<T,3>(site_in,plus_minus,res);
				}
				// Hopefully res is registerized/in cache, write it out
				for(int spin=0; spin < 2; spin++) {
					for(int color=0; color < 3; ++color) {
						for(int reim=0; reim < 2; ++reim) {
							site_out(spin,color,reim) = res(spin,color,reim);
						}
					}
				}

	});


}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosRecons23Dir0(const Kokkos::View<T[2][3][2]>& hspinor_in,
						Kokkos::View<T[4][3][2]>& spinor_out,
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
		spinor_out(2,color,RE) = sign*hspinor_in(1,color,IM);
		spinor_out(2,color,IM) = -sign*hspinor_in(1,color,RE);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(3,color,RE) = sign*hspinor_in(0,color,IM);
		spinor_out(3,color,IM) = -sign*hspinor_in(0,color,RE);
	}
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosRecons23Dir1(const Kokkos::View<T[2][3][2]>& hspinor_in,
						Kokkos::View<T[4][3][2]>& spinor_out,
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
		spinor_out(2,color,RE) = sign*hspinor_in(1,color,RE);
		spinor_out(2,color,IM) = sign*hspinor_in(1,color,IM);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(3,color,RE) = -sign*hspinor_in(0,color,RE);
		spinor_out(3,color,IM) = -sign*hspinor_in(0,color,IM);
	}
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosRecons23Dir2(const Kokkos::View<T[2][3][2]>& hspinor_in,
						Kokkos::View<T[4][3][2]>& spinor_out,
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
		spinor_out(2,color,RE) = sign*hspinor_in(0,color,IM);
		spinor_out(2,color,IM) = -sign*hspinor_in(0,color,RE);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(3,color,RE) = -sign*hspinor_in(1,color,IM);
		spinor_out(3,color,IM) = sign*hspinor_in(1,color,RE);
	}
}

template<typename T>
KOKKOS_INLINE_FUNCTION
void KokkosRecons23Dir3(const Kokkos::View<T[2][3][2]>& hspinor_in,
						Kokkos::View<T[4][3][2]>& spinor_out,
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
		spinor_out(2,color,RE) = sign*hspinor_in(0,color,RE);
		spinor_out(2,color,IM) = sign*hspinor_in(0,color,IM);
	}

	for(int color=0; color < 3; ++color) {
		spinor_out(3,color,RE) = sign*hspinor_in(1,color,RE);
		spinor_out(3,color,IM) = sign*hspinor_in(1,color,IM);
	}
}

template<typename T, const int dir, const bool accum=false>
KOKKOS_INLINE_FUNCTION
void  KokkosReconsDir(const Kokkos::View<T[2][3][2]>& hspinor_in,
			int plus_minus,
			Kokkos::View<T[4][3][2]>& spinor_out)
{
	T sign = static_cast<T>( plus_minus == 1 ? 1 : -1 );

	Kokkos::View<T[4][3][2]> res("res_recons_dir");

	// Keep the first two spin components
	for(int spin=0; spin < 2; ++spin ) {
		for(int color=0; color < 3; ++color) {
			for(int reim=0; reim < 2; ++reim) {
				res(spin,color,reim) = hspinor_in(spin,color,reim);
			}
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
	else if (dir == 3 ) {
		KokkosRecons23Dir3(hspinor_in, res, sign);
	}
	else {
		MasterLog(ERROR, "Unknown dir in reconsDir: %d", dir);
	}


	// Reconstruct/Accumulate
	if( accum == false) {
		// Set the result from hopefully registerized 'res'
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
				for(int reim=0; reim < 2 ; ++ reim ) {
					spinor_out(spin,color,reim) = res(spin,color,reim);
				}
			}
		}
	}
	else {
		// Accumulate result
		for(int spin=0; spin < 4; ++spin) {
			for(int color=0; color < 3; ++color) {
				for(int reim=0; reim < 2 ; ++ reim ) {
					spinor_out(spin,color,reim) += res(spin,color,reim);
				}
			}
		}
	}
}

template<typename T>
void KokkosRecons(const KokkosCBFineSpinor<T,2>& kokkos_hspinor_in,
				   int dir, const int plus_minus,
				   KokkosCBFineSpinor<T,4>& kokkos_spinor_out)
{
	const int num_sites = kokkos_hspinor_in.GetInfo().GetNumCBSites();

	Kokkos::parallel_for(num_sites,
			KOKKOS_LAMBDA(int i) {

			    // Input Half Spinor
				const Kokkos::View<T[2][3][2]> site_in = Kokkos::subview(kokkos_hspinor_in.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				// Output Half Spinor
				Kokkos::View<T[4][3][2]> site_out = Kokkos::subview(kokkos_spinor_out.GetData(),i,Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

				// Do the do!
				if (dir == 0 ) {
					KokkosReconsDir<T,0,false>(site_in,plus_minus,site_out);
				}
				else if (dir == 1 ) {
					KokkosReconsDir<T,1,false>(site_in,plus_minus,site_out);
				}
				else if ( dir == 2 ) {
					KokkosReconsDir<T,2,false>(site_in,plus_minus,site_out);
				}
				else {
					KokkosReconsDir<T,3,false>(site_in,plus_minus,site_out);
				}

			});


}



}



#endif /* TEST_KOKKOS_KOKKOS_SPINPROJ_H_ */
