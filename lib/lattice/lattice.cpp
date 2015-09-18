#include "lattice/lattice.h"

/** Constructor for lattice class
 * \param lx_   The X-dimension of the lattice
 * \param ly_   The Y-dimension of the lattice
 * \param lz_   The Z-dimension of the lattice
 * \param lt_   The T-dimension of the lattice
 * \param n_spin_ The number of spin components
 * \param n_color The number of color components
 */

Lattice::Lattice(int lx_,
		 int ly_,
		 int lz_,
		 int lt_,
		 int n_spin_,
		 int n_color_) : lx(lx_), ly(ly_), lz(lz_), lt(lt_), n_spin(n_spin_), n_color(n_color_)
{
}

/** Destructor for the lattice class
 */
inline
Lattice::~Lattice() {}
