/*
 * transf.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: bjoo
 */

#include "lattice/fine_qdpxx/transf.h"

namespace MG {

    void FermToProp(const LatticeFermion &a, LatticePropagator &b, int color_index,
                    int spin_index) {
        for (int j = 0; j < Ns; ++j) {
            LatticeColorMatrix bb = peekSpin(b, j, spin_index);
            LatticeColorVector aa = peekSpin(a, j);

            for (int i = 0; i < Nc; ++i) pokeColor(bb, peekColor(aa, i), i, color_index);

            pokeSpin(b, bb, j, spin_index);
        }
    }

    void PropToFerm(const LatticePropagator &b, LatticeFermion &a, int color_index,
                    int spin_index) {
        for (int j = 0; j < Ns; ++j) {
            LatticeColorMatrix bb = peekSpin(b, j, spin_index);
            LatticeColorVector aa = peekSpin(a, j);

            for (int i = 0; i < Nc; ++i) pokeColor(aa, peekColor(bb, i, color_index), i);

            pokeSpin(a, aa, j);
        }
    }
}
