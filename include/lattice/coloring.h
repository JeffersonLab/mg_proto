// -*- C++ -*-                                                                   
/*! \file                                                                    
 * \brief Greedy coloring for Lattice probing
 *                                                                             
 */

#ifndef INCLUDE_LATTICE_COLORING_H_
#define INCLUDE_LATTICE_COLORING_H_

#include "lattice/lattice_info.h"
#include "lattice/coarse/subset.h"
#include "lattice/lattice_info.h"
#include <array>
#include <vector>
#include <memory>
#include <cassert>

namespace MG {

	/** Computes k-distance coloring for toroidal lattices.
	 *
	 *  A k-distance coloring assigns a label (color) to each lattice's node so that
	 *  all neighbors up to distance k of a node have different label than the node.
	 *  The spin-color components are not considered on the distance, but every spin-
	 *  color component on a node has a distinct color to the rest of spin-color
	 *  components on the node.
	 *
	 *  The spin-color colors are the labels considering the spin-color components on
	 *  each node. And, the node colors are the labels considering the nodes only.
	 *  Each spin-color color has a node color associated.
	 *
	 *  The code uses greedy coloring to compute the k-distance coloring with a small
	 *  number of distinct colors.
	 */

	class Coloring {
		public:
			/** Computes the k-distance coloring for a lattice of the given size
			 *  \param info: lattice info
			 *  \param distance: k-distance
			 *
			 *  Compute a k-distance coloring so that the spin-color components
			 *  of a node have different color than spin-color components from
			 *  other nodes up to distance k.
			 */

			Coloring(const std::shared_ptr<LatticeInfo> info, unsigned int distance);

			/** Return probing vectors starting from the given color and spin-color component
			 *  \param out: returned probing vectors
			 *  \param spincolorcolor0: the spin-color color index of the first probing vector
			 */

			template<class Spinor> void GetProbingVectors(Spinor& out, unsigned int spincolorcolor0) const {
				IndexType num_cbsites = _info->GetNumCBSites();
				IndexType num_colorspin = _info->GetNumColorSpins();
				IndexType num_color = _info->GetNumColors();
				IndexType num_spin = _info->GetNumSpins();
				CBSubset subset = SUBSET_ALL;

				assert(out.GetNCol() + spincolorcolor0 <= _num_colors * num_colorspin);
				IndexType ncol = out.GetNCol();

				ZeroVec(out);
#pragma omp parallel for collapse(3) schedule(static)
				for (unsigned int col=0; col < ncol; ++col) {
					for(int cb=subset.start; cb < subset.end; ++cb ) {
						for(int cbsite = 0; cbsite < num_cbsites; ++cbsite) {
							unsigned int color = (spincolorcolor0 + col) / num_colorspin;
							unsigned int spincolor = (spincolorcolor0 + col) % num_colorspin;
							unsigned int sp_color = spincolor / num_spin;
							unsigned int sp_spin = spincolor % num_spin;

							if (color != _local_colors[cb][cbsite]) continue;
							out(col,cb,cbsite,sp_spin,sp_color,0) = 1.0;
						}
					}
				}
			}

			/** Decompose a spin-color color index
			 *  \param sp_color: spin-color color index
			 *  \param node_color: (out) the node color
			 *  \param col_spin: (out) the spin index
			 *  \param col_color: (out) the color index
			 */

			void SpinColorColorComponents(unsigned int sp_color, IndexType& node_color, IndexType& col_spin, IndexType& col_color) const {
				node_color = sp_color / _info->GetNumColorSpins();
				unsigned int colorspin = sp_color % _info->GetNumColorSpins();
				col_color = colorspin / _info->GetNumSpins();
				col_spin = colorspin % _info->GetNumSpins();
			}

			/** Get spin-color color index from the color components
			 *  \param node_color: the node color
			 *  \param col_spin: the spin index
			 *  \param col_color: the color index
			 */

			unsigned int GetSpinColorColor(IndexType node_color, IndexType col_spin, IndexType col_color) const {
				return node_color * _info->GetNumColorSpins() + col_color * _info->GetNumSpins() + col_spin;
			}

			// Get color of a site
			unsigned int GetColorCBIndex(IndexType cb, IndexType cbsite) const {
				return _local_colors[cb][cbsite];
			}

			/** Return the number of node colors
			 */

			unsigned int GetNumNodeColors() const { return _num_colors; }

			/** Return the number of spin-color colors
			 */

			unsigned int GetNumSpinColorColors() const { return _num_colors * _info->GetNumColorSpins(); }

		private:
			std::array<std::vector<unsigned int>,2> _local_colors;
			unsigned int _num_colors;
			const std::shared_ptr<LatticeInfo> _info;
	};

}

#endif // INCLUDE_LATTICE_COLORING_H_
