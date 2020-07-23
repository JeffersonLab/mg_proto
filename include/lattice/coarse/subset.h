/*
 * subset.h
 *
 *  Created on: Aug 7, 2018
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_COARSE_SUBSET_H_
#define INCLUDE_LATTICE_COARSE_SUBSET_H_

#include "lattice/constants.h"
namespace MG
{
	struct CBSubset {
		IndexType start;
		IndexType end;

		bool operator==(const CBSubset& t) const {
			return ((empty() && t.empty()) || (start == t.start && end == t.end));
		}

		bool empty() const {
			return start == end;
		}

		CBSubset complementary() const {
			if (empty()) return CBSubset{0,2};
			return ((*this) == CBSubset{0,1} ? CBSubset{1,2} : CBSubset{0,1});
		}
	};

	constexpr CBSubset SUBSET_EVEN = {0,1};
	constexpr CBSubset SUBSET_ODD  = {1,2};
	constexpr CBSubset SUBSET_ALL  = {0,2};

	constexpr CBSubset RB[2] = {SUBSET_EVEN, SUBSET_ODD};
}


#endif /* INCLUDE_LATTICE_COARSE_SUBSET_H_ */
