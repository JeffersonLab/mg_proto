#ifndef  INCLUDE_LATTICE_AUXILIARY_H
#define  INCLUDE_LATTICE_AUXILIARY_H

#include <vector>
#include <algorithm>
#include <cmath>

namespace MG {
	namespace {
		template<typename T> std::vector<T> sqrt(const std::vector<T>& v) {
			std::vector<T> out(v.size());
			std::transform(v.begin(), v.end(), out.begin(), [](const T& x){ return std::sqrt(x); });
			return out;
		}
	}
}

#endif
