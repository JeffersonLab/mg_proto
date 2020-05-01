#ifndef  INCLUDE_LATTICE_AUXILIARY_H
#define  INCLUDE_LATTICE_AUXILIARY_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <lattice/lattice_info.h>

namespace MG {
	namespace aux {
		template<typename T> std::vector<T> sqrt(const std::vector<T>& v) {
			std::vector<T> out(v.size());
			std::transform(v.begin(), v.end(), out.begin(), [](const T& x){ return std::sqrt(x); });
			return out;
		}

		template<typename T> std::vector<T> operator/(const std::vector<T>& a, const std::vector<T>& b) {
			assert(a.size() == b.size());
			std::vector<T> out(a.size());
			std::transform(a.begin(), a.end(), b.begin(), out.begin(), [](const T& x, const T& y){ return x/y; });
			return out;
		}
	}

	template<typename Spinor>
	class AbstractSpinor {
	public:
		// virtual bool is_like(const Spinor& s) const = 0;
		// virtual bool is_like(const LatticeInfo& info, int ncol) const = 0;
	};

	template<typename Spinor>
	class AuxiliarySpinors {
	public:
		AuxiliarySpinors(const AuxiliarySpinors<Spinor>* subrogate_=nullptr) : subrogate(subrogate_) {}

		// Return a spinor with a shape like the given one
		std::shared_ptr<Spinor> tmp(const LatticeInfo& info, int ncol) const {
			if (subrogate) return subrogate->tmp(info, ncol);

			std::shared_ptr<Spinor> s;

			// Find a spinor not being used
			for (auto it=_tmp.begin(); it != _tmp.end(); it++) {
				if (it->use_count() <= 1) {
					// If the free spinor is not like the given
					// one, replace it with a new one.
					if (! it->get()->is_like(info, ncol)) {
						it->reset(new Spinor(info, ncol));
					}
					s = *it;
					break;
				}
			}

			// If every spinor is busy, create a new one
			if (!s) {
				s.reset(new Spinor(info, ncol));
				_tmp.emplace_back(s);
			}
			return s;
		}

		// Return a spinor with a shape like the given one
		std::shared_ptr<Spinor> tmp(const Spinor& like_this) const {
			return tmp(like_this.GetInfo(), like_this.GetNCol());
		}
	private:
		mutable std::vector<std::shared_ptr<Spinor>> _tmp;
		const AuxiliarySpinors<Spinor>* subrogate;
	};
}

#endif
