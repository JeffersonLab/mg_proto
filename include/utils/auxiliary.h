#ifndef  INCLUDE_LATTICE_AUXILIARY_H
#define  INCLUDE_LATTICE_AUXILIARY_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>

namespace MG {
	namespace aux {
		template<typename T> std::vector<T> sqrt(const std::vector<T>& v) {
			std::vector<T> out(v.size());
			std::transform(v.begin(), v.end(), out.begin(), [](const T& x){ return std::sqrt(x); });
			return out;
		}
	}

	template<typename Spinor>
	class AbstractSpinor {
	public:
		virtual bool is_like(const Spinor& s) const = 0;
		virtual Spinor* create_new() const = 0;
	};

	template<typename Spinor>
	class AuxiliarySpinors {
	public:
		// Return a spinor with a shape like the given one
		std::shared_ptr<Spinor> tmp(const Spinor& like_this) const {
			std::shared_ptr<Spinor> s;

			// Find a spinor not being used
			for (auto it=_tmp.begin(); it != _tmp.end(); it++) {
				if (it->use_count() <= 1) {
					// If the free spinor is not like the given
					// one, replace it with a new one.
					if (! it->get()->is_like(like_this)) {
						it->reset(like_this.create_new());
					}
					s = *it;
					break;
				}
			}

			// If every spinor is busy, create a new one
			if (!s) {
				s.reset(like_this.create_new());
				_tmp.emplace_back(s);
			}
			return s;
		}
	private:
		mutable std::vector<std::shared_ptr<Spinor>> _tmp;
	};
}

#endif
