/*
 * linear_operator.h
 *
 *  Created on: Jan 9, 2017
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_LINEAR_OPERATOR_H_
#define INCLUDE_LATTICE_LINEAR_OPERATOR_H_

#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/coarse/subset.h"
#include "utils/print_utils.h"
namespace MG {

/*! Abstract Linear Operator Class */
template<typename Spinor_t, typename Gauge_t>
class LinearOperator {
public:

	// Export your traits...
	using Gauge = Gauge_t;
	using Spinor = Spinor_t;

	virtual
	void operator()(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const  = 0;


	virtual ~LinearOperator() {}

	virtual int GetLevel(void) const =  0;
	virtual const LatticeInfo& GetInfo() const = 0;
	virtual const CBSubset& GetSubset() const = 0;

private:
};

template <typename Spinor_t, typename Gauge_t>
class EOLinearOperator : public LinearOperator<Spinor_t,Gauge_t> {
	public:
		using Gauge = Gauge_t;
		using Spinor = Spinor_t;

		virtual void unprecOp(Spinor& out, const Spinor& in, IndexType type = LINOP_OP) const = 0;
		virtual void leftOp(Spinor& out, const Spinor& in) const = 0;
		virtual void leftInvOp(Spinor& out, const Spinor& in) const = 0;

#if 0
  // Special case when in even is known to be zero.
  virtual void leftInvOpZero(Spinor& out,const Spinor& in) const = 0;
#endif

  	  virtual void rightOp(Spinor& out, const Spinor& in) const = 0;
  	  virtual void rightInvOp(Spinor& out, const Spinor& in) const  = 0;

  	 virtual void M_ee_inv(Spinor& out, const Spinor& in, IndexType type=LINOP_OP) const = 0;
};

} // Namespace



#endif /* INCLUDE_LATTICE_LINEAR_OPERATOR_H_ */
