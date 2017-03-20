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

private:
};


} // Namespace



#endif /* INCLUDE_LATTICE_LINEAR_OPERATOR_H_ */
