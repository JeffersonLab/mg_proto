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


	virtual
	void M_eo(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) {
		MasterLog(ERROR, "M_eo not yet implemented.");
	}

	virtual
	void M_oe(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) {
		MasterLog(ERROR, "M_oe not yet implemented.");
	}

        virtual
        void M_ee(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) {
                MasterLog(ERROR, "M_ee not yet implemented.");
        }

        virtual
        void M_oo(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) {
                MasterLog(ERROR, "M_oo not yet implemented.");
        }


	virtual
	void M_ee_inv(Spinor& out, const Spinor& in, IndexType type = LINOP_OP ) {
		MasterLog(ERROR, "M_ee not yet implemented.");
	}


	virtual ~LinearOperator() {}

	virtual int GetLevel(void) const =  0;
	virtual const LatticeInfo& GetInfo() const = 0;

private:
};


} // Namespace



#endif /* INCLUDE_LATTICE_LINEAR_OPERATOR_H_ */
