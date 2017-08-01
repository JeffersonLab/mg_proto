#include "gtest/gtest.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "../qdpxx/qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/fine_qdpxx/dslashm_w.h"
#include "./kokkos_types.h"
#include "./kokkos_qdp_utils.h"
#include "./kokkos_spinproj.h"
#include "./kokkos_matvec.h"
#include "./kokkos_dslash.h"
#include "./kokkos_vectype.h"

using namespace MG;
using namespace MGTesting;
using namespace QDP;

TEST(TestKokkos, Initialization)
{
	IndexArray latdims={{8,8,8,8}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;


}

TEST(TestVectype, VectypeCreateD4)
{
	SIMDComplex<double,4> v2;
}

TEST(TestVectype, VectypeLenD4)
{
	SIMDComplex<double,4> v4;
	ASSERT_EQ( v4.len(), 4);
}

TEST(TestVectype, TestLaneAccessorsD4)
{
	SIMDComplex<double,4> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,Kokkos::complex<double>(i,-i));

	for(int i=0; i < v4.len(); ++i) {
		double re = v4(i).real();
		double im = v4(i).imag();
		ASSERT_DOUBLE_EQ( re, static_cast<double>(i) );
		ASSERT_DOUBLE_EQ( im, static_cast<double>(-i) );

	}
}

using ThreadExecPolicy =  Kokkos::TeamPolicy<>;
using TeamHandle =  ThreadExecPolicy::member_type;


TEST(TestVectype, VectypeCopyD4)
{

	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,Kokkos::complex<double>(i,-i));

	SIMDComplex<double,4> v4_copy;

	ComplexCopy(v4_copy,v4);
		
	
	for(int i=0; i < v4.len(); ++i) {
		ASSERT_DOUBLE_EQ(  v4_copy(i).real(), v4(i).real());
		ASSERT_DOUBLE_EQ(  v4_copy(i).imag(), v4(i).imag());
	}

	      });
	  });

}

TEST(TestVectype, VectypeLoadStore)
{
	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,Kokkos::complex<double>(i,-i));

	SIMDComplex<double,4> v4_3;
	{
		SIMDComplex<double,4> v4_2;

		Store(v4_2, v4);
		Load(v4_3, v4_2);
	}
	for(int i=0; i < v4.len(); ++i) {
		ASSERT_DOUBLE_EQ(  v4(i).real(), v4_3(i).real());
		ASSERT_DOUBLE_EQ(  v4(i).imag(), v4_3(i).imag());
	}
	      });
	  });

}

TEST(TestVectype, VectypeZeroD4)
{
	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<double,4> v4;
	ComplexZero(v4);
	for(int i=0; i < v4.len(); ++i) {
		ASSERT_DOUBLE_EQ(  v4(i).real(),0);
		ASSERT_DOUBLE_EQ(  v4(i).imag(),0);
	}
	      });
	  });
}



TEST(TestVectype, VectypeComplexPeqD4 )
{

	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4a,v4b,v4c;
	for(int i=0; i < v4a.len(); ++i) {
		v4a.set(i,Kokkos::complex<double>(i,-i));
		v4b.set(i,Kokkos::complex<double>(1.4*i,-0.3*i));
		v4c.set(i, v4b(i));
	}

	ComplexPeq(v4b, v4a);

	for(int i=0; i < v4c.len(); ++i) {
		Kokkos::complex<double> result = v4c(i);
		ComplexPeq(result, v4a(i));
		ASSERT_DOUBLE_EQ( result.real(), v4b(i).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4b(i).imag());
	}
	      });
	  });

}

TEST(TestVectype, VectypeComplexCMaddSalarD4 )
{

	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	Kokkos::complex<double> a=Kokkos::complex<double>(-2.3,1.2);
	SIMDComplex<double,4> v4b,v4c,v4d;
	for(int l=0; l < v4b.len(); ++l) {
		v4b.set(l,Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l,Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l,v4c(l));
	}

	ComplexCMadd(v4c, a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);
		ComplexCMadd( result, a, v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}

	      });
	  });

}


TEST(TestVectype, VectypeComplexConjMaddSalarD4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	Kokkos::complex<double> a=Kokkos::complex<double>(-2.3,1.2);
	SIMDComplex<double,4> v4b,v4c,v4d;
	for(int l=0; l < v4b.len(); ++l) {
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexConjMadd(v4c, a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);
		ComplexConjMadd( result, a, v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}

	});
    });
}



TEST(TestVectype, VectypeComplexCMaddD4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexCMadd(v4c, v4a, v4b);


	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);
		ComplexCMadd( result, v4a(l), v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}

	});
    });
}

TEST(TestVectype, VectypeComplexConjMaddD4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexConjMadd(v4c, v4a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);
		ComplexConjMadd( result, v4a(l), v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}
	});
    });

}

TEST(TestVectype, Test_A_add_sign_B_D4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}
	double sign = -1;

	A_add_sign_B(v4c, v4a, sign, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);

		A_add_sign_B( result, v4a(l), sign, v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}
	});
    });

}

TEST(TestVectype, Test_A_add_sign_iB_D4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<double>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}
	double sign = -1.0;

	A_add_sign_iB(v4c, v4a, sign, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4d(l);
		A_add_sign_iB( result, v4a(l), sign, v4b(l));
		ASSERT_DOUBLE_EQ( result.real(), v4c(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4c(l).imag());
	}

	});
    });

}

TEST(TestVectype, Test_A_peq_sign_miB_D4 )
{
	ThreadExecPolicy policy(1,1,4);
	Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
	    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, v4b(l));
	}
	double sign = -1.0;

	A_peq_sign_miB(v4b, sign, v4a);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4c(l);
		A_peq_sign_miB( result, sign, v4a(l));
		ASSERT_DOUBLE_EQ( result.real(), v4b(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4b(l).imag());
	}
	      });
	  });
}

TEST(TestVectype, Test_A_peq_sign_B_D4 )
{
  ThreadExecPolicy policy(1,1,4);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<double,4> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<double>(l,-l));
		v4b.set(l, Kokkos::complex<double>(1.4*l,-0.3*l));
		v4c.set(l, v4b(l));
	}
	double sign = -1.0;

	A_peq_sign_B(v4b, sign, v4a);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<double> result = v4c(l);

		A_peq_sign_B( result, sign, v4a(l));
		ASSERT_DOUBLE_EQ( result.real(), v4b(l).real());
		ASSERT_DOUBLE_EQ( result.imag(), v4b(l).imag());
	}

	});
    });

}


/* --- FLOAT 8 tests --- */

TEST(TestVectype, VectypeCreateF8)
{
	SIMDComplex<float,8> v2;
}

TEST(TestVectype, VectypeLenF8)
{
	SIMDComplex<float,8> v4;
	ASSERT_EQ( v4.len(), 8);
}

TEST(TestVectype, TestLaneAccessorsF8)
{
	SIMDComplex<float,8> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,Kokkos::complex<float>(i,-i));

	for(int i=0; i < v4.len(); ++i) {
		float re = v4(i).real();
		float im = v4(i).imag();
		ASSERT_FLOAT_EQ( re, static_cast<float>(i) );
		ASSERT_FLOAT_EQ( im, static_cast<float>(-i) );

	}
}
TEST(TestVectype, VectypeCopyF8)
{

  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4;
	for(int i=0; i < v4.len(); ++i)
		v4.set(i,Kokkos::complex<float>(i,-i));

	SIMDComplex<float,8> v4_copy;
	ComplexCopy(v4_copy,v4);
	for(int i=0; i < v4.len(); ++i) {
		ASSERT_FLOAT_EQ(  v4_copy(i).real(), v4(i).real());
		ASSERT_FLOAT_EQ(  v4_copy(i).imag(), v4(i).imag());
	}
	});
    });
}

TEST(TestVectype, VectypeZeroF8)
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4;
	ComplexZero(v4);
	for(int i=0; i < v4.len(); ++i) {
		ASSERT_FLOAT_EQ(  v4(i).real(),0);
		ASSERT_FLOAT_EQ(  v4(i).imag(),0);
	}
	});
    });
}



TEST(TestVectype, VectypeComplexPeqF8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<float,8> v4a,v4b,v4c;
	for(int i=0; i < v4a.len(); ++i) {
		v4a.set(i,Kokkos::complex<float>(i,-i));
		v4b.set(i,Kokkos::complex<float>(1.4*i,-0.3*i));
		v4c.set(i, v4b(i));
	}

	ComplexPeq(v4b, v4a);

	for(int i=0; i < v4c.len(); ++i) {
		Kokkos::complex<float> result = v4c(i);
		ComplexPeq(result, v4a(i));
		ASSERT_FLOAT_EQ( result.real(), v4b(i).real());
		ASSERT_FLOAT_EQ( result.imag(), v4b(i).imag());
	}
	});
    });
}

TEST(TestVectype, VectypeComplexCMaddSalarF8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	Kokkos::complex<float> a=Kokkos::complex<float>(-2.3,1.2);
	SIMDComplex<float,8> v4b,v4c,v4d;
	for(int l=0; l < v4b.len(); ++l) {
		v4b.set(l,Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l,Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l,v4c(l));
	}

	ComplexCMadd(v4c, a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);
		ComplexCMadd( result, a, v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });
}


TEST(TestVectype, VectypeComplexConjMaddSalarF8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	Kokkos::complex<float> a=Kokkos::complex<float>(-2.3,1.2);
	SIMDComplex<float,8> v4b,v4c,v4d;
	for(int l=0; l < v4b.len(); ++l) {
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexConjMadd(v4c, a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);
		ComplexConjMadd( result, a, v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });
}



TEST(TestVectype, VectypeComplexCMaddF8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexCMadd(v4c, v4a, v4b);


	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);
		ComplexCMadd( result, v4a(l), v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });
}

TEST(TestVectype, VectypeComplexConjMaddF8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}

	ComplexConjMadd(v4c, v4a, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);
		ComplexConjMadd( result, v4a(l), v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });
}

TEST(TestVectype, Test_A_add_sign_B_F8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {


	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}
	float sign = -1;

	A_add_sign_B(v4c, v4a, sign, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);

		A_add_sign_B( result, v4a(l), sign, v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });

}

TEST(TestVectype, Test_A_add_sign_iB_F8 )
{

  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, Kokkos::complex<float>(0.1*l,0.5*l));
		v4d.set(l, v4c(l));
	}
	float sign = -1.0;

	A_add_sign_iB(v4c, v4a, sign, v4b);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4d(l);
		A_add_sign_iB( result, v4a(l), sign, v4b(l));
		ASSERT_FLOAT_EQ( result.real(), v4c(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4c(l).imag());
	}
	});
    });
}

TEST(TestVectype, Test_A_peq_sign_miB_F8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, v4b(l));
	}
	float sign = -1.0;

	A_peq_sign_miB(v4b, sign, v4a);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4c(l);
		A_peq_sign_miB( result, sign, v4a(l));
		ASSERT_FLOAT_EQ( result.real(), v4b(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4b(l).imag());
	}
	});
    });
}

TEST(TestVectype, Test_A_peq_sign_B_F8 )
{
  ThreadExecPolicy policy(1,1,8);
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const TeamHandle& team) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,1),[=](const int site) {

	SIMDComplex<float,8> v4a,v4b,v4c,v4d;
	for(int l=0; l < v4a.len(); ++l) {
		v4a.set(l, Kokkos::complex<float>(l,-l));
		v4b.set(l, Kokkos::complex<float>(1.4*l,-0.3*l));
		v4c.set(l, v4b(l));
	}
	float sign = -1.0;

	A_peq_sign_B(v4b, sign, v4a);

	for(int l=0; l < v4d.len(); ++l) {
		Kokkos::complex<float> result = v4c(l);

		A_peq_sign_B( result, sign, v4a(l));
		ASSERT_FLOAT_EQ( result.real(), v4b(l).real());
		ASSERT_FLOAT_EQ( result.imag(), v4b(l).imag());
	}
	});
    });
}

int main(int argc, char *argv[]) 
{
	return ::MGTesting::TestMain(&argc, argv);
}

