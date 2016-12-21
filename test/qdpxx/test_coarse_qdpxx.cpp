#include "gtest/gtest.h"
#include "aggregate_qdpxx.h"
#include "../test_env.h"
#include "../mock_nodeinfo.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "qdpxx_helpers.h"
#include "reunit.h"
#include "transf.h"
#include "clover_fermact_params_w.h"
#include "clover_term_qdp_w.h"
#include "lattice/geometry_utils.h"
#include "lattice/coarse/coarse_types.h"
#include "lattice/coarse/coarse_op.h"
#include "lattice/coarse/coarse_l1_blas.h"
#include "dslashm_w.h"
#include <complex>

using namespace MG;
using namespace MGTesting;
using namespace QDP;

class Block {
public:
	// I am doing it this way so we can make a std::vector of these...
	// That  means it needs an argument free constructor

	Block(void) {
		_created = false;
		_num_sites = 0;
	}

	void create(const IndexArray local_lattice_dimensions,
				const IndexArray block_origin,
				const IndexArray block_dimensions)
	{

		// Check the lock is feasible: origin must be nonnegative
		// origin + extent must not be greater than local_lattice size
		for(int mu=0; mu < n_dim; ++mu) {
			assert( block_origin[mu] >= 0 );
			assert( block_origin[mu] + block_dimensions[mu] <= local_lattice_dimensions[mu]);
		}

		_origin = block_origin;
		_dimensions = block_dimensions;
		_num_sites = block_dimensions[X_DIR]*block_dimensions[Y_DIR]*block_dimensions[Z_DIR]*block_dimensions[T_DIR];
		// Resize the site list
		_site_list.resize(_num_sites);

		// Loop through the blocks sites and build up a map mapping from block index to
		// Site index. We can use this for accessing the block sites in the global lattice.

		// We can do it this way, or flatten by hand. That is a later level optimization
#pragma omp parallel for collapse(4) shared(_site_list)
		for(IndexType t=0; t < _dimensions[T_DIR]; ++t) {
			for(IndexType z=0; z < _dimensions[Z_DIR]; ++z) {
				for(IndexType y=0; y < _dimensions[Y_DIR]; ++y) {
					for(IndexType x=0; x < _dimensions[X_DIR]; ++x) {


						IndexArray block_coords = {{x,y,z,t}};

						// Offset the block coords by the block origin in the local lattice
						IndexArray local_lattice_coords = {{ x + _origin[X_DIR], y + _origin[Y_DIR], z + _origin[T_DIR], t + _origin[Z_DIR] }};

						IndexType block_idx = CoordsToIndex( block_coords, _dimensions);
						IndexType lattice_idx = CoordsToIndex( local_lattice_coords, local_lattice_dimensions) ;

						_site_list[ block_idx ] = lattice_idx;

					} // x
				} // y

			} // z
		}// t

		_created = true; // Mark it as done

	}

	const std::vector<IndexType>& getSiteList(void) const {
		return _site_list;
	}

	const bool isCreated(void) {
		return _created;
	}

	unsigned int getNumSites() const {
		return _num_sites;
	}

	// Destructor is automatic
	~Block() {}
private:
	IndexArray _origin;
	IndexArray _dimensions;
	unsigned int _num_sites;
	bool _created = false;
	std::vector<IndexType> _site_list;
};


// Create a list of blocks
void CreateBlockList(std::vector<Block>& blocklist, IndexArray& blocked_lattice_dimensions, const IndexArray& local_lattice_dimensions, const IndexArray& block_dimensions )
{
	// Compute the dimensions of the blocked lattice. Check local lattice is divisible by block size
	for(int mu=0; mu < n_dim; mu++) {
		blocked_lattice_dimensions[mu] = local_lattice_dimensions[mu] / block_dimensions[mu];
		if( blocked_lattice_dimensions[mu] % block_dimensions[mu] != 0 ) {
			MasterLog(ERROR,"CreateBlockList: block_dimensions[%d]=%d does not divide local_lattice_dimensions[%d]=%d",
						mu,block_dimensions[mu], mu, local_lattice_dimensions[mu]);
		}
	}

	// Compute the number of blocks
	unsigned int num_blocks = 1;
	for(IndexType mu=0; mu < n_dim; ++mu ) num_blocks *= blocked_lattice_dimensions[mu];
	unsigned int num_cb_blocks = num_blocks/2;

	// Now create the blocks. Loop through the coordinates of the blocked lattice.
	// For each one work out the block origin
	blocklist.resize(num_blocks);
	for(int t = 0; t < blocked_lattice_dimensions[T_DIR]; ++t ) {
		for(int z=0; z < blocked_lattice_dimensions[Z_DIR]; ++z ) {
			for(int y=0; y < blocked_lattice_dimensions[Y_DIR]; ++y) {
				for(int x=0; x < blocked_lattice_dimensions[X_DIR]; ++x) {
					IndexArray block_coords = {{x,y,z,t}};
					IndexArray block_origin = {{x*block_dimensions[X_DIR],
												y*block_dimensions[Y_DIR],
												z*block_dimensions[Z_DIR],
												t*block_dimensions[T_DIR] }};


				//	int block_index = CoordsToIndex(block_coords, blocked_lattice_dimensions );
					int block_cb = (x+y+z+t) & 1;

					IndexArray block_cbcoords = {{x/2,y,z,t}};
					IndexArray block_cbdims = {{ blocked_lattice_dimensions[X_DIR]/2,
											     blocked_lattice_dimensions[Y_DIR],
												 blocked_lattice_dimensions[Z_DIR],
												 blocked_lattice_dimensions[T_DIR] }};

					IndexType block_cbsite = CoordsToIndex(block_cbcoords,block_cbdims);

					IndexType block_index = block_cbsite + block_cb*(num_cb_blocks);
					blocklist[block_index].create(local_lattice_dimensions, block_origin, block_dimensions);
				}
			}
		}
	}

}

//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggrQDPXX(const double alpha, LatticeFermion& v, const Block& block, int aggr);

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y,  const Block& block, int aggr);

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggrQDPXX(const LatticeFermion& v, const Block& block, int aggr);

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, const Block& block, int aggr);

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, const Block& block, int aggr );

//! Orthonormalize vecs over the spin aggregates within the sites
void orthonormalizeBlockAggregatesQDPXX(multi1d<LatticeFermion>& vecs, const std::vector<Block>& block_list);



// NEED TO WORK FROM HERE ON.


//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinorQDPXXFineToCoarse( const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v, const LatticeFermion& ferm_in, CoarseSpinor& out);

//! 'Prolongate' a CoarseSpinor to a QDP++ Fine Spinor
void prolongateSpinorCoarseToQDPXXFine(const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v, const CoarseSpinor& coarse_in, LatticeFermion& fine_out);

//! Coarsen one direction of a 'dslash' link
void dslashTripleProductDirQDPXX(const std::vector<Block>& blocklist, int dir, const multi1d<LatticeColorMatrix>& u, const multi1d<LatticeFermion>& in_vecs, CoarseGauge& u_coarse);

//! Coarsen the clover term (1 block = 1 site )
void clovTripleProductQDPXX(const std::vector<Block>& blocklist, const QDPCloverTerm& clov,const multi1d<LatticeFermion>& in_vecs, CoarseClover& cl_coarse);



// Implementation -- where possible call the site versions
//! v *= alpha (alpha is real) over and aggregate in a block, v is a QDP++ Lattice Fermion
void axBlockAggrQDPXX(const double alpha, LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {

		axAggrQDPXX(alpha,v,block_sitelist[site],aggr);
	}
}

//! y += alpha * x (alpha is complex) over aggregate in a block, x, y are QDP++ LatticeFermions;
void caxpyBlockAggrQDPXX(const std::complex<double>& alpha, const LatticeFermion& x, LatticeFermion& y,  const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		// Map to sites...
		caxpyAggrQDPXX(alpha,x,y,block_sitelist[site],aggr);
	}

}

//! return || v ||^2 over an aggregate in a block, v is a QDP++ LatticeFermion
double norm2BlockAggrQDPXX(const LatticeFermion& v, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();
	double block_sum=0;


#pragma omp parallel for reduction(+:block_sum)
	for(int site=0; site < num_sites; ++site) {
		// Map to sites...
		block_sum += norm2AggrQDPXX(v,block_sitelist[site],aggr);
	}

	return block_sum;
}

//! return < left | right > = sum left^\dagger_i * right_i for an aggregate, over a block
std::complex<double>
innerProductBlockAggrQDPXX(const LatticeFermion& left, const LatticeFermion& right, const Block& block, int aggr)
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();
	double real_part=0;
	double imag_part=0;

#pragma omp parallel for reduction(+:real_part) reduction(+:imag_part)
	for(int site=0; site < num_sites; ++site) {
		std::complex<double> site_prod=innerProductAggrQDPXX(left,right,block_sitelist[site],aggr);
		real_part += real(site_prod);
		imag_part += imag(site_prod);
	}

	std::complex<double> ret_val(real_part,imag_part);
	return ret_val;
}

//! Extract the spins belonging to a given aggregate from QDP++ source vector src, into QDP++ target vector target
void extractAggregateQDPXX(LatticeFermion& target, const LatticeFermion& src, const Block& block, int aggr )
{
	auto block_sitelist = block.getSiteList();
	int num_sites = block.getNumSites();

#pragma omp parallel for
	for(int site=0; site < num_sites; ++site) {
		int qdpsite = block_sitelist[site];
		for(int spin=aggr*Ns/2; spin < (aggr+1)*Ns/2; ++spin) {
			for(int color=0; color < 3; ++color) {
				target.elem(qdpsite).elem(spin).elem(color).real() = src.elem(qdpsite).elem(spin).elem(color).real();
				target.elem(qdpsite).elem(spin).elem(color).imag() = src.elem(qdpsite).elem(spin).elem(color).imag();
			}
		}
	}
}

//! Orthonormalize vecs over the spin aggregates within the sites
void orthonormalizeBlockAggregatesQDPXX(multi1d<LatticeFermion>& vecs, const std::vector<Block>& block_list)
{
	int num_blocks = block_list.size();

	for(int aggr=0; aggr < 2; ++aggr) {

		for(int block_id=0; block_id < num_blocks; block_id++) {

			const Block& block = block_list[block_id];

			MasterLog(DEBUG, "Orthonormalizing Aggregate: %d on Block: %d\n",aggr, block_id);

			// This will be over blocks...
			// do vecs[0] ... vecs[N]
			for(int curr_vec=0; curr_vec < vecs.size(); curr_vec++) {

				// orthogonalize against previous vectors
				// if curr_vec == 0 this will be skipped
				for(int prev_vec=0; prev_vec < curr_vec; prev_vec++) {

					std::complex<double> iprod = innerProductBlockAggrQDPXX( vecs[prev_vec], vecs[curr_vec], block, aggr);
					std::complex<double> minus_iprod=std::complex<double>(-real(iprod), -imag(iprod) );

					// curr_vec <- curr_vec - <curr_vec|prev_vec>*prev_vec = -iprod*prev_vec + curr_vec
					caxpyBlockAggrQDPXX( minus_iprod, vecs[prev_vec], vecs[curr_vec], block, aggr);
				}

				// Normalize current vector
				double inv_norm = ((double)1)/sqrt(norm2BlockAggrQDPXX(vecs[curr_vec], block, aggr));

				// vecs[curr_vec] = inv_norm * vecs[curr_vec]
				axBlockAggrQDPXX(inv_norm, vecs[curr_vec], block, aggr);
			}


		}	// block
	}// aggregates
}

//! 'Restrict' a QDP++ spinor to a CoarseSpinor with the same geometry
void restrictSpinorQDPXXFineToCoarse( const std::vector<Block>& blocklist, const multi1d<LatticeFermion>& v,
		const LatticeFermion& ferm_in, CoarseSpinor& out)
{

	int num_coarse_cbsites = out.GetInfo().GetNumCBSites();
	int num_coarse_color = out.GetNumColor();

	// Sanity check. The number of sites in the coarse spinor
	// Has to equal the number of blocks
	assert( n_checkerboard*num_coarse_cbsites == blocklist.size() );

	// The number of vectors has to eaqual the number of coarse colors
	assert( v.size() == num_coarse_color );

	// This will be a loop over blocks
	for(int block_cb=0; block_cb < n_checkerboard; ++block_cb) {
		for(int block_cbsite = 0; block_cbsite < num_coarse_cbsites; ++block_cbsite) {
			IndexType block_idx = block_cbsite + block_cb*num_coarse_cbsites;

			// Identify the current block
			const Block& block = blocklist[block_idx];


			// The coarse site spinor is where we will write the result
			float* coarse_site_spinor = out.GetSiteDataPtr(block_cb,block_cbsite);


			// Get the list of fine sites in the blocks
			auto block_sitelist = block.getSiteList();

			// and their number -- this is redundant, I could get it from block_sitelist.size()
			auto num_sites_in_block = block.getNumSites();


			// Zero the accumulation in the current site
			for(int chiral = 0; chiral < 2; ++chiral ) {
				for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {
					int coarse_colorspin = coarse_color + chiral * num_coarse_color;
					coarse_site_spinor[ RE + n_complex*coarse_colorspin ] = 0;
					coarse_site_spinor[ IM + n_complex*coarse_colorspin ] = 0;
				}
			}

			// Our loop is over coarse_colors and chiralities -- to fill out the colorspin components
			// However, each colorspin component will involve a site loop, and we can compute the contributions
			// to both chiralities of the color component from a vector in a single loop. So I put the fine site
			// loop outside of the chirality loop.
			//
			// An optimization/stabilization will be to  accumulate these loops in double since they are
			// over potentially large number of sites (e.g. 4^4)

			// Remember that coarse color picks the vector so have this outermost now,
			// Since we will be working in a vector at a time
			for(int coarse_color=0; coarse_color  < num_coarse_color; coarse_color++) {

				// Now aggregate over all the sites in the block -- this will be over a single vector...
				// NB: The loop indices may be later rerolled, e.g. if we can restrict multiple vectors at once
				// Then having the coarse_color loop inner will be better.
				for( IndexType fine_site_idx = 0; fine_site_idx < num_sites_in_block; fine_site_idx++ ) {

					// Find the fine site
					int fine_site = block_sitelist[fine_site_idx];

					// Now loop over the chiral components. These are local in a site at the level of spin
					for(int chiral = 0; chiral < 2; ++chiral ) {

						// Identify the color spin component we are accumulating
						int coarse_colorspin = coarse_color + chiral * num_coarse_color;

						// Aggregate the spins for the site.
						for(int spin=0; spin < Ns/2; ++spin ) {
							for(int color=0; color < Nc; ++color ) {
								int targ_spin = spin + chiral*(Ns/2); // Offset by whether upper/lower

								REAL left_r = v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).real();
								REAL left_i = v[ coarse_color ].elem(fine_site).elem(targ_spin).elem(color).imag();

								REAL right_r = ferm_in.elem(fine_site).elem(targ_spin).elem(color).real();
								REAL right_i = ferm_in.elem(fine_site).elem(targ_spin).elem(color).imag();

								// It is V_j^H  ferm_in so conj(left)*right.
								coarse_site_spinor[ RE + n_complex*coarse_colorspin ] += left_r * right_r + left_i * right_i;
								coarse_site_spinor[ IM + n_complex*coarse_colorspin ] += left_r * right_i - right_r * left_i;

							} // color
						}	 // spin aggregates
					} // chiral component
				} // fine site_idx
			} // coarse_color
		} // block_cbsite
	} // block_cb
}


TEST(TestBlocking, TestBlockConstruction )
{
	Block b;
	ASSERT_FALSE( b.isCreated()  );
	ASSERT_EQ( b.getNumSites() , 0);
	auto uninitialized_site_list = b.getSiteList();
	ASSERT_EQ( uninitialized_site_list.size() , 0 );
	ASSERT_TRUE( uninitialized_site_list.empty() );
}

TEST(TestBlocking, TestBlockCreateTrivialSingleBlock )
{
	Block b;
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray local_origin = {{0,0,0,0}};
	IndexArray local_lattice_dims = {{2,2,2,2}};

	b.create(local_lattice_dims,local_origin, block_dims);

	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );
	auto block_sites = b.getSiteList();
	for( IndexType i=0; i < block_sites.size(); ++i) {
		// This is a trivial example expect block_site[i] to map to lattice site[i];
		ASSERT_EQ(i , block_sites[i] );
	}
}

TEST(TestBlocking, TestBlockCreateSingleBlock )
{
	Block b;
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray local_origin = {{0,0,2,2}};
	IndexArray local_lattice_dims = {{4,4,4,4}};

	b.create(local_lattice_dims,local_origin, block_dims);

	ASSERT_TRUE( b.isCreated() );
	ASSERT_EQ( b.getNumSites() , 16 );
	auto block_sites = b.getSiteList();
	for( int i=0; i < block_sites.size(); ++i) {
		IndexArray block_coords = {{0,0,0,0}};
		IndexToCoords(i,block_dims,block_coords);

		// Now offset these coords by the local origin, to get coords in local lattice
		for(int mu=0; mu < n_dim;++mu ) block_coords[mu] += local_origin[mu];
		IndexType lattice_idx = CoordsToIndex(block_coords, local_lattice_dims);
		ASSERT_EQ( block_sites[i], lattice_idx);
	}
}

TEST(TestBlocking, TestBlockCreateBadOriginDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,3,3}};
		IndexArray local_lattice_dims = {{4,4,4,4}};

	EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims) ,
			::testing::KilledBySignal(SIGABRT), "Assertion failed:*");


}

TEST(TestBlocking, TestBlockCreateBadOriginNegativeDeath )
{

	Block b;
		IndexArray block_dims = {{2,2,2,2}};
		IndexArray local_origin = {{0,0,-1,0}};
		IndexArray local_lattice_dims = {{4,4,4,4}};

		EXPECT_EXIT( b.create(local_lattice_dims,local_origin, block_dims) ,
				::testing::KilledBySignal(SIGABRT), "Assertion failed:*");

}

TEST(TestBlocking, TestCreateBlockList)
{
	using BlockList = std::vector<Block>;

	BlockList my_blocks;
	IndexArray local_lattice_dims = {{4,4,4,4}};
	IndexArray block_dims = {{2,2,2,2}};
	IndexArray blocked_lattice_dims;

	CreateBlockList(my_blocks,blocked_lattice_dims,local_lattice_dims,block_dims);

	ASSERT_EQ( my_blocks.size(), 16);
	ASSERT_EQ( blocked_lattice_dims[0],2);
	ASSERT_EQ( blocked_lattice_dims[1],2);
	ASSERT_EQ( blocked_lattice_dims[2],2);
	ASSERT_EQ( blocked_lattice_dims[3],2);

	for(IndexType block_t=0; block_t < blocked_lattice_dims[T_DIR]; block_t++) {
		for(IndexType block_z=0; block_z < blocked_lattice_dims[Z_DIR]; block_z++) {
			for(IndexType block_y=0; block_y < blocked_lattice_dims[Y_DIR]; block_y++) {
				for(IndexType block_x=0; block_x < blocked_lattice_dims[X_DIR]; block_x++) {

					// Compute block_index in the block_list -- for selecting the block list
					IndexArray block_coords = {{ block_x, block_y, block_z, block_t }};
					IndexType block_idx = CoordsToIndex( block_coords, blocked_lattice_dims);

					// We will explicitly construct this block for comparing with the block-list
					// for that we need the block origin.
					IndexArray block_origin;
					for(int mu=0; mu < n_dim; ++mu) block_origin[mu]=block_coords[mu]*block_dims[mu];

					Block compare; compare.create(local_lattice_dims, block_origin, block_dims);
					Block& from_list = my_blocks[block_idx];

					// Now compare stuff
					ASSERT_TRUE( from_list.isCreated() );
					ASSERT_TRUE( compare.isCreated() );
					ASSERT_EQ( from_list.getNumSites(), compare.getNumSites() );
					auto compare_sitelist = compare.getSiteList();
					auto from_list_sitelist = from_list.getSiteList();
					ASSERT_EQ( compare_sitelist.size(), from_list_sitelist.size() );
					for(int i=0; i < compare_sitelist.size(); ++i) {
						ASSERT_EQ( compare_sitelist[i], from_list_sitelist[i] );
					}
				}
			}
		}
	}
}

TEST(TestCoarseQDPXXBlock, TestBlockOrthogonalize)
{
	IndexArray latdims={{4,4,4,4}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
	}

	// Orthonormalize -- to it twice just to make sure
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	for(int block_idx=0; block_idx < my_blocks.size(); ++block_idx) {

		// Check orthonormality for each block
		const Block& block = my_blocks[block_idx];

		// Check orthonormality separately for each aggregate
		for(int aggr=0; aggr < 2; ++aggr ) {

			// Check normalization:
			for(int curr_vec = 0; curr_vec < 6; ++curr_vec) {


				for(int test_vec = 0; test_vec < 6; ++test_vec ) {

					if( test_vec != curr_vec ) {
						MasterLog(DEBUG, "Checking inner product of pair (%d,%d), block=%d aggr=%d", curr_vec,test_vec, block_idx,aggr);
						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec],block, aggr);
						ASSERT_NEAR( real(iprod), 0, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

					}
					else {

						std::complex<double> iprod = innerProductBlockAggrQDPXX(vecs[test_vec],vecs[curr_vec], block, aggr);
						ASSERT_NEAR( real(iprod), 1, 1.0e-15);
						ASSERT_NEAR( imag(iprod), 0, 1.0e-15);

						MasterLog(DEBUG, "Checking norm2 of vector %d block=%d aggr=%d\n", curr_vec, block_idx,aggr);
						double norm = sqrt(norm2BlockAggrQDPXX(vecs[curr_vec],block,aggr));
						ASSERT_NEAR(norm, 1, 1.0e-15);

					}
				}
			}
		}
	}


}

TEST(TestCoarseQDPXXBlock,TestOrthonormal2)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	multi1d<LatticeFermion> compare_vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
			compare_vecs[vec] = vecs[vec];
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Do the site oritented one...
	orthonormalizeAggregatesQDPXX(compare_vecs);
	orthonormalizeAggregatesQDPXX(compare_vecs);

	for(IndexType vec=0; vec < 6; ++vec) {
		Double ndiff = norm2(compare_vecs[vec] -vecs[vec]);

		QDPIO::cout << "Ndiff["<<vec<<"] = " << ndiff << std::endl;
		ASSERT_DOUBLE_EQ( toDouble(ndiff), 0);
	}
}

TEST(TestCoarseQDPXXBlock, TestRestrictorTrivial)
{
	IndexArray latdims={{2,2,2,2}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{1,1,1,1}}; // Trivial blocking -- can check against site variant

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	// 1) Create the blocklist
	std::vector<Block> my_blocks;
	IndexArray blocked_lattice_dims;
	CreateBlockList(my_blocks,blocked_lattice_dims,latdims,blockdims);

	// 2) Create the test vectors
	multi1d<LatticeFermion> vecs(6);
	multi1d<LatticeFermion> compare_vecs(6);
	for(int vec=0; vec < 6; ++vec) {
			gaussian(vecs[vec]);
			compare_vecs[vec] = vecs[vec];
	}

	// Do the proper block orthogonalize
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);
	orthonormalizeBlockAggregatesQDPXX(vecs, my_blocks);

	// Do the site oritented one...
	orthonormalizeAggregatesQDPXX(compare_vecs);
	orthonormalizeAggregatesQDPXX(compare_vecs);

	for(IndexType vec=0; vec < 6; ++vec) {
		Double ndiff = norm2(compare_vecs[vec] -vecs[vec]);
		QDPIO::cout << "Ndiff["<<vec<<"] = " << ndiff << std::endl;
	}

    LatticeInfo info(blocked_lattice_dims, 2, 6, NodeInfo());
	CoarseSpinor coarse_block(info);
	CoarseSpinor coarse_site(info);
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
			float* coarse_site_cursite = coarse_site.GetSiteDataPtr(cb,cbsite);
			float* coarse_block_cursite = coarse_block.GetSiteDataPtr(cb,cbsite);
			// Loop over the components - contiguous NumColorspin x n_complex
			for(int comp = 0; comp < n_complex*coarse_block.GetNumColorSpin(); ++comp) {
				coarse_site_cursite[comp]=0;
				coarse_block_cursite[comp]=0;
			}
		}
	}

	LatticeFermion fine_in;

	gaussian(fine_in);

	// Restrict -- this should be just like packing
	restrictSpinorQDPXXFineToCoarse(compare_vecs,fine_in,coarse_site);
	restrictSpinorQDPXXFineToCoarse(my_blocks,vecs, fine_in,coarse_block);


	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {

			float* coarse_site_cursite = coarse_site.GetSiteDataPtr(cb,cbsite);
			float* coarse_block_cursite = coarse_block.GetSiteDataPtr(cb,cbsite);
			// Loop over the components - contiguous NumColorspin x n_complex
			for(int comp = 0; comp < n_complex*coarse_block.GetNumColorSpin(); ++comp) {
				QDPIO::cout << "cb="<< cb << " site=" <<  cbsite << " component = " << comp
							<< " coarse_site=" << coarse_site_cursite[comp] << " coarse_block=" << coarse_block_cursite[comp] << std::endl;



			}
		}
	}

}




#if 0
//

//  We want to test:  D_c v_c = ( R D_f P )
//
// This relationship should hold true always, both
// when R and P aggregate over sites or blocks of sites.
// We can use the existing functionality without blocking
// over sites to test functionality, and to test the interface.

TEST(TestCoarseQDPXXBlock, TestFakeCoarseClov)
{
	IndexArray latdims={{4,4,4,4}};   // Fine lattice. Make it 4x4x4x4 so we can block it
	IndexArray blockdims = {{2,2,2,2}};

	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	// Initialize the gauge field
	QDPIO::cout << "Initializing Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}

	// Initialize the Clover Op
	QDPIO::cout << "Initializing The Clover Term" << std::endl;

	CloverFermActParams clparam;
	AnisoParam_t aniso;

	// Aniso prarams
	aniso.anisoP=true;
	aniso.xi_0 = 1.5;
	aniso.nu = 0.95;
	aniso.t_dir = 3;

	// Set up the Clover params
	clparam.anisoParam = aniso;

	// Some mass
	clparam.Mass = Real(0.1);

	// Some random clover coeffs
	clparam.clovCoeffR=Real(1.35);
	clparam.clovCoeffT=Real(0.8);
	QDPCloverTerm clov_qdp;
	clov_qdp.create(u,clparam);

	QDPIO::cout << "Initializing Random Null-Vectors" << std::endl;

	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	QDPIO::cout << "Orthonormalizing Nullvecs" << std::endl;
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);

	QDPIO::cout << "Coarsening Clover to create D_c" << std::endl;
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseClover c_clov(info);
	clovTripleProductSiteQDPXX(clov_qdp, vecs, c_clov);


	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	restrictSpinorQDPXXFineToCoarse(vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	// Now evaluate  D_c v_c
	int n_smt = 1;
	CoarseDiracOp D(info,n_smt);

	// Apply Coarsened Clover
#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		D.CloverApply(out, c_clov, v_c,0,tid);
		D.CloverApply(out, c_clov, v_c,1,tid);
	}

	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	for(int cb=0; cb < 2; ++cb) {
		clov_qdp.apply(D_f_out, P_v_c, 0, cb);
	}

	// Now restrict back:
	restrictSpinorQDPXXFineToCoarse(vecs, D_f_out, fake_out);

	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);
}


TEST(TestCoarseQDPXXBlock, TestFakeCoarseDslash)
{
	IndexArray latdims={{4,4,4,4}};
	initQDPXXLattice(latdims);
	QDPIO::cout << "QDP++ Testcase Initialized" << std::endl;

	multi1d<LatticeColorMatrix> u(Nd);

	QDPIO::cout << "Generating Random Gauge with Gaussian Noise" << std::endl;
	for(int mu=0; mu < Nd; ++mu) {
		gaussian(u[mu]);
		reunit(u[mu]);
	}


	// Random Basis vectors
	multi1d<LatticeFermion> vecs(6);
	for(int k=0; k < 6; ++k) {
		gaussian(vecs[k]);
	}

	// Someone once said doing this twice is good
	orthonormalizeAggregatesQDPXX(vecs);
	orthonormalizeAggregatesQDPXX(vecs);


	// Next step should be to copy this into the fields needed for gauge and clover ops
	LatticeInfo info(latdims, 2, 6, NodeInfo());
	CoarseGauge u_coarse(info);

	// Generate the triple products directly into the u_coarse
	for(int mu=0; mu < 8; ++mu) {
		QDPIO::cout << " Attempting Triple Product in direction: " << mu << std::endl;
		dslashTripleProductSiteDirQDPXX(mu, u, vecs, u_coarse);
	}

	int n_smt = 1;
	CoarseDiracOp D_op_coarse(info, n_smt);

	// Now create a LatticeFermion and apply both the QDP++ and the Coarse Clover
	LatticeFermion v_f;
	gaussian(v_f);

	// Coarsen v_f to R(v_f) give us coarse RHS for tests
	CoarseSpinor v_c(info);
	restrictSpinorQDPXXFineToCoarse(vecs, v_f, v_c);

	// Output
	CoarseSpinor out(info);
	CoarseSpinor fake_out(info);

	// Apply Coarse Op Dslash in Threads
#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		D_op_coarse.Dslash(out, u_coarse, v_c, 0, tid);
		D_op_coarse.Dslash(out, u_coarse, v_c, 1, tid);
	}


	// Now apply the fake operator:
	LatticeFermion P_v_c = zero;
	prolongateSpinorCoarseToQDPXXFine(vecs, v_c, P_v_c); // NB: This is not the same as v_f, but rather P R v_f

	// Now apply the Clover Term to form D_f P
	LatticeFermion D_f_out = zero;

	// Apply Dslash to both CBs, isign=1
	// Result in m_psiu
	for(int cb=0; cb < 2; ++cb) {
		dslash(D_f_out, u, P_v_c, 1, cb);
	}


	// Now restrict back: fake_out = R D_f P  v_c
	restrictSpinorQDPXXFineToCoarse(vecs, D_f_out, fake_out);

	// We should now compare out, with fake_out. For this we need an xmy
	double norm_diff = sqrt(xmyNorm2Coarse(fake_out,out));
	double norm_diff_per_site = norm_diff / (double)fake_out.GetInfo().GetNumSites();

	MasterLog(INFO, "Diff Norm = %16.8e", norm_diff);
	MasterLog(INFO, "Diff Norm per site = %16.8e", norm_diff_per_site);

}
#endif

int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

