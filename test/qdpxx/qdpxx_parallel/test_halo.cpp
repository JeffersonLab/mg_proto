#include "gtest/gtest.h"
#include "../../test_env.h"
#include "../qdpxx_utils.h"
#include "lattice/constants.h"
#include "lattice/lattice_info.h"
#include "lattice/nodeinfo.h"

#include "utils/print_utils.h"
#include "lattice/geometry_utils.h"
#include "lattice/fine_qdpxx/qdpxx_helpers.h"
#include <vector>
#include <random>


#include "lattice/halo.h"
#include "lattice/coarse/coarse_op.h"
using namespace MG; 
using namespace MGTesting;


TEST(TestHalo, TestDiracOpFacePack)
{
	// Check the Halo is initialized properly in a coarse Dirac Op
	IndexArray latdims={{4,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,2,node);
	initQDPXXLattice(latdims);

	int ncol=3;	
	CoarseSpinor spinor(info);
	const IndexArray& cb_latdims = info.GetCBLatticeDimensions();
	const IndexArray& origin = info.GetLatticeOrigin();

	// Fill the spinor with the GLOBAL coordinates:
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
			for(int col=0; col < ncol; ++col) {
				float* spinor_data = spinor.GetSiteDataPtr(cb,cbsite,col);
				IndexArray coords;
				CBIndexToCoords(cbsite,cb,latdims,origin,coords);
				spinor_data[RE + n_complex*0]=(float)coords[0] +origin[0];
				spinor_data[RE + n_complex*1]=(float)coords[1] +origin[1];
				spinor_data[RE + n_complex*2]=(float)coords[2] +origin[2];
				spinor_data[RE + n_complex*3]=(float)coords[3] +origin[3];
			}
		}
	}

	CoarseDiracOp d_op(info);

	for(int cb=0; cb < n_checkerboard; ++cb) {

#pragma omp parallel
		{

			// Face packers use OMP for internally
			for(int mu=0; mu < n_dim; ++mu) {
				if(! d_op.GetSpinorHalo().LocalDir( mu )) {
					packFace<CoarseSpinor,CoarseAccessor>(d_op.GetSpinorHalo(),spinor,cb,mu,MG_BACKWARD);
					packFace<CoarseSpinor,CoarseAccessor>(d_op.GetSpinorHalo(),spinor,cb,mu,MG_FORWARD);
				}
			}

		}
		const SpinorHaloCB& halo = d_op.GetSpinorHalo();
		int site_offset =  n_complex*info.GetNumColorSpins();

		int local_cb = (cb + info.GetCBOrigin())&1;
		for(int mu=0; mu < n_dim; ++mu) {
			if( ! halo.LocalDir(mu) ) {
			// Check forward face in direction mu
			const float* buffer_back = halo.GetSendToDirBuf(2*mu+MG_BACKWARD);
			const float* buffer_forw = halo.GetSendToDirBuf(2*mu+MG_FORWARD);

			int num_sites = halo.NumSitesInFace(mu);
			for(int sites = 0; sites < num_sites; ++sites) {

				IndexArray back_coord;
				IndexArray forw_coord;
				back_coord[mu] = 0;
				forw_coord[mu] = latdims[mu]-1; // NB: THis is checkerboard independent

				if ( mu == X_DIR ) {

					// Working out the X-Coordinates is special.
					// X-face is 'checkberboarded' in Y as it were
					IndexArray x_cb_dims(cb_latdims); x_cb_dims[Y_DIR]/=2;




					IndexToCoords3(sites,x_cb_dims,mu,back_coord);
					IndexToCoords3(sites,x_cb_dims,mu,forw_coord);

					// We need to find out the true Y_coordinates
					// We use the local cb here, which should take into account if the origin is odd
					int yb = 2*back_coord[Y_DIR]
										 + ((local_cb + back_coord[Z_DIR] + back_coord[T_DIR])&1);

					int yf = 2*forw_coord[Y_DIR]
										  + ((local_cb + forw_coord[X_DIR] + forw_coord[Z_DIR] + forw_coord[T_DIR])&1);


					back_coord[Y_DIR] = yb;
					forw_coord[Y_DIR] = yf;

					// NB: For other directions, X direction will be a checkebroarded index in [0,Nxh-1]
					// However, here in this case we have already fixed it to Nx-1 so no need to do anything
					// for comparing to coordinates.
				}
				else {
					// Y Z and T directions
					IndexToCoords3(sites,cb_latdims,mu,back_coord); // return CB xcoord
					IndexToCoords3(sites,cb_latdims,mu,forw_coord); // return CB xcoord

					// X_coord is checkerboarded here
					// That is fine, but to compare to global coords I need
					// to do a proper conversion
					back_coord[X_DIR] *= 2;
					back_coord[X_DIR] +=  ((local_cb + back_coord[Y_DIR] + back_coord[Z_DIR] + back_coord[T_DIR])&1);
					forw_coord[X_DIR] *= 2;
					forw_coord[X_DIR] +=  ((local_cb + forw_coord[Y_DIR] + forw_coord[Z_DIR] + forw_coord[T_DIR])&1);


				}

				// Add on Origin to make coords global
				for(int dir=0; dir < n_dim; ++dir) {
					back_coord[dir] += origin[dir];
					forw_coord[dir] += origin[dir];
				}

				// Check against buffer.
				IndexArray back_coord_from_buffer;
				back_coord_from_buffer[0] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*0]);
				back_coord_from_buffer[1] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*1]);
				back_coord_from_buffer[2] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*2]);
				back_coord_from_buffer[3] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*3]);

				// Check against buffer
				IndexArray forw_coord_from_buffer;
				forw_coord_from_buffer[0] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*0]);
				forw_coord_from_buffer[1] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*1]);
				forw_coord_from_buffer[2] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*2]);
				forw_coord_from_buffer[3] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*3]);

				// Assert equality
				for(int dir=0; dir < n_dim; ++dir) {
					ASSERT_EQ( back_coord_from_buffer[dir], back_coord[dir]);
					ASSERT_EQ( forw_coord_from_buffer[dir], forw_coord[dir]);
				}


			} // sites


			}// LocalDir

		} // mu

	} //cb

}

TEST(TestHalo, TestDiracOpFaceTransf)
{
	// Check the Halo is initialized properly in a coarse Dirac Op
	IndexArray latdims={{6,4,4,4}};
	NodeInfo node;
	LatticeInfo info(latdims,2,2,node);
	IndexArray gdims;
	info.LocalDimsToGlobalDims(gdims,latdims);
	initQDPXXLattice(latdims);

	int ncol=3;
	CoarseSpinor spinor(info, ncol);
	const IndexArray& cb_latdims = info.GetCBLatticeDimensions();
	const IndexArray& origin = info.GetLatticeOrigin();

	// Fill the spinor with the GLOBAL coordinates:
	for(int cb=0; cb < n_checkerboard; ++cb) {
		for(int cbsite=0; cbsite < info.GetNumCBSites(); ++cbsite) {
			for(int col=0; col < ncol; ++col) {
				float* spinor_data = spinor.GetSiteDataPtr(cb,cbsite,col);
				IndexArray coords;
				CBIndexToCoords(cbsite,cb,latdims,origin,coords);
				spinor_data[RE + n_complex*0]=(float)coords[0] +origin[0];
				spinor_data[RE + n_complex*1]=(float)coords[1] +origin[1];
				spinor_data[RE + n_complex*2]=(float)coords[2] +origin[2];
				spinor_data[RE + n_complex*3]=(float)coords[3] +origin[3];
			}
		}
	}

	CoarseDiracOp d_op(info);
	SpinorHaloCB& halo = d_op.GetSpinorHalo();
	for(int cb=0; cb < n_checkerboard; ++cb) {


#pragma omp parallel
		{
			for(int mu=0; mu < n_dim; ++mu) {
				if(! halo.LocalDir( mu )) {
					packFace<CoarseSpinor,CoarseAccessor>(halo,spinor,cb,mu,MG_BACKWARD);
					packFace<CoarseSpinor,CoarseAccessor>(halo,spinor,cb,mu,MG_FORWARD);
				}
			}
		}

		MasterLog(INFO, "Exchanging Faces with source CB=%d",cb);
			halo.StartAllRecvs();
			halo.StartAllSends();
			halo.FinishAllSends();
			halo.FinishAllRecvs();


		int target_cb = 1-cb;


		int site_offset =  n_complex*info.GetNumColorSpins();

		int local_target_cb = (target_cb + info.GetCBOrigin())&1;


		for(int mu=0; mu < n_dim; ++mu) {
			if( ! halo.LocalDir(mu) ) {
			// Check forward face in direction mu
			const float* buffer_back = halo.GetRecvFromDirBuf(2*mu+MG_BACKWARD);
			const float* buffer_forw = halo.GetRecvFromDirBuf(2*mu+MG_FORWARD);

			int num_sites = halo.NumSitesInFace(mu);
			for(int sites = 0; sites < num_sites; ++sites) {

				IndexArray back_coord;
				IndexArray forw_coord;
				back_coord[mu] = 0;
				forw_coord[mu] = latdims[mu]-1; // NB: THis is checkerboard independent

				if ( mu == X_DIR ) {

					// Working out the X-Coordinates is special.
					// X-face is 'checkberboarded' in Y as it were
					IndexArray x_cb_dims(cb_latdims); x_cb_dims[Y_DIR]/=2;




					IndexToCoords3(sites,x_cb_dims,mu,back_coord);
					IndexToCoords3(sites,x_cb_dims,mu,forw_coord);

					// We need to find out the true Y_coordinates
					// We use the local cb here, which should take into account if the origin is odd
					int yb = 2*back_coord[Y_DIR]
										 + ((local_target_cb + back_coord[Z_DIR] + back_coord[T_DIR])&1);

					int yf = 2*forw_coord[Y_DIR]
										  + ((local_target_cb + forw_coord[X_DIR] + forw_coord[Z_DIR] + forw_coord[T_DIR])&1);


					back_coord[Y_DIR] = yb;
					forw_coord[Y_DIR] = yf;

					// NB: For other directions, X direction will be a checkebroarded index in [0,Nxh-1]
					// However, here in this case we have already fixed it to Nx-1 so no need to do anything
					// for comparing to coordinates.
				}
				else {
					// Y Z and T directions
					IndexToCoords3(sites,cb_latdims,mu,back_coord); // return CB xcoord
					IndexToCoords3(sites,cb_latdims,mu,forw_coord); // return CB xcoord

					// X_coord is checkerboarded here
					// That is fine, but to compare to global coords I need
					// to do a proper conversion
					back_coord[X_DIR] *= 2;
					back_coord[X_DIR] +=  ((local_target_cb + back_coord[Y_DIR] + back_coord[Z_DIR] + back_coord[T_DIR])&1);
					forw_coord[X_DIR] *= 2;
					forw_coord[X_DIR] +=  ((local_target_cb + forw_coord[Y_DIR] + forw_coord[Z_DIR] + forw_coord[T_DIR])&1);


				}

				// Add on Origin to make coords global
				for(int dir=0; dir < n_dim; ++dir) {
					back_coord[dir] += origin[dir];
					forw_coord[dir] += origin[dir];
				}

				// Now offset them to what we expect to find in the buffers



					back_coord[mu]--;
					if( back_coord[mu] < 0) back_coord[mu] = gdims[mu]-1;
					forw_coord[mu]++;
					if( forw_coord[mu] >= gdims[mu] ) forw_coord[mu]=0;

								// Check against buffer.
				IndexArray back_coord_from_buffer;
				back_coord_from_buffer[0] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*0]);
				back_coord_from_buffer[1] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*1]);
				back_coord_from_buffer[2] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*2]);
				back_coord_from_buffer[3] = static_cast<IndexType>(buffer_back[ sites*site_offset + RE + n_complex*3]);

				// Check against buffer
				IndexArray forw_coord_from_buffer;
				forw_coord_from_buffer[0] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*0]);
				forw_coord_from_buffer[1] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*1]);
				forw_coord_from_buffer[2] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*2]);
				forw_coord_from_buffer[3] = static_cast<IndexType>(buffer_forw[ sites*site_offset + RE + n_complex*3]);

#if 0
				MasterLog(INFO, "mu=%d sites=%d Back Coord=(%d %d %d %d) Back Buffer Coord=(%d %d %d %d) Forw Coord=(%d %d %d %d) Forw Buffer Coord=(%d %d %d %d)",
						mu,sites,
						back_coord[X_DIR],back_coord[Y_DIR],back_coord[Z_DIR],back_coord[T_DIR],
						back_coord_from_buffer[X_DIR],back_coord_from_buffer[Y_DIR],back_coord_from_buffer[Z_DIR],
						back_coord_from_buffer[T_DIR],
						forw_coord[X_DIR],forw_coord[Y_DIR],forw_coord[Z_DIR],forw_coord[T_DIR],
						forw_coord_from_buffer[X_DIR],forw_coord_from_buffer[Y_DIR],forw_coord_from_buffer[Z_DIR],
						forw_coord_from_buffer[T_DIR]);

#endif
				// Assert equality
				for(int dir=0; dir < n_dim; ++dir) {
					ASSERT_EQ( back_coord_from_buffer[dir], back_coord[dir]);
					ASSERT_EQ( forw_coord_from_buffer[dir], forw_coord[dir]);
				}


			} // sites


			}// LocalDir

		} // mu

	} //cb

}


int main(int argc, char *argv[]) 
{
	return MGTesting::TestMain(&argc, argv);
}

