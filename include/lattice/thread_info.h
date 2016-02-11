/*
 * thread_info.h
 *
 *  Created on: Jan 4, 2016
 *      Author: bjoo
 */

#ifndef INCLUDE_LATTICE_THREAD_INFO_H_
#define INCLUDE_LATTICE_THREAD_INFO_H_

namespace MG {

	struct ThreadInfo {

		int n_mv_par;    // Number of threads per MatVec
		int n_dir_par;   // Number of parallel groups processing the directions
		int n_site_par;  // Number of parallel groups processing the sites

		/* 3d thread IDs  mv_par_id fasetst, then dir, then site i.e.
		 * mv_par_id + n_mv_par*(dir_par_id + n_dir_par*site_par_id) = tid
		 */
		int mv_par_id;
		int dir_par_id;
		int site_par_id;

		/* Computed loop bounds */
		int min_vrow; 	// Minimum vrow for thread
		int max_vrow;   // Maximum vrow for thread

		int min_dir;    // Minimum direction for thread
		int max_dir;    // Maximum direction for thread

		int min_site;   // Minimum site for thread
		int max_site;   // Maximum site for thread

	};
}



#endif /* INCLUDE_LATTICE_THREAD_INFO_H_ */
