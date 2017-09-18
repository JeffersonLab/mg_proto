#ifndef TEST_KOKKOS_VTYPE_H
#define TEST_KOKKOS_VTYPE_H

#include <memory>
#include "Kokkos_Core.hpp"
#include "lattice/lattice_info.h"
#include "kokkos_defaults.h"
#include "kokkos_vectype.h"
#include "kokkos_vnode.h"

namespace MG {

 template<typename T, typename VN, int _num_spins>
 class KokkosCBFineVSpinor {
 public:
 KokkosCBFineVSpinor(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
     if( _g_info.GetNumSpins() != _num_spins ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have %d spins in info. Info has %d",
		 _num_spins,_g_info.GetNumSpins());
     }

     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3 };
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VNDims[mu] == 0 ) { 
	 l_orig[mu] /= VNDims[mu];
	 l_dims[mu] /= VNDims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VNDims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumColors(), _g_info.GetNodeInfo());
     
     // Init the data
     _cb_data=DataType("cb_data", _info->GetNumCBSites());
   }


   inline
     const LatticeInfo& GetInfo() const {
     return *(_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   using VecType = SIMDComplex<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = Kokkos::View<VecType*[_num_spins][3],Layout,MemorySpace>;


   const DataType& GetData() const {
     return _cb_data;
   }
   
   DataType& GetData() {
     return _cb_data;
   }

   
 private:
   DataType _cb_data;
   const LatticeInfo& _g_info;
   std::shared_ptr<LatticeInfo> _info;

   const IndexType _cb;
 };

 
 template<typename T, typename VN>
   using VSpinorView =  typename KokkosCBFineVSpinor<T,VN,4>::DataType;

 template<typename T, typename VN>
   using VHalfSpinorView =  typename KokkosCBFineVSpinor<T,VN,2>::DataType;

 template<typename T, typename VN>
 class KokkosCBFineVGaugeField {
 public:
 KokkosCBFineVGaugeField(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     IndexArray VNDims = { VN::Dim0, VN::Dim1, VN::Dim2, VN::Dim3};
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VNDims[mu] == 0 ) { 
	 l_orig[mu] /= VNDims[mu];
	 l_dims[mu] /= VNDims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VNDims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumColors(), _g_info.GetNodeInfo());
     
     // Init the data
     _cb_data=DataType("cb_data", _info->GetNumCBSites());
   }


   inline
     const LatticeInfo& GetInfo() const {
     return *(_info);
   }

   inline 
     const LatticeInfo& GetGlobalInfo() const {
     return _g_info;
   }


   inline
     IndexType GetCB() const {
     return _cb;
   }

   using VecType = SIMDComplex<typename BaseType<T>::Type, VN::VecLen>;
   using DataType = Kokkos::View<VecType*[4][3][3],GaugeLayout,MemorySpace>;


   const DataType& GetData() const {
     return _cb_data;
   }
   
   DataType& GetData() {
     return _cb_data;
   }

   
 private:
   DataType _cb_data;
   const LatticeInfo& _g_info;
   std::shared_ptr<LatticeInfo> _info;
   const IndexType _cb;
 };

 template<typename T,typename VN>
   using VGaugeView = typename KokkosCBFineVGaugeField<T,VN>::DataType;

 template<typename T, typename VN>
   class KokkosFineVGaugeField {
 private:
   const LatticeInfo& _info;
   KokkosCBFineVGaugeField<T,VN>  _gauge_data_even;
   KokkosCBFineVGaugeField<T,VN>  _gauge_data_odd;
 public:
 KokkosFineVGaugeField(const LatticeInfo& info) :  _info(info), _gauge_data_even(info,EVEN), _gauge_data_odd(info,ODD) {
		}

   const KokkosCBFineVGaugeField<T,VN>& operator()(IndexType cb) const
     {
       return  (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
       //return *(_gauge_data[cb]);
     }
   
   KokkosCBFineVGaugeField<T,VN>& operator()(IndexType cb) {
     return (cb == EVEN) ? _gauge_data_even : _gauge_data_odd;
     //return *(_gauge_data[cb]);
   }
 };

}

#endif
