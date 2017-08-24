#ifndef TEST_KOKKOS_VTYPE_H
#define TEST_KOKKOS_VTYPE_H

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
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VN::Dims[mu] == 0 ) { 
	 l_orig[mu] /= VN::Dims[mu];
	 l_dims[mu] /= VN::Dims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VN::Dims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumSpins(), _g_info.GetNodeInfo());
     
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
   using DataType = Kokkos::View<VecType*[3][_num_spins],Layout,MemorySpace>;


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
 class KokkosCBFineVGaugeField {
 public:
 KokkosCBFineVGaugeField(const LatticeInfo& info, IndexType cb)
   : _g_info(info), _cb(cb) {

     if( _g_info.GetNumColors() != 3 ) {
       MasterLog(ERROR, "KokkosCBFineSpinor has to have 3 colors in info. Info has %d", _g_info.GetNumColors());
     }
    
     IndexArray l_orig = _g_info.GetLatticeOrigin();
     IndexArray l_dims = _g_info.GetLatticeDimensions();
     for(int mu=0; mu < 4; ++mu ) {
       if( l_dims[mu] % VN::Dims[mu] == 0 ) { 
	 l_orig[mu] /= VN::Dims[mu];
	 l_dims[mu] /= VN::Dims[mu];
       }
       else{
	 MasterLog(ERROR, "Local dimension %d (=%d) not divisible by VNode::dims[%d]=%d",
		   mu, l_dims[mu], mu, VN::Dims[mu]);
       }
     }
     
     _info=std::make_shared<LatticeInfo>(l_orig,l_dims,_g_info.GetNumSpins(), _g_info.GetNumSpins(), _g_info.GetNodeInfo());
     
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



}

#endif
