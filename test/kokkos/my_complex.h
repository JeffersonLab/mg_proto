#ifndef MY_COMPLEX_H
#define MY_COMPLEX_H

namespace MG {
  namespace Balint {
  template<typename T>
    class complex { 
  private: 
    T _re,_im;
  public:
    explicit KOKKOS_INLINE_FUNCTION complex(): _re(0),_im(0) {}

    template<typename T1, typename T2>
      KOKKOS_INLINE_FUNCTION complex(const T1& re, const T2& im) : _re(re),_im(im) {}

    template<typename T1>
    KOKKOS_INLINE_FUNCTION
      complex<T>& operator=(const complex<T1>& src)
      { 
	_re = src._re;
	_im = src._im;
	return *this;
      }

      KOKKOS_INLINE_FUNCTION
	complex( const complex<T>& src ) : _re(src._re), _im(src._im){}

      KOKKOS_INLINE_FUNCTION
	T& real() {
	return _re;
      }

      KOKKOS_INLINE_FUNCTION
	const T& real() const {
	return _re;
      }

      KOKKOS_INLINE_FUNCTION
	T& imag() { 
	return _im;
      }

      KOKKOS_INLINE_FUNCTION
	const T& imag() const {
	return _im;
      }
  
  }; // class complex

  template<>
    class complex<float> : public float2 { 
  public:
    explicit KOKKOS_INLINE_FUNCTION complex<float>() {
      x = 0.;
      y = 0.;
    }

    template<typename T1, typename T2>
    explicit  KOKKOS_INLINE_FUNCTION complex<float>(const T1& re, const T2& im) {
      x = re;
      y = im;
    }

    
    explicit KOKKOS_INLINE_FUNCTION complex<float>(const float& re, const float& im) {
      x = re; y = im;
    }

    template<typename T1>
    KOKKOS_INLINE_FUNCTION
      complex<float>& operator=(const complex<T1>& src)
      {
	x = src.x;
	y = src.y;
	return *this;
      }

       KOKKOS_INLINE_FUNCTION
      complex<float>& operator=(const complex<float>& src)
      {
	x = src.x;
        y = src.y;
	return *this;
      }

      KOKKOS_INLINE_FUNCTION
	complex<float>( const complex<float>& src ) {
	(*this) = src;
      }

      KOKKOS_INLINE_FUNCTION
	float& real() {
	return x;
      }

      KOKKOS_INLINE_FUNCTION
	const float& real() const {
	return x;
      }

      KOKKOS_INLINE_FUNCTION
	float& imag() { 
	return y;
      }

      KOKKOS_INLINE_FUNCTION
	const float& imag() const {
	return y;
      }
  
  }; // class complex



  }; // Namespace Balint
}; // Namespace MG

#endif
