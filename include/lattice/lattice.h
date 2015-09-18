#ifndef LATTICE_H
#define LATTICE_H

class Lattice {
 public:
  Lattice(int lx_, int ly_, int lz_, int lt_, int n_spin_, int n_color_);
  ~Lattice();

  const int lx;
  const int ly;
  const int lz;
  const int lt;
  const int n_spin;
  const int n_color;

};




#endif
