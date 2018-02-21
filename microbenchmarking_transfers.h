#include <iostream>

void microbenchmarkingPCI(int gpu, float *LoHTD, float *LoDTH, float *GHTD, float *overlappedGHTD, 
					   float *GDTH, float *overlappedGDTH, int nIter);
float getLoHTD(char *d_data, char *h_data, int nreps);
float getLoDTH(char *d_data, char *h_data, int nreps);
float getGDTH(char *d_data, char *h_data, float LoDTH, int nreps);
float getGHTD(char *d_data, char *h_data, float LoHTD, int nreps);
float getOverlappedGDTH(char *d_data, char *h_data, float LoDTH, int nreps);
float getOverlappedGHTD(char *d_data, char *h_data, float LoHTD, int nreps);