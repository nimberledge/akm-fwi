#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  unsigned long * size;
  unsigned long * npsize;
  unsigned long * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;


int Kernel(struct dataobj *restrict b_vec, struct dataobj *restrict damp_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, const float s, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const float dt, const float h_x, const float h_y, const float o_x, const float o_y, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x_size, const int y_size, struct profiler * timers)
{
  float *r8_vec;
  posix_memalign((void**)(&r8_vec),64,(x_size + 1)*(y_size + 1)*sizeof(float));
  float *r9_vec;
  posix_memalign((void**)(&r9_vec),64,(x_size + 1)*(y_size + 1)*sizeof(float));

  float (*restrict b)[b_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[b_vec->size[1]]) b_vec->data;
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict r8)[y_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1]) r8_vec;
  float (*restrict r9)[y_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1]) r9_vec;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r4 = 1.0F/h_x;
  float r5 = 1.0F/h_y;
  float r6 = 1.0F/(dt*dt);
  float r7 = 1.0F/dt;

  for (int time = time_m, t0 = (time)%(491), t1 = (time + 490)%(491), t2 = (time + 1)%(491); time <= time_M; time += 1, t0 = (time)%(491), t1 = (time + 490)%(491), t2 = (time + 1)%(491))
  {
    /* Begin section0 */
    START_TIMER(section0)
    for (int x = x_m - 1; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(b,u,vp:32)
      for (int y = y_m - 1; y <= y_M; y += 1)
      {
        float r16 = -u[t0][x + 2][y + 3];
        float r15 = -u[t0][x + 3][y + 3];
        float r14 = -u[t0][x + 3][y + 2];
        float r13 = -u[t0][x + 4][y + 2];
        float r12 = -u[t0][x + 2][y + 4];
        float r11 = r5*r12 + r5*u[t0][x + 2][y + 5];
        float r10 = r4*r13 + r4*u[t0][x + 5][y + 2];
        r8[x + 1][y + 1] = ((r4*(vp[x + 4][y + 2]*vp[x + 4][y + 2])*(r4*(-r10*b[x + 4][y + 2]) + r4*(r4*(-u[t0][x + 5][y + 2]) + r4*u[t0][x + 6][y + 2])*b[x + 5][y + 2] + r5*(-(r5*r13 + r5*u[t0][x + 4][y + 3])*b[x + 4][y + 2]) + r5*(r5*(-u[t0][x + 4][y + 3]) + r5*u[t0][x + 4][y + 4])*b[x + 4][y + 3]))/b[x + 4][y + 2] + (-r4*vp[x + 3][y + 2]*vp[x + 3][y + 2]*(r4*r10*b[x + 4][y + 2] + r4*(-(r4*r14 + r4*u[t0][x + 4][y + 2])*b[x + 3][y + 2]) + r5*(-(r5*r14 + r5*u[t0][x + 3][y + 3])*b[x + 3][y + 2]) + r5*(r5*r15 + r5*u[t0][x + 3][y + 4])*b[x + 3][y + 3]))/b[x + 3][y + 2])*b[x + 3][y + 2];
        r9[x + 1][y + 1] = ((r5*(vp[x + 2][y + 4]*vp[x + 2][y + 4])*(r4*(-(r4*r12 + r4*u[t0][x + 3][y + 4])*b[x + 2][y + 4]) + r4*(r4*(-u[t0][x + 3][y + 4]) + r4*u[t0][x + 4][y + 4])*b[x + 3][y + 4] + r5*(-r11*b[x + 2][y + 4]) + r5*(r5*(-u[t0][x + 2][y + 5]) + r5*u[t0][x + 2][y + 6])*b[x + 2][y + 5]))/b[x + 2][y + 4] + (-r5*vp[x + 2][y + 3]*vp[x + 2][y + 3]*(r4*(-(r4*r16 + r4*u[t0][x + 3][y + 3])*b[x + 2][y + 3]) + r4*(r4*r15 + r4*u[t0][x + 4][y + 3])*b[x + 3][y + 3] + r5*r11*b[x + 2][y + 4] + r5*(-(r5*r16 + r5*u[t0][x + 2][y + 4])*b[x + 2][y + 3])))/b[x + 2][y + 3])*b[x + 2][y + 3];
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(b,damp,u,vp:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        float r18 = -u[t0][x + 2][y + 2];
        float r17 = 1.0F/(vp[x + 2][y + 2]*vp[x + 2][y + 2]);
        u[t2][x + 2][y + 2] = (r7*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] + r17*(-r6*(-2.0F*u[t0][x + 2][y + 2]) - r6*u[t1][x + 2][y + 2]) + ((1.0F/12.0F)*(s*s)*(r4*(-r8[x][y + 1]) + r4*r8[x + 1][y + 1] + r5*(-r9[x + 1][y]) + r5*r9[x + 1][y + 1]) + r4*(-(r4*r18 + r4*u[t0][x + 3][y + 2])*b[x + 2][y + 2]) + r4*(r4*(-u[t0][x + 3][y + 2]) + r4*u[t0][x + 4][y + 2])*b[x + 3][y + 2] + r5*(-(r5*r18 + r5*u[t0][x + 2][y + 3])*b[x + 2][y + 2]) + r5*(r5*(-u[t0][x + 2][y + 3]) + r5*u[t0][x + 2][y + 4])*b[x + 2][y + 3])/b[x + 2][y + 2])/(r6*r17 + r7*damp[x + 1][y + 1]);
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */

    /* Begin section1 */
    START_TIMER(section1)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float posx = -o_x + src_coords[p_src][0];
      float posy = -o_y + src_coords[p_src][1];
      int ii_src_0 = (int)(floor(posx/h_x));
      int ii_src_1 = (int)(floor(posy/h_y));
      int ii_src_2 = 1 + (int)(floor(posy/h_y));
      int ii_src_3 = 1 + (int)(floor(posx/h_x));
      float px = (float)(-h_x*(int)(floor(posx/h_x)) + posx);
      float py = (float)(-h_y*(int)(floor(posy/h_y)) + posy);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
      {
        float r0 = (s*s)*(vp[ii_src_0 + 2][ii_src_1 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2])*(1 - py/h_y - px/h_x + px*py/(h_x*h_y))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2] += r0;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
      {
        float r1 = (s*s)*(vp[ii_src_0 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_2 + 2])*(py/h_y - px*py/(h_x*h_y))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_2 + 2] += r1;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r2 = (s*s)*(vp[ii_src_3 + 2][ii_src_1 + 2]*vp[ii_src_3 + 2][ii_src_1 + 2])*(px/h_x - px*py/(h_x*h_y))*src[time][p_src];
        u[t2][ii_src_3 + 2][ii_src_1 + 2] += r2;
      }
      if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r3 = px*py*(s*s)*(vp[ii_src_3 + 2][ii_src_2 + 2]*vp[ii_src_3 + 2][ii_src_2 + 2])*src[time][p_src]/(h_x*h_y);
        u[t2][ii_src_3 + 2][ii_src_2 + 2] += r3;
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin section2 */
    START_TIMER(section2)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      int ii_rec_0 = (int)(floor(posx/h_x));
      int ii_rec_1 = (int)(floor(posy/h_y));
      int ii_rec_2 = 1 + (int)(floor(posy/h_y));
      int ii_rec_3 = 1 + (int)(floor(posx/h_x));
      float px = (float)(-h_x*(int)(floor(posx/h_x)) + posx);
      float py = (float)(-h_y*(int)(floor(posy/h_y)) + posy);
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        sum += (1 - py/h_y - px/h_x + px*py/(h_x*h_y))*u[t2][ii_rec_0 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        sum += (py/h_y - px*py/(h_x*h_y))*u[t2][ii_rec_0 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += (px/h_x - px*py/(h_x*h_y))*u[t2][ii_rec_3 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += px*py*u[t2][ii_rec_3 + 2][ii_rec_2 + 2]/(h_x*h_y);
      }
      rec[time][p_rec] = sum;
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
  }

  free(r8_vec);
  free(r9_vec);

  return 0;
}

