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
  double section3;
} ;


int Kernel(struct dataobj *restrict damp_vec, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const float dt, const float o_x, const float o_y, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x_size, const int y_size, struct profiler * timers)
{
  float *r4_vec;
  posix_memalign((void**)(&r4_vec),64,(x_size + 2)*(y_size + 2)*sizeof(float));
  float *r5_vec;
  posix_memalign((void**)(&r5_vec),64,(x_size + 2)*(y_size + 2)*sizeof(float));
  float *r8_vec;
  posix_memalign((void**)(&r8_vec),64,(x_size + 2)*(y_size + 2)*sizeof(float));

  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict r4)[y_size + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2]) r4_vec;
  float (*restrict r5)[y_size + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2]) r5_vec;
  float (*restrict r8)[y_size + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2]) r8_vec;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* Begin section0 */
  START_TIMER(section0)
  for (int x = x_m - 1; x <= x_M + 1; x += 1)
  {
    #pragma omp simd aligned(damp,vp:32)
    for (int y = y_m - 1; y <= y_M + 1; y += 1)
    {
      float r9 = 1.0F/(vp[x + 2][y + 2]*vp[x + 2][y + 2]);
      r4[x + 1][y + 1] = 1.0F/(r9/((dt*dt)) + damp[x + 1][y + 1]/dt);
      r5[x + 1][y + 1] = r9;
    }
  }
  STOP_TIMER(section0,timers)
  /* End section0 */

  float r6 = 1.0F/(dt*dt);
  float r7 = 1.0F/dt;

  for (int time = time_m, t0 = (time)%(491), t1 = (time + 490)%(491), t2 = (time + 1)%(491); time <= time_M; time += 1, t0 = (time)%(491), t1 = (time + 490)%(491), t2 = (time + 1)%(491))
  {
    /* Begin section1 */
    START_TIMER(section1)
    for (int x = x_m - 1; x <= x_M + 1; x += 1)
    {
      #pragma omp simd aligned(u:32)
      for (int y = y_m - 1; y <= y_M + 1; y += 1)
      {
        r8[x + 1][y + 1] = 3.47140078142802e+1F*(4.0e-4F*(u[t0][x + 1][y + 2] + u[t0][x + 2][y + 1] + u[t0][x + 2][y + 3] + u[t0][x + 3][y + 2]) - 1.59999996e-3F*u[t0][x + 2][y + 2])*r5[x + 1][y + 1] + u[t0][x + 2][y + 2];
      }
    }
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd aligned(damp,u:32)
      for (int y = y_m; y <= y_M; y += 1)
      {
        u[t2][x + 2][y + 2] = (r7*damp[x + 1][y + 1]*u[t0][x + 2][y + 2] - (r6*(-2.0F*u[t0][x + 2][y + 2]) + r6*u[t1][x + 2][y + 2])*r5[x + 1][y + 1] + 4.0e-4F*(r8[x][y + 1] + r8[x + 1][y] + r8[x + 1][y + 2] + r8[x + 2][y + 1]) - 1.59999996e-3F*r8[x + 1][y + 1])*r4[x + 1][y + 1];
      }
    }
    STOP_TIMER(section1,timers)
    /* End section1 */

    /* Begin section2 */
    START_TIMER(section2)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float posx = -o_x + src_coords[p_src][0];
      float posy = -o_y + src_coords[p_src][1];
      int ii_src_0 = (int)(floor(2.0e-2F*posx));
      int ii_src_1 = (int)(floor(2.0e-2F*posy));
      int ii_src_2 = 1 + (int)(floor(2.0e-2F*posy));
      int ii_src_3 = 1 + (int)(floor(2.0e-2F*posx));
      float px = (float)(posx - 5.0e+1F*(int)(floor(2.0e-2F*posx)));
      float py = (float)(posy - 5.0e+1F*(int)(floor(2.0e-2F*posy)));
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
      {
        float r0 = (dt*dt)*(vp[ii_src_0 + 2][ii_src_1 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2])*(4.0e-4F*px*py - 2.0e-2F*px - 2.0e-2F*py + 1)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2] += r0;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
      {
        float r1 = (dt*dt)*(vp[ii_src_0 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_2 + 2])*(-4.0e-4F*px*py + 2.0e-2F*py)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_2 + 2] += r1;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_src_3 + 2][ii_src_1 + 2]*vp[ii_src_3 + 2][ii_src_1 + 2])*(-4.0e-4F*px*py + 2.0e-2F*px)*src[time][p_src];
        u[t2][ii_src_3 + 2][ii_src_1 + 2] += r2;
      }
      if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        float r3 = 4.0e-4F*px*py*(dt*dt)*(vp[ii_src_3 + 2][ii_src_2 + 2]*vp[ii_src_3 + 2][ii_src_2 + 2])*src[time][p_src];
        u[t2][ii_src_3 + 2][ii_src_2 + 2] += r3;
      }
    }
    STOP_TIMER(section2,timers)
    /* End section2 */

    /* Begin section3 */
    START_TIMER(section3)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      int ii_rec_0 = (int)(floor(2.0e-2F*posx));
      int ii_rec_1 = (int)(floor(2.0e-2F*posy));
      int ii_rec_2 = 1 + (int)(floor(2.0e-2F*posy));
      int ii_rec_3 = 1 + (int)(floor(2.0e-2F*posx));
      float px = (float)(posx - 5.0e+1F*(int)(floor(2.0e-2F*posx)));
      float py = (float)(posy - 5.0e+1F*(int)(floor(2.0e-2F*posy)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        sum += (4.0e-4F*px*py - 2.0e-2F*px - 2.0e-2F*py + 1)*u[t2][ii_rec_0 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        sum += (-4.0e-4F*px*py + 2.0e-2F*py)*u[t2][ii_rec_0 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += (-4.0e-4F*px*py + 2.0e-2F*px)*u[t2][ii_rec_3 + 2][ii_rec_1 + 2];
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        sum += 4.0e-4F*px*py*u[t2][ii_rec_3 + 2][ii_rec_2 + 2];
      }
      rec[time][p_rec] = sum;
    }
    STOP_TIMER(section3,timers)
    /* End section3 */
  }

  free(r4_vec);
  free(r5_vec);
  free(r8_vec);

  return 0;
}
