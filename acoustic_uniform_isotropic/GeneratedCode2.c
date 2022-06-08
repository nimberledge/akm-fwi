#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
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
  int * size;
  int * npsize;
  int * dsize;
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


int Kernel(const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, struct profiler * timers)
{
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  float r0 = 1.0F/(dt*dt);

  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0; x <= MIN(x0_blk0 + x0_blk0_size - 1, x_M); x += 1)
        {
          for (int y = y0_blk0; y <= MIN(y0_blk0 + y0_blk0_size - 1, y_M); y += 1)
          {
            #pragma omp simd aligned(u,vp:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r9 = vp[x + 2][y + 2][z + 2]*vp[x + 2][y + 2][z + 2];
              u[t2][x + 2][y + 2][z + 2] = r9*(dt*dt)*(4.0e-2F*(u[t0][x + 1][y + 2][z + 2] + u[t0][x + 2][y + 1][z + 2] + u[t0][x + 2][y + 2][z + 1] + u[t0][x + 2][y + 2][z + 3] + u[t0][x + 2][y + 3][z + 2] + u[t0][x + 3][y + 2][z + 2]) - 2.39999995e-1F*u[t0][x + 2][y + 2][z + 2] + (-r0*(-2.0F*u[t0][x + 2][y + 2][z + 2]) - r0*u[t1][x + 2][y + 2][z + 2])/r9);
            }
          }
        }
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
      float posz = -o_z + src_coords[p_src][2];
      int ii_src_0 = (int)(floor(2.0e-1F*posx));
      int ii_src_1 = (int)(floor(2.0e-1F*posy));
      int ii_src_2 = (int)(floor(2.0e-1F*posz));
      int ii_src_3 = 1 + (int)(floor(2.0e-1F*posz));
      int ii_src_4 = 1 + (int)(floor(2.0e-1F*posy));
      int ii_src_5 = 1 + (int)(floor(2.0e-1F*posx));
      float px = (float)(posx - 5.0F*(int)(floor(2.0e-1F*posx)));
      float py = (float)(posy - 5.0F*(int)(floor(2.0e-1F*posy)));
      float pz = (float)(posz - 5.0F*(int)(floor(2.0e-1F*posz)));
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r1 = 6.25F*(vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2])*(-8.0e-3F*px*py*pz + 4.0e-2F*px*py + 4.0e-2F*px*pz - 2.0e-1F*px + 4.0e-2F*py*pz - 2.0e-1F*py - 2.0e-1F*pz + 1)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2] += r1;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r2 = 6.25F*(vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2])*(8.0e-3F*px*py*pz - 4.0e-2F*px*pz - 4.0e-2F*py*pz + 2.0e-1F*pz)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r3 = 6.25F*(vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2])*(8.0e-3F*px*py*pz - 4.0e-2F*px*py - 4.0e-2F*py*pz + 2.0e-1F*py)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2] += r3;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r4 = 6.25F*(vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2]*vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2])*(-8.0e-3F*px*py*pz + 4.0e-2F*py*pz)*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2] += r4;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r5 = 6.25F*(vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2]*vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2])*(8.0e-3F*px*py*pz - 4.0e-2F*px*py - 4.0e-2F*px*pz + 2.0e-1F*px)*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2] += r5;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r6 = 6.25F*(vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2]*vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2])*(-8.0e-3F*px*py*pz + 4.0e-2F*px*pz)*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2] += r6;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r7 = 6.25F*(vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2]*vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2])*(-8.0e-3F*px*py*pz + 4.0e-2F*px*py)*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2] += r7;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r8 = 5.00000022351742e-2F*px*py*pz*(vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2]*vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2])*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2] += r8;
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
      float posz = -o_z + rec_coords[p_rec][2];
      int ii_rec_0 = (int)(floor(2.0e-1F*posx));
      int ii_rec_1 = (int)(floor(2.0e-1F*posy));
      int ii_rec_2 = (int)(floor(2.0e-1F*posz));
      int ii_rec_3 = 1 + (int)(floor(2.0e-1F*posz));
      int ii_rec_4 = 1 + (int)(floor(2.0e-1F*posy));
      int ii_rec_5 = 1 + (int)(floor(2.0e-1F*posx));
      float px = (float)(posx - 5.0F*(int)(floor(2.0e-1F*posx)));
      float py = (float)(posy - 5.0F*(int)(floor(2.0e-1F*posy)));
      float pz = (float)(posz - 5.0F*(int)(floor(2.0e-1F*posz)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
      {
        sum += (-8.0e-3F*px*py*pz + 4.0e-2F*px*py + 4.0e-2F*px*pz - 2.0e-1F*px + 4.0e-2F*py*pz - 2.0e-1F*py - 2.0e-1F*pz + 1)*u[t2][ii_rec_0 + 2][ii_rec_1 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
      {
        sum += (8.0e-3F*px*py*pz - 4.0e-2F*px*pz - 4.0e-2F*py*pz + 2.0e-1F*pz)*u[t2][ii_rec_0 + 2][ii_rec_1 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (8.0e-3F*px*py*pz - 4.0e-2F*px*py - 4.0e-2F*py*pz + 2.0e-1F*py)*u[t2][ii_rec_0 + 2][ii_rec_4 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (-8.0e-3F*px*py*pz + 4.0e-2F*py*pz)*u[t2][ii_rec_0 + 2][ii_rec_4 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (8.0e-3F*px*py*pz - 4.0e-2F*px*py - 4.0e-2F*px*pz + 2.0e-1F*px)*u[t2][ii_rec_5 + 2][ii_rec_1 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-8.0e-3F*px*py*pz + 4.0e-2F*px*pz)*u[t2][ii_rec_5 + 2][ii_rec_1 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-8.0e-3F*px*py*pz + 4.0e-2F*px*py)*u[t2][ii_rec_5 + 2][ii_rec_4 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += 8.0e-3F*px*py*pz*u[t2][ii_rec_5 + 2][ii_rec_4 + 2][ii_rec_3 + 2];
      }
      rec[time][p_rec] = sum;
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
  }

  return 0;
}

