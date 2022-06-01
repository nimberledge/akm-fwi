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


int Kernel(const float dt, const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, struct profiler * timers)
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
  float r1 = 1.0F/(h_x*h_x);
  float r2 = 1.0F/(h_y*h_y);
  float r3 = 1.0F/(h_z*h_z);

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
              float r13 = -2.0F*u[t0][x + 2][y + 2][z + 2];
              float r12 = vp[x + 2][y + 2][z + 2]*vp[x + 2][y + 2][z + 2];
              u[t2][x + 2][y + 2][z + 2] = r12*(dt*dt)*(r1*r13 + r1*u[t0][x + 1][y + 2][z + 2] + r1*u[t0][x + 3][y + 2][z + 2] + r2*r13 + r2*u[t0][x + 2][y + 1][z + 2] + r2*u[t0][x + 2][y + 3][z + 2] + r3*r13 + r3*u[t0][x + 2][y + 2][z + 1] + r3*u[t0][x + 2][y + 2][z + 3] + (-r0*r13 - r0*u[t1][x + 2][y + 2][z + 2])/r12);
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
      int ii_src_0 = (int)(floor(posx/h_x));
      int ii_src_1 = (int)(floor(posy/h_y));
      int ii_src_2 = (int)(floor(posz/h_z));
      int ii_src_3 = 1 + (int)(floor(posz/h_z));
      int ii_src_4 = 1 + (int)(floor(posy/h_y));
      int ii_src_5 = 1 + (int)(floor(posx/h_x));
      float px = (float)(-h_x*(int)(floor(posx/h_x)) + posx);
      float py = (float)(-h_y*(int)(floor(posy/h_y)) + posy);
      float pz = (float)(-h_z*(int)(floor(posz/h_z)) + posz);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r4 = 6.25F*(vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2])*(1 - pz/h_z - py/h_y + py*pz/(h_y*h_z) - px/h_x + px*pz/(h_x*h_z) + px*py/(h_x*h_y) - px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2][ii_src_2 + 2] += r4;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r5 = 6.25F*(vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2]*vp[ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2])*(pz/h_z - py*pz/(h_y*h_z) - px*pz/(h_x*h_z) + px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_1 + 2][ii_src_3 + 2] += r5;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r6 = 6.25F*(vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2]*vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2])*(py/h_y - py*pz/(h_y*h_z) - px*py/(h_x*h_y) + px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_4 + 2][ii_src_2 + 2] += r6;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r7 = 6.25F*(vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2]*vp[ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2])*(py*pz/(h_y*h_z) - px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_0 + 2][ii_src_4 + 2][ii_src_3 + 2] += r7;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r8 = 6.25F*(vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2]*vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2])*(px/h_x - px*pz/(h_x*h_z) - px*py/(h_x*h_y) + px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_1 + 2][ii_src_2 + 2] += r8;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r9 = 6.25F*(vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2]*vp[ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2])*(px*pz/(h_x*h_z) - px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_1 + 2][ii_src_3 + 2] += r9;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r10 = 6.25F*(vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2]*vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2])*(px*py/(h_x*h_y) - px*py*pz/(h_x*h_y*h_z))*src[time][p_src];
        u[t2][ii_src_5 + 2][ii_src_4 + 2][ii_src_2 + 2] += r10;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r11 = 6.25F*px*py*pz*(vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2]*vp[ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2])*src[time][p_src]/(h_x*h_y*h_z);
        u[t2][ii_src_5 + 2][ii_src_4 + 2][ii_src_3 + 2] += r11;
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
      int ii_rec_0 = (int)(floor(posx/h_x));
      int ii_rec_1 = (int)(floor(posy/h_y));
      int ii_rec_2 = (int)(floor(posz/h_z));
      int ii_rec_3 = 1 + (int)(floor(posz/h_z));
      int ii_rec_4 = 1 + (int)(floor(posy/h_y));
      int ii_rec_5 = 1 + (int)(floor(posx/h_x));
      float px = (float)(-h_x*(int)(floor(posx/h_x)) + posx);
      float py = (float)(-h_y*(int)(floor(posy/h_y)) + posy);
      float pz = (float)(-h_z*(int)(floor(posz/h_z)) + posz);
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
      {
        sum += (1 - pz/h_z - py/h_y + py*pz/(h_y*h_z) - px/h_x + px*pz/(h_x*h_z) + px*py/(h_x*h_y) - px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_0 + 2][ii_rec_1 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
      {
        sum += (pz/h_z - py*pz/(h_y*h_z) - px*pz/(h_x*h_z) + px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_0 + 2][ii_rec_1 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (py/h_y - py*pz/(h_y*h_z) - px*py/(h_x*h_y) + px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_0 + 2][ii_rec_4 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (py*pz/(h_y*h_z) - px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_0 + 2][ii_rec_4 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (px/h_x - px*pz/(h_x*h_z) - px*py/(h_x*h_y) + px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_5 + 2][ii_rec_1 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (px*pz/(h_x*h_z) - px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_5 + 2][ii_rec_1 + 2][ii_rec_3 + 2];
      }
      if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (px*py/(h_x*h_y) - px*py*pz/(h_x*h_y*h_z))*u[t2][ii_rec_5 + 2][ii_rec_4 + 2][ii_rec_2 + 2];
      }
      if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += px*py*pz*u[t2][ii_rec_5 + 2][ii_rec_4 + 2][ii_rec_3 + 2]/(h_x*h_y*h_z);
      }
      rec[time][p_rec] = sum;
    }
    STOP_TIMER(section2,timers)
    /* End section2 */
  }

  return 0;
}