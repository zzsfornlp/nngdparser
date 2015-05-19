/*
 * This file is part of the continuous space language and translation model toolkit
 * for statistical machine translation and large vocabulary speech recognition.
 *
 * Copyright 2014, Holger Schwenk, LIUM, University of Le Mans, France
 *
 * The CSLM toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
 *
 * $Id: Blas.h,v 1.21 2014/03/25 21:52:53 schwenk Exp $
 */

#ifndef _Blas_h
#define _Blas_h

#include <string.h>	// memcpy()
#include "Tools.h"	


//-------------------------------------------
// support for Intel's MKL
//-------------------------------------------

#ifdef BLAS_INTEL_MKL
  extern int inc1;
  #include "mkl_blas.h"
  #include "mkl_vml.h"
// for single precision
  #define COPY scopy	// BLAS1
  #define ASUM sasum
  #define AXPY saxpy
  #define SCAL sscal
  #define GEMV sgemv	// BLAS2
  #define GEMM sgemm	// BLAS3
    // special vectorized functions of MKL
#if 0
  #define VSQR atlas_vsqr
  #define VLOG atlas_vlog
  #define VTANH atlas_vtanh
  #define VEXP atlas_vexp
  extern "C" void atlas_vtanh(int *n, float *d);
  extern "C" void atlas_vlog(int *n, float *d);
  extern "C" void atlas_vexp(int *n, float *d);
  extern "C" void atlas_vsqr(int *n, float *d);
#else
  #define VSQR(n,d) vssqr_(n,d,d)
  #define VLOG(n,d) vslog_(n,d,d)
  #define VTANH(n,d) vstanh_(n,d,d)
  #define VEXP(n,d) vsexp_(n,d,d)
#endif

#endif

//-------------------------------------------
// support for Nvidia GPU cards
//-------------------------------------------

#ifdef BLAS_CUDA
  #define CUDA float			// memory on the GPU card 
  #define CUDA_SIZE (sizeof(float))	// memory on the GPU card 

  #include <vector>
  #include "cublas.h"
  #include "curand.h"
  #include "npps.h"

  extern cublasStatus cuda_stat;
  extern curandGenerator_t cuda_gen;
  extern vector<int> cuda_devs;	// user specified list of GPUs tu be used

    // error checking of calls to cublas
  inline void cuda_check_error(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err){
      ErrorN("CUDA: ERROR %d in %s: %s\n",
             cuda_stat, msg, cudaGetErrorString(err));
    }
    cuda_stat=cublasGetError();
    if (cuda_stat != CUBLAS_STATUS_SUCCESS){
      const char* err_str;
      switch (cuda_stat){
        case CUBLAS_STATUS_SUCCESS:
            err_str = "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            err_str = "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            err_str = "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            err_str = "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            err_str = "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            err_str = "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            err_str = "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            err_str = "CUBLAS_STATUS_INTERNAL_ERROR";
	default:
	    err_str = "unknown error code";
      }
      ErrorN("CUDA: ERROR %d in %s: %s \n",
             cuda_stat, msg, err_str);
    }
  }

    // memory allocation with error checks
  inline REAL *cuda_alloc(int dim,const char *msg) {
    void *gpu_mem;
    char err_msg[1024];
    sprintf(err_msg, "CUDA: can't allocate memory for %s", msg);
    if (dim>0) {
      cublasAlloc(dim, CUDA_SIZE, &gpu_mem);
#ifdef DEBUG
      int dev = -1;
      cudaGetDevice(&dev);
             dim*CUDA_SIZE, gpu_mem, dev);
#endif
      cuda_check_error(err_msg);
      if (!gpu_mem) Error(err_msg);
      return (CUDA*) gpu_mem;
    }
    else return NULL;
  } 

  #define COPY cublasScopy	// Blas1
  #define ASUM cublasSasum
  #define AXPY cublasSaxpy
  #define SCAL cublasSscal
  #define GEMV cublasSgemv	// Blas2
  #define GEMM cublasSgemm	// Blas3
  
  #define VSQR(n,d) nppsSqr_32f_I(d,*n)
  #define VLOG(n,d) nppsLn_32f_I(d,*n)
  #define VEXP(n,d)  nppsExp_32f_I(d,*n)
#endif

//-------------------------------------------
// support for standard BLAS
//-------------------------------------------

#ifdef BLAS_ATLAS
extern "C" void sscal_(const int *n, float *a, const float *x, const int *incx);
extern "C" float sasum_(const int *n, const float *x, const int *incx);
extern "C" void saxpy_(const int *n, const float *a, const float *x, const int *incx, float *y, const int *incy);
extern "C" void scopy_(int *n, const float *x, int *incx, float *y, int *incy);
extern "C" void sgemv_(const char *trans, const int *m, const int *n, const float *alpha,
		      const float *a, const int *lda, const float *x, const int *incx,
		      const float *beta, float *y, const int *incy);
extern "C" void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
		      const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
		      const float *beta, float *c, const int *ldc);
  #define COPY scopy_
  #define GEMV sgemv_
  #define GEMM sgemm_
  #define AXPY saxpy_
  #define SCAL sscal_
  #define ASUM sasum_

  extern int inc1;

  extern "C" void atlas_vtanh(int *n, float *d);
  extern "C" void atlas_vlog(int *n, float *d);
  extern "C" void atlas_vexp(int *n, float *d);
  extern "C" void atlas_vsqr(int *n, float *d);

  #define VSQR atlas_vsqr
  #define VLOG atlas_vlog
  #define VTANH atlas_vtanh
  #define VEXP atlas_vexp
#endif

// matrix/vector multiplication: c = 1.0*A * b + 1.0 * c
// the matrix must be stored in COLUM MAJOR order

/*--------------------------------------------------------------------------*
 *
 * Wrapper routine for GEMV function
 * that uses the TRANSPOSED fortran routine
 *
 * dest = matrix * source + bias
 *
 *   dest: dim_dest x 1
 * matrix: dim_dest x dim_src
 * source: dim_src x 1
 *
 *--------------------------------------------------------------------------*/

inline void call_gemv (REAL *dest, REAL *matrix, REAL *source, REAL *bias,
				int dim_dest, int dim_src)
{
	char    trans = 'N';
	REAL    fact = 1.0;
	int	inc = 1;
        
	// int sgemv(char *trans, integer *m, integer *n,
	//	 real *alpha, *real *a, integer *lda,
	//	 real *x, integer *incx, real *beta, real *y, *integer *incy)
	//
	// y := alpha*A*x + beta*y
	//         m x n


#ifdef BLAS_CUDA
	COPY(dim_dest,bias,inc,dest,inc);  // TODO: verify
  	GEMV(trans, dim_dest, dim_src, fact, matrix, dim_dest, source, inc, fact, dest, inc);
        cuda_check_error("call_gemv");
#else
	memcpy(dest, bias, dim_dest * sizeof(REAL));
  	GEMV(&trans, &dim_dest, &dim_src, &fact, matrix, &dim_dest, source, &inc, &fact, dest, &inc);
#endif
}


// matrix/matrix multiplication: C = alpha*A * B + beta * C
// both must be stored in COLUM MAJOR order

inline void call_gemm (REAL *C, REAL *A, REAL *B, REAL beta, int dimy, int dimx, int dimk)
{
  char    transN = 'N';
  REAL    alpha = 1.0;

    // gemm ( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc )
    //   * C = alpha*A * B + beta * b
    //              mxn mxk        kxn
    //              lda     ldb  ldc

  TRACE("-mkl- call gemm\n");
#ifdef BLAS_CUDA
  GEMM (transN, transN, dimy, dimx, dimk, alpha, A, dimy, B, dimk, beta, C, dimy);
  cuda_check_error("call_gemm");
#else
  GEMM (&transN, &transN, &dimy, &dimx, &dimk, &alpha, A, &dimy, B, &dimk, &beta, C, &dimy);
#endif
}

#endif
