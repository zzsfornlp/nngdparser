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
 * $Id: Gpu.cuh,v 1.19 2014/03/25 21:52:53 schwenk Exp $
 */


#ifndef _Gpu_cuh
#define _Gpu_cuh

void GpuMachTabForw(const int bsize, const int odim,
		    REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_data_out);

void GpuMachTabBackw(const REAL lrate, const int bsize, const int odim,
		    REAL *gpu_data_in, REAL *gpu_t, REAL *gpu_grad_out);

void GpuMachSoftmaxForw(const int bsize, const int odim, REAL *gpu_data_out);
void GpuMachSoftmaxStableForw(const int bsize, const int odim, REAL *gpu_data_out);

void GpuLinRectifForw(const int n, REAL *gpu_data_out);
void GpuLinRectifBackw(const int n, REAL *gpu_data_out, REAL *gpu_grad_out);

void GpuDropOut(const int n, REAL *gpu_vect, REAL *rand, REAL thresh);

REAL GpuErrFctSoftmCrossEntNgramCalcValue(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_target);

void GpuErrFctSoftmCrossEntNgramCalcGrad(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res);
void GpuErrFctSoftmCrossEntNgramCalcGradNull(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target, REAL * gpu_res);
void GpuErrFctSoftmCrossEntNgramCalcGradCumul(const int bsize, const int odim, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target);

REAL GpuErrFctSoftmCrossEntNgramMultiCalcGrad(const int bsize, const int dim, const int nb, REAL *gpu_data_out, REAL *gpu_grad, REAL *gpu_target);

void GpuCopyVectorToMatrix(REAL * mat, REAL * vec, const int M, const int N);
void GpuCopyMatrixToMatrixStrided(REAL * dst, REAL * src, const int M, const int N, const int row_stride);
void GpuCopyMatrixStridedToMatrix(REAL * dst, REAL * src, const int M, const int N, const int row_stride);

void GpuBatchedAXPY(const int n, const REAL a, REAL * x, const int incx,
                    REAL * y, const int incy, const int nb_batch);

void GpuResSet(REAL val);
REAL GpuResGet();

#endif
