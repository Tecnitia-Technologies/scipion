/***************************************************************************
 *
 * Authors:     Miguel Ascanio Gómez (miguel.ascanio@tecnitia.com)
 * 				Alberto Casas Ortiz (alberto.casas@tecnitia.com)
 * 				David Gómez Blanco (david.gomez@tecnitia.com)
 * 				Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/
#include "cuda_gpu_estimate_ctf.h"

#include <stdio.h>
#include <algorithm>
#include <complex>
#include <cuda.h>
#include <cufft.h>

#include <cufftXt.h>
#include <cmath>

#include <data/xmipp_macros.h>

#include <TicTocHeaderOnly.h>

// Error handling
#include <iostream>

namespace cuda_gpu_estimate_ctf_err {
cudaError_t __e;
cufftResult_t __fftr;

const char* cudaGetCuFFTResultString(cufftResult_t r) {
	switch (r) {
	case CUFFT_SUCCESS:
		return "The cuFFT operation was successful";
	case CUFFT_INVALID_PLAN:
		return "cuFFT was passed an invalid plan handle";
	case CUFFT_ALLOC_FAILED:
		return "cuFFT failed to allocate GPU or CPU memory";
	case CUFFT_INVALID_TYPE:
		return "No longer used CUFFT_INVALID_TYPE";
	case CUFFT_INVALID_VALUE:
		return "User specified an invalid pointer or parameter";
	case CUFFT_INTERNAL_ERROR:
		return "Driver or internal cuFFT library error";
	case CUFFT_EXEC_FAILED:
		return "Failed to execute an FFT on the GPU";
	case CUFFT_SETUP_FAILED:
		return "The cuFFT library failed to initialize";
	case CUFFT_INVALID_SIZE:
		return "User specified an invalid transform size";
	case CUFFT_UNALIGNED_DATA:
		return "No longer used CUFFT_UNALIGNED_DATA";
	case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "Missing parameters in call";
	case CUFFT_INVALID_DEVICE:
		return "Execution of a plan was on different GPU than plan creation";
	case CUFFT_PARSE_ERROR:
		return "Internal plan database error";
	case CUFFT_NO_WORKSPACE:
		return "No workspace has been provided prior to plan execution";
	case CUFFT_NOT_IMPLEMENTED:
		return "Function does not implement functionality for parameters given.";
	case CUFFT_LICENSE_ERROR:
		return "Used in previous versions.";
	case CUFFT_NOT_SUPPORTED:
		return "Operation is not supported for parameters given.";
	default:
		return "Unknown result code";
	}
}

#define CU_CHK(x) __e = x;\
if (__e != 0) { \
std::cerr << "ERROR, " << __FILE__ << ":" << __LINE__ << ", " <<  cudaGetErrorString(__e) << std::endl;\
exit(EXIT_FAILURE); }

#define FFT_CHK(x) __fftr = x;\
if (__fftr != 0) { \
	std::cerr << "ERROR, " << __FILE__ << ":" << __LINE__ << ", " <<  cudaGetCuFFTResultString(__fftr) << std::endl;\
exit(EXIT_FAILURE); }
}

using namespace cuda_gpu_estimate_ctf_err;

void computePieceAvgStd(double* in, size_t pieceDim, size_t y0, size_t yEnd,
		size_t x0, size_t xEnd, size_t yDim, double& avg, double& stddev) {
	size_t len = pieceDim * pieceDim;
	avg = 0.0;
	stddev = 0.0;
	double val;
	for (size_t y = y0; y < yEnd; y++) {
		for (size_t x = x0; x < xEnd; x++) {
			//std::cerr << "x0 " << x0 << " y0 " << y0 << " xEnd " << xEnd << " yEnd " << yEnd << " x " << x << " y " << y << " p " << y * yDim + x << std::endl;
			val = in[y * yDim + x];
			avg += val;
			stddev += val * val;
		}
	}
	avg /= len;
	stddev = stddev / len - avg * avg;
	stddev *= len / (len - 1);
	// Foreseeing numerical instabilities
	stddev = std::sqrt(std::fabs(stddev));
}

const int K_SMOOTH_BLOCK_SIZE_X = 32;
const int K_SMOOTH_BLOCK_SIZE_Y = 32;

__global__ void smooth(double* piece, double* mic, double* pieceSmoother, size_t pieceDim, size_t y0, size_t x0, size_t yDim, double a, double b) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	size_t it = y * pieceDim + x;
	size_t micIt = (y0 + y) * yDim + x0 + x;
	piece[it] = (mic[micIt] * a + b) * pieceSmoother[it];
}


__global__ void post(cuDoubleComplex* fft, cuDoubleComplex* out, size_t pieceDim, size_t n, size_t pieceFFTNumPixels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int fila = ((-2)*pieceDim + 1 + sqrtf(((2 * pieceDim - 1) * (2 * pieceDim - 1)) - 8 * x)) * (-0.5);

	int triangular = ((fila)*(fila + 1))/2;

	cuDoubleComplex r = fft[x];
	out[x + triangular] = r;
	out[(pieceDim*pieceDim)-1-x-triangular] = r;

//	for (size_t i = 0; i < pieceDim; ++i) {
//		for (size_t j = 0; j < pieceDim; ++j) {
//			ptrDest = (double*) &psd[i * XSIZE_REAL + j];
//
//			if (j < XSIZE_FOURIER) {
//				iterator  = n * pieceFFTNumPixels + i * XSIZE_FOURIER + j;
//			} else {
//				iterator  = n * pieceFFTNumPixels
//						+ (((YSIZE_REAL) - i) % (YSIZE_REAL))
//								* XSIZE_FOURIER + ((XSIZE_REAL) - j);
//			}
//			val = *(h_fourier + iterator);
//			double real = cuCreal(val);
//			double imag = cuCimag(val);
//			*ptrDest += (real * real + imag * imag) * pieceDim * pieceDim;
//		}
//	}
}

void getAvgStd(double* avgSubpieces, double* stdSubpieces, size_t blocki,
		size_t blockj, size_t numSubpiecesX, size_t pieceDim, double& avg2,
		double& std2) {
	avg2 =    avgSubpieces[blocki       * numSubpiecesX + blockj]
			+ avgSubpieces[blocki       * numSubpiecesX + blockj + 1]
			+ avgSubpieces[(blocki + 1) * numSubpiecesX + blockj]
			+ avgSubpieces[(blocki + 1) * numSubpiecesX + blockj + 1];
	std2 =    stdSubpieces[blocki       * numSubpiecesX + blockj]
			+ stdSubpieces[blocki       * numSubpiecesX + blockj + 1]
			+ stdSubpieces[(blocki + 1) * numSubpiecesX + blockj]
			+ stdSubpieces[(blocki + 1) * numSubpiecesX + blockj + 1];
	size_t len = pieceDim * pieceDim;
	avg2 /= len;
	std2 = std2 / len - avg2 * avg2;
	std2 *= len / (len - 1);
	// Foreseeing numerical instabilities
	std2 = std::sqrt(std::fabs(std2));
}

// Piece normalization values
const double avgF = 0.0, stddevF = 1.0;
void cudaRunGpuEstimateCTF(double* mic, size_t xDim, size_t yDim, double overlap, size_t pieceDim, int skipBorders, double* pieceSmoother, double* psd) {
	TicToc t(true), tAvg(false), tPost(false), tTotal(true), tTotalCompute(true);

	tTotal.tic();
	// Calculate reduced input dim (exact multiple of pieceDim, without skipBorders)
	size_t divNumberX         = std::floor((double) xDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	size_t divNumberY         = std::floor((double) yDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	size_t divNumber          = divNumberX * divNumberY;

	size_t inNumPixels        = divNumber * pieceDim * pieceDim;
	size_t inSize             = inNumPixels * sizeof(double);

	size_t outNumPixels       = divNumber * pieceDim * (pieceDim / 2 + 1);
	size_t outSize            = outNumPixels * sizeof(cufftDoubleComplex);

	size_t pieceNumPixels     = pieceDim * pieceDim;
	size_t pieceSize          = pieceNumPixels * sizeof(double);

	size_t pieceFFTNumPixels  = pieceDim * (pieceDim / 2 + 1);
	size_t pieceFFTSize       = pieceFFTNumPixels * sizeof(cufftDoubleComplex);

	if (divNumberX <= 0 || divNumberY <= 0) {
		std::cerr << "Error, can't split a " << xDim << "X" << yDim << " MIC into " << pieceDim << "X" << pieceDim << " pieces" << std::endl
				  << "with " << overlap << " overlap and " << skipBorders << " skip borders," << std::endl
				  << "resulted in divNumberX=" << divNumberX << ", divNumberY=" << divNumberY << std::endl;
		exit(EXIT_FAILURE);
	}

	// Host page-locked memory
	cuDoubleComplex* h_fourier;
	t.tic();
#ifdef USE_PINNED
	CU_CHK(cudaMallocHost((void**) &h_fourier, outSize));
#else
	h_fourier = (cuDoubleComplex*) malloc(outSize);
#endif
	t.toc("Time to pinned malloc:\t\t");


	///////////////////////////////////////////////////////////////////////////
	size_t numSubpiecesX = divNumberX + 1;
	size_t numSubpiecesY = divNumberY + 1;

	double* avgSubpieces = new double[numSubpiecesX * numSubpiecesY];
	double* stdSubpieces = new double[numSubpiecesX * numSubpiecesY];

	std::fill_n(avgSubpieces, numSubpiecesX * numSubpiecesY, 0.0);
	std::fill_n(stdSubpieces, numSubpiecesX * numSubpiecesY, 0.0);

	size_t xTotalLim = numSubpiecesX * (pieceDim * (1-overlap));
	size_t yTotalLim = numSubpiecesY * (pieceDim * (1-overlap));

	std::cout << "xTotalLim " << xTotalLim << std::endl;
	std::cout << "yTotalLim " << yTotalLim << std::endl;

	t.tic();
	double val;
	int subPieceX = -1, subPieceY = -1;
	for (size_t y = 0; y < yTotalLim; y++) {
		if ((y % 256) == 0) {
			subPieceY++;
		}
		subPieceX = -1;
		for (size_t x = 0; x < xTotalLim; x++) {
			if ((x % 256) == 0) {
				subPieceX++;
			}
			val = mic[y * xDim + x];

			avgSubpieces[subPieceY * numSubpiecesX + subPieceX] += val;
			stdSubpieces[subPieceY * numSubpiecesX + subPieceX] += val * val;
		}
	}
	t.toc("Create subpieces");

	///////////////////////////////////////////////////////////////////////////

	// Test subpiece avg

//	t.tic();
//	for (size_t n = 0; n < divNumber; ++n) {
//		// Extract piece
//		size_t blocki = n / divNumberX;
//		size_t blockj = n % divNumberX;
//		size_t y0 = blocki * step + skipBorders * pieceDim;
//		size_t x0 = blockj * step + skipBorders * pieceDim;
//
//		// Test if the full piece is inside the micrograph
//		if (y0 + pieceDim > yDim)
//			y0 = yDim - pieceDim;
//
//		if (x0 + pieceDim > xDim)
//			x0 = xDim - pieceDim;
//
//		size_t yEnd = y0 + pieceDim;
//		size_t xEnd = x0 + pieceDim;
//
//		// ComputeAvgStdev
//		double avg = 0.0, stddev = 0.0;
//		computePieceAvgStd(mic, pieceDim, y0, yEnd, x0, xEnd, yDim, avg, stddev);
//
//		double avg2, std2;
//
//		getAvgStd(avgSubpieces, stdSubpieces, blocki, blockj, numSubpiecesX, pieceDim, avg2, std2);
//		if ((avg - avg2)  > 10e-9 || (stddev - std2) > 10e-9) {
//			std::cerr << "Blocki " << blocki << " Blockj " << blockj << std::endl
//					  << "avg " << avg << " avg2 " << avg2 << std::endl
//					  << "std " << stddev << " std2 " << std2 << std::endl;
//		}
//	}
//	t.toc("org avg");
//
//
//
//	return;


	// Device memory
	double* d_mic;
	double* d_pieces;
	cuDoubleComplex*d_fourier;
	double* d_pieceSmoother;
	t.tic();
	CU_CHK(cudaMalloc((void**)&d_mic, xDim * yDim * sizeof(double)));
	CU_CHK(cudaMalloc((void**)&d_pieces,  inSize));
	CU_CHK(cudaMalloc((void**)&d_fourier, outSize));
	CU_CHK(cudaMalloc((void**)&d_pieceSmoother, pieceSize));
	t.toc("Time to cuda malloc:\t\t");

	t.tic();
	CU_CHK(cudaMemcpy(d_mic, mic, xDim * yDim * sizeof(double), cudaMemcpyHostToDevice));
	CU_CHK(cudaMemcpy(d_pieceSmoother, pieceSmoother, pieceSize, cudaMemcpyHostToDevice));
	t.toc("Time to init memcpy:\t\t");

	// Kernel config
	dim3 dimBlock(K_SMOOTH_BLOCK_SIZE_X, K_SMOOTH_BLOCK_SIZE_Y);
	// Calculate number of grids needed for dimBlock,
	// wich is pieceDim % block == 0 ? pieceDim / block : pieceDim / block + 1
	// in a hacky way
	int k_grid_size_x = (pieceDim + K_SMOOTH_BLOCK_SIZE_X - 1) / K_SMOOTH_BLOCK_SIZE_X;
	int k_grid_size_y = (pieceDim + K_SMOOTH_BLOCK_SIZE_Y - 1) / K_SMOOTH_BLOCK_SIZE_Y;
	dim3 dimGrid(k_grid_size_x, k_grid_size_y);

	// CU FFT PLAN
	cudaStream_t* streams = new cudaStream_t[divNumber];
	cufftHandle*    plans = new cufftHandle[divNumber];

	t.tic();
	for (size_t n = 0; n < divNumber; ++n) {
		CU_CHK(cudaStreamCreate(streams + n));
		FFT_CHK(cufftPlan2d(plans + n, pieceDim, pieceDim, CUFFT_D2Z));
		FFT_CHK(cufftSetStream(plans[n], streams[n]));
	}
	t.toc("Time to plans and streams:\t");

	tTotalCompute.tic();
	// Iterate over all pieces
	size_t step = (size_t) (((1 - overlap) * pieceDim));
	for (size_t n = 0; n < divNumber; ++n) {
		tAvg.tic();
		// Extract piece
		size_t blocki = n / divNumberX;
		size_t blockj = n % divNumberX;
		size_t y0 = blocki * step + skipBorders * pieceDim;
		size_t x0 = blockj * step + skipBorders * pieceDim;

		// Test if the full piece is inside the micrograph
		if (y0 + pieceDim > yDim)
			y0 = yDim - pieceDim;

		if (x0 + pieceDim > xDim)
			x0 = xDim - pieceDim;

		size_t yEnd = y0 + pieceDim;
		size_t xEnd = x0 + pieceDim;

		// ComputeAvgStdev
		double avg = 0.0, stddev = 0.0;
		getAvgStd(avgSubpieces, stdSubpieces, blocki, blockj, numSubpiecesX, pieceDim, avg, stddev);
		//computePieceAvgStd(mic, pieceDim, y0, yEnd, x0, xEnd, yDim, avg, stddev);
		tAvg.toc("Time to avg std:\t\t");

		// Normalize and smooth
		double a, b;
		if (stddev != 0.0)
			a = stddevF / stddev;
		else
			a = 0.0;

		b = avgF - a * avg;

		// Aux pointers
		double* d_piecePtr            = d_pieces  + n * pieceNumPixels;
		cuDoubleComplex* d_fourierPtr = d_fourier + n * pieceFFTNumPixels;
		cuDoubleComplex* h_fourierPtr = h_fourier + n * pieceFFTNumPixels;

		smooth<<<dimGrid, dimBlock, 0, streams[n]>>>(d_piecePtr, d_mic, d_pieceSmoother, pieceDim, y0, x0, yDim, a, b);
		CU_CHK(cudaPeekAtLastError()); // test kernel was created correctly

		// FFT Execution
		FFT_CHK(cufftExecD2Z(plans[n], d_piecePtr, d_fourierPtr));
		CU_CHK(cudaMemcpyAsync(h_fourierPtr, d_fourierPtr, pieceFFTSize, cudaMemcpyDeviceToHost, streams[n]));
	}

	// Expand + redux

	size_t XSIZE_FOURIER = (pieceDim / 2 + 1);
	size_t YSIZE_FOURIER = pieceDim;
	size_t XSIZE_REAL = pieceDim;
	size_t YSIZE_REAL = pieceDim;

	for (size_t n = 0; n < divNumber; ++n) {
		// Wait for fft completed for this piece
		CU_CHK(cudaStreamSynchronize(streams[n]));
		CU_CHK(cudaStreamDestroy(streams[n]));
		FFT_CHK(cufftDestroy(plans[n]));

		tPost.tic();
		cuDoubleComplex val;
		double* ptrDest;
		size_t iterator;
		for (size_t i = 0; i < pieceDim; ++i) {
			for (size_t j = 0; j < pieceDim; ++j) {
				ptrDest = (double*) &psd[i * XSIZE_REAL + j];

				if (j < XSIZE_FOURIER) {
					iterator  = n * pieceFFTNumPixels + i * XSIZE_FOURIER + j;
				} else {
					iterator  = n * pieceFFTNumPixels
							+ (((YSIZE_REAL) - i) % (YSIZE_REAL))
									* XSIZE_FOURIER + ((XSIZE_REAL) - j);
				}
				val = *(h_fourier + iterator);
				double real = cuCreal(val);
				double imag = cuCimag(val);
				*ptrDest += (real * real + imag * imag) * pieceDim * pieceDim;
			}
		}
//TODO: gpu version check
//		post<<<dimGrid, dimBlock, 0, streams[n]>>>(d_fft, define_out, d_pieceSmoother, pieceDim, y0, x0, yDim, a, b);
		tPost.toc("Time to post:\t\t\t");
	}

	// Average
	t.tic();
	double idiv_Number = 1.0 / (divNumber);
	for(size_t i = 0; i < pieceDim * pieceDim; ++i)	{
		psd[i] *= idiv_Number;
	}
	t.toc("Final reduction:\t\t");
	tTotalCompute.toc("Total compute:\t\t\t");

	// Free memory
#ifdef USE_PINNED
	CU_CHK(cudaFreeHost(h_fourier));
#else
	free(h_fourier);
#endif
	CU_CHK(cudaFree(d_pieces));
	CU_CHK(cudaFree(d_fourier));
	CU_CHK(cudaFree(d_pieceSmoother));
	CU_CHK(cudaFree(d_mic));

	tTotal.toc("Total:\t\t\t\t");
}

void cudaRunGpuEstimateCTFwithInterResults(double* mic, size_t xDim, size_t yDim, double overlap, size_t pieceDim, int skipBorders, double* pieceSmoother,
			double* basePieces, double* normalizedSmoothPieces, std::complex<double>* piecesFFT, double* psd) {
	TicToc t;

	// Calculate reduced input dim (exact multiple of pieceDim, without skipBorders)
	size_t divNumberX         = std::ceil((double) xDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	size_t divNumberY         = std::ceil((double) yDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	size_t divNumber          = divNumberX * divNumberY;

	size_t inNumPixels        = divNumber * pieceDim * pieceDim;
	size_t inSize             = inNumPixels * sizeof(double);

	size_t outNumPixels       = divNumber * pieceDim * (pieceDim / 2 + 1);
	size_t outSize            = outNumPixels * sizeof(cufftDoubleComplex);

	size_t pieceNumPixels     = pieceDim * pieceDim;
	size_t pieceSize          = pieceNumPixels * sizeof(double);

	size_t pieceFFTNumPixels  = pieceDim * (pieceDim / 2 + 1);
	size_t pieceFFTSize       = pieceFFTNumPixels * sizeof(cufftDoubleComplex);

	if (divNumberX <= 0 || divNumberY <= 0) {
		std::cerr << "Error, can't split a " << xDim << "X" << yDim << " MIC into " << pieceDim << "X" << pieceDim << " pieces" << std::endl
				  << "with " << overlap << " overlap and " << skipBorders << " skip borders," << std::endl
				  << "resulted in divNumberX=" << divNumberX << ", divNumberY=" << divNumberY << std::endl;
		exit(EXIT_FAILURE);
	}

	// Host page-locked memory
	double* in;
	cuDoubleComplex* out;
	CU_CHK(cudaMallocHost((void**) &in,  inSize));
	CU_CHK(cudaMallocHost((void**) &out, outSize));

	// Device memory
	double* d_in;
	cuDoubleComplex*d_out;
	CU_CHK(cudaMalloc((void**)&d_in, inSize));
	CU_CHK(cudaMalloc((void**)&d_out, outSize));

	// CuFFT Callback
//	cufftCallbackStoreZ h_storeCallbackPtr;
//	CU_CHK(cudaMemcpyFromSymbol(&h_storeCallbackPtr, d_storeCallbackPtr, sizeof(h_storeCallbackPtr)));

	// CU FFT PLAN
	 cudaStream_t* streams = new cudaStream_t[divNumber];
	 cufftHandle*    plans = new cufftHandle[divNumber];

//	cufftHandle   plan;
//	FFT_CHK(cufftPlan2d(&plan,pieceDim, pieceDim, CUFFT_D2Z));

	// Iterate over all pieces
	size_t step = (size_t) (((1 - overlap) * pieceDim));
	for (size_t n = 0; n < divNumber; ++n) {
		// Extract piece
		size_t blocki = n / divNumberX;
		size_t blockj = n % divNumberX;
		size_t y0 = blocki * step + skipBorders * pieceDim;
		size_t x0 = blockj * step + skipBorders * pieceDim;

		// Test if the full piece is inside the micrograph
		if (y0 + pieceDim > yDim)
			y0 = yDim - pieceDim;

		if (x0 + pieceDim > xDim)
			x0 = xDim - pieceDim;

		size_t yEnd = y0 + pieceDim;
		size_t xEnd = x0 + pieceDim;

		// ComputeAvgStdev
		double avg = 0.0, stddev = 0.0;
		computePieceAvgStd(mic, pieceDim, y0, yEnd, x0, xEnd, yDim, avg, stddev);

		// Normalize and smooth
		double a, b;
		if (stddev != 0.0)
			a = stddevF / stddev;
		else
			a = 0.0;

		b = avgF - a * avg;

		size_t it = n * pieceNumPixels; // Host page-locked memory iterator
		size_t smoothIt = 0;
		for (size_t y = y0; y < yEnd; y++) {
			for (size_t x = x0; x < xEnd; x++) {
				//std::cerr << "x0 " << x0 << " y0 " << y0 << " xEnd " << xEnd << " yEnd " << yEnd << " x " << x << " y " << y << " p " << y * yDim + x << std::endl;
				//std::cerr << "it " << it << " micIt " << y * step + x << " smoothIt " << smoothIt << std::endl;
				in[it] = (mic[y * xDim + x] * a + b) * pieceSmoother[smoothIt];
				////////////////////////////
				// Copy piece
				basePieces[it] = mic[y * xDim + x];
				normalizedSmoothPieces[it] = (mic[y * xDim + x] * a + b) * pieceSmoother[smoothIt];

				it++;
				smoothIt++;
			}
		}

		double* d_inPtr           = d_in  + n * pieceNumPixels;
		cuDoubleComplex* d_outPtr = d_out + n * pieceFFTNumPixels;

		double* h_inPtr           = in  + n * pieceNumPixels;
		cuDoubleComplex* h_outPtr = out + n * pieceFFTNumPixels;

		// Execution
		CU_CHK (cudaStreamCreate(streams + n));
		FFT_CHK(cufftPlan2d(plans + n,pieceDim, pieceDim, CUFFT_D2Z));
		FFT_CHK(cufftSetStream(plans[n], streams[n]));

//		FFT_CHK(cufftXtSetCallback(plans[n], (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)NULL));

		CU_CHK(cudaMemcpyAsync(d_inPtr, h_inPtr, pieceSize, cudaMemcpyHostToDevice, streams[n]));
		FFT_CHK(cufftExecD2Z(plans[n], d_inPtr, d_outPtr));
		CU_CHK(cudaMemcpyAsync(h_outPtr, d_outPtr, pieceFFTSize, cudaMemcpyDeviceToHost, streams[n]));

//		CU_CHK(cudaMemcpy(d_inPtr, h_inPtr, pieceSize, cudaMemcpyHostToDevice));
//		FFT_CHK(cufftExecD2Z(plan, d_inPtr, d_outPtr));
//		CU_CHK(cudaDeviceSynchronize());
//		CU_CHK(cudaMemcpy(h_outPtr, d_outPtr, pieceFFTSize, cudaMemcpyDeviceToHost));
	}

	// Expand + redux
	for (size_t n = 0; n < divNumber; ++n) {
		CU_CHK(cudaStreamSynchronize(streams[n]));
		CU_CHK(cudaStreamDestroy(streams[n]));

		/////////////////////////////////////
		// Copy fft
		for (size_t i = 0; i < pieceFFTNumPixels; i++) {
			piecesFFT[n * pieceFFTNumPixels + i] = std::complex<double>(out[n * pieceFFTNumPixels + i].x, out[n * pieceFFTNumPixels + i].y);
		}

		size_t XSIZE_FOURIER = (pieceDim / 2 + 1);
		size_t YSIZE_FOURIER = pieceDim;
		size_t XSIZE_REAL = pieceDim;
		size_t YSIZE_REAL = pieceDim;

		cuDoubleComplex val;
		double* ptrDest;
		size_t iterator;
		for (size_t i = 0; i < pieceDim; ++i) {
			for (size_t j = 0; j < pieceDim; ++j) {
				ptrDest = (double*) &psd[i * XSIZE_REAL + j];

				if (j < XSIZE_FOURIER) {
					iterator  = n * pieceFFTNumPixels + i * XSIZE_FOURIER + j;
				} else {
					iterator  = n * pieceFFTNumPixels
							+ (((YSIZE_REAL) - i) % (YSIZE_REAL))
									* XSIZE_FOURIER + ((XSIZE_REAL) - j);
				}

				val = *(out + iterator);
				double real = cuCreal(val);
				double imag = cuCimag(val);
				*ptrDest += (real * real + imag * imag) * pieceDim * pieceDim;
			}
		}
	}

	// Average
	double idiv_Number = 1.0 / (divNumber);
	for(size_t i = 0; i < pieceDim * pieceDim; ++i)	{
		psd[i] *= idiv_Number;
	}

	// Free memory
	CU_CHK(cudaFreeHost(in));
	CU_CHK(cudaFreeHost(out));
	CU_CHK(cudaFree(d_in));
	CU_CHK(cudaFree(d_out));
}
