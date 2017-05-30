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
#include <cuda.h>
#include <cufft.h>

#include <cufftXt.h>
#include <cmath>

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const int K_SMOOTH_BLOCK_SIZE_X = 32;
const int K_SMOOTH_BLOCK_SIZE_Y = 32;

__global__ void smooth(double* mic, double* pieceSmoother, double* out, size_t pieceDim, size_t x0, size_t yDim, double a, double b) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	size_t it    = y * pieceDim + x;
	size_t micIt = y * yDim + x0 + x;
	out[it]      = (mic[micIt] * a + b) * pieceSmoother[it];
}

__global__ void naivePost(cuDoubleComplex* fft, double* out, size_t XSIZE_FOURIER, size_t XSIZE_REAL, size_t YSIZE_REAL, double iSize) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	cuDoubleComplex val;
	double* ptrDest;
	size_t iterator;
	ptrDest = out + i * XSIZE_REAL + j;

	if (j < XSIZE_FOURIER) {
		iterator  = i * XSIZE_FOURIER + j;
	} else {
		// Using {x & (n-1)} as {x % (n)}, only works if n is POW of 2
		// TODO error if n not pow of 2, or use different kernel
		iterator  = (((YSIZE_REAL) - i) & (YSIZE_REAL-1)) * XSIZE_FOURIER + ((XSIZE_REAL) - j);
	}
	val = fft[iterator];
	double real = cuCreal(val) * iSize;
	double imag = cuCimag(val) * iSize;
	*ptrDest = (real * real + imag * imag);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class AvgStd {

	size_t pieceDim;
	size_t numSubpiecesX;
	size_t len;
	double* avgSubpieces;
	double* stdSubpieces;

public:

	AvgStd(double* mic, size_t pieceDim, size_t xDim, double overlap,
			size_t divNumberX, size_t divNumberY, size_t skipBorders = 0) :
			pieceDim(pieceDim) {

		if (overlap != 0.5) {
			std::cerr << "Error, only overlap of 0.5 allowed" << std::endl;
			exit(EXIT_FAILURE);
		}
		bool isPow2 = true;
		if ((pieceDim & (pieceDim - 1)) != 0) {
			std::cerr << std::endl;
			std::cerr << "**************************************************************" << std::endl;
			std::cerr << "WARNING, pieceDim should be power of 2 for optimal performance" << std::endl;
			std::cerr << "**************************************************************" << std::endl;
			std::cerr << std::endl;
			isPow2 = false;
		}

		size_t initPos = skipBorders * pieceDim;

		len = pieceDim * pieceDim;

		numSubpiecesX = divNumberX + 1;
		size_t numSubpiecesY = divNumberY + 1;

		avgSubpieces = new double[numSubpiecesX * numSubpiecesY];
		stdSubpieces = new double[numSubpiecesX * numSubpiecesY];

		std::fill_n(avgSubpieces, numSubpiecesX * numSubpiecesY, 0.0);
		std::fill_n(stdSubpieces, numSubpiecesX * numSubpiecesY, 0.0);

		size_t xTotalLim = numSubpiecesX * (pieceDim * (1-overlap));
		size_t yTotalLim = numSubpiecesY * (pieceDim * (1-overlap));

		double val;
		int subPieceX = -1, subPieceY = -1;
		size_t half = pieceDim / 2;
		if (isPow2) {
			half--;
			for (size_t y = initPos; y < yTotalLim; y++) {
				if ((y & half) == 0) {
					subPieceY++;
				}
				subPieceX = -1;
				for (size_t x = initPos; x < xTotalLim; x++) {
					if ((x & half) == 0) {
						subPieceX++;
					}
					val = mic[y * xDim + x];

					avgSubpieces[subPieceY * numSubpiecesX + subPieceX] += val;
					stdSubpieces[subPieceY * numSubpiecesX + subPieceX] += val * val;
				}
			}
		} else {
			for (size_t y = initPos; y < yTotalLim; y++) {
				if ((y % half) == 0) {
					subPieceY++;
				}
				subPieceX = -1;
				for (size_t x = initPos; x < xTotalLim; x++) {
					if ((x % half) == 0) {
						subPieceX++;
					}
					val = mic[y * xDim + x];

					avgSubpieces[subPieceY * numSubpiecesX + subPieceX] += val;
					stdSubpieces[subPieceY * numSubpiecesX + subPieceX] += val * val;
				}
			}
		}
	}

	~AvgStd() {
		delete [] avgSubpieces;
		delete [] stdSubpieces;
	}

	void getAvgStd(size_t blocki, size_t blockj, double& avg, 	double& stddev) {
		avg =     avgSubpieces[blocki       * numSubpiecesX + blockj]
				+ avgSubpieces[blocki       * numSubpiecesX + blockj + 1]
				+ avgSubpieces[(blocki + 1) * numSubpiecesX + blockj]
				+ avgSubpieces[(blocki + 1) * numSubpiecesX + blockj + 1];
		stddev =  stdSubpieces[blocki       * numSubpiecesX + blockj]
				+ stdSubpieces[blocki       * numSubpiecesX + blockj + 1]
				+ stdSubpieces[(blocki + 1) * numSubpiecesX + blockj]
				+ stdSubpieces[(blocki + 1) * numSubpiecesX + blockj + 1];
		avg /= len;
		stddev = stddev / len - avg * avg;
		stddev *= len / (len - 1);
		// Foreseeing numerical instabilities
		stddev = std::sqrt(std::fabs(stddev));
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CudaPsdCalculator::firstExecutionConfiguration(size_t xDim, size_t yDim) {
	TicToc t(true && verbose);

	this->xDim              = xDim;
	this->yDim              = yDim;

	this->divNumberX        = std::floor((double) xDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	this->divNumberY        = std::floor((double) yDim / (pieceDim * (1-overlap))) - 1 - 2 * skipBorders;
	this->divNumber         = divNumberX * divNumberY;

	this->startingX         = skipBorders;
	this->startingY         = skipBorders;

	this->piecesPerChunk    = divNumberX;
	this->numChunks         = divNumberY;

	this->inNumPixels       = piecesPerChunk * pieceDim * pieceDim;
	this->inSize            = inNumPixels * sizeof(double);

	this->outNumPixels      = piecesPerChunk * pieceDim * (pieceDim / 2 + 1);
	this->outSize           = outNumPixels * sizeof(cufftDoubleComplex);

	this->pieceNumPixels    = pieceDim * pieceDim;
	this->pieceSize         = pieceNumPixels * sizeof(double);

	this->pieceFFTNumPixels = pieceDim * (pieceDim / 2 + 1);
	this->pieceFFTSize      = pieceFFTNumPixels * sizeof(cufftDoubleComplex);

	this->XSIZE_FOURIER     = (pieceDim / 2 + 1);
	this->XSIZE_REAL        = pieceDim;
	this->YSIZE_REAL        = pieceDim;
	this->iSize             = 1.0 / (pieceDim * pieceDim);

	this->step = (size_t) (((1 - overlap) * pieceDim));

	if (verbose) {
		std::cout << std::endl;
		std::cout << "xDim: " << xDim << std::endl
				  << "yDim: " << yDim << std::endl
				  << std::endl
				  << "divNumberX: " << divNumberX << std::endl
				  << "divNumberY: " << divNumberY << std::endl
				  << "divNumber : " << divNumber  << std::endl
				  << std::endl
				  << "piecesPerChunk : " << piecesPerChunk << std::endl
				  << std::endl
				  << "pieceDim: " << pieceDim << std::endl
				  << "overlap:  " << overlap  << std::endl;
		std::cout << std::endl;
	}

	if (divNumberX <= 0 || divNumberY <= 0) {
		std::cerr << "ERROR, can't split a " << xDim << "X" << yDim << " MIC into " << pieceDim << "X" << pieceDim << " pieces" << std::endl
				  << "with " << overlap << " overlap and " << skipBorders << " skip borders," << std::endl
				  << "resulted in divNumberX=" << divNumberX << ", divNumberY=" << divNumberY << std::endl;
		exit(EXIT_FAILURE);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Host page-locked memory
	t.tic();
	CU_CHK(cudaMallocHost((void**) &h_pieces, inSize));
	t.toc("Time to pinned malloc:\t\t");

	// Device memory
	t.tic();
	CU_CHK(cudaMalloc((void**)&d_mic, xDim * pieceDim * sizeof(double)));
	CU_CHK(cudaMalloc((void**)&d_pieces,  inSize));
	CU_CHK(cudaMalloc((void**)&d_fourier, outSize));
	CU_CHK(cudaMalloc((void**)&d_pieceSmoother, pieceSize));
	t.toc("Time to cuda malloc:\t\t");

	t.tic();
	createPieceSmoother();
	t.toc("Time to create piece smoother:\t");

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Smooth Kernel config
	dimBlockSmooth = dim3(K_SMOOTH_BLOCK_SIZE_X, K_SMOOTH_BLOCK_SIZE_Y);
	// Calculate number of grids needed for dimBlock,
	// wich is pieceDim % block == 0 ? pieceDim / block : pieceDim / block + 1
	// in a hacky way
	int k_grid_size_x = (pieceDim + K_SMOOTH_BLOCK_SIZE_X - 1) / K_SMOOTH_BLOCK_SIZE_X;
	int k_grid_size_y = (pieceDim + K_SMOOTH_BLOCK_SIZE_Y - 1) / K_SMOOTH_BLOCK_SIZE_Y;
	dimGridSmooth = dim3(k_grid_size_x, k_grid_size_y);

	// CU FFT PLAN
	streams = new cudaStream_t[piecesPerChunk];
	plans   = new cufftHandle[piecesPerChunk];

	t.tic();
	for (size_t n = 0; n < piecesPerChunk; ++n) {
		CU_CHK(cudaStreamCreate(streams + n));
		FFT_CHK(cufftPlan2d(plans + n, pieceDim, pieceDim, CUFFT_D2Z));
		FFT_CHK(cufftSetStream(plans[n], streams[n]));
	}
	t.toc("Time to plans and streams:\t");

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}

void CudaPsdCalculator::createPieceSmoother() {
	CU_CHK(cudaMemcpy(d_pieceSmoother, pieceSmoother, pieceSize, cudaMemcpyHostToDevice));
}

void CudaPsdCalculator::calculatePsd(double* mic, size_t xDim, size_t yDim, double* psd) {
	TicToc t(true && verbose), tAvg(false && verbose), tPost(true && verbose), tTotal(true && verbose), tTotalCompute(true && verbose);

	tTotal.tic();
	if (firstExecution) {
		firstExecutionConfiguration(xDim, yDim);
		firstExecution = false;
	} else {
		// Check if same size than previous mic
		if (this->xDim != xDim || this->yDim != yDim) {
			std::cerr << "ERROR, size of current image (" << xDim << ", " << yDim << ")" << std::endl
					  << "is different from previous image (" << this->xDim << ", " << this->yDim << ")"  << std::endl
					  << "Use different instances of CudaPsdCalculator for each MIC size";
			exit(EXIT_FAILURE);
		}
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Create avgStd structure
	t.tic();
	AvgStd avgStd(mic, pieceDim, xDim, overlap, divNumberX, divNumberY);
	t.toc("Time to subpiece avg:\t\t");

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	tTotalCompute.tic();
	// Iterate over all pieces

	size_t subpieceNumAbsolute = 0;

	for (size_t chunk = 0; chunk < numChunks; chunk++) {
		// Copy next
		size_t sizeOfHostChunk = xDim * (size_t) pieceDim * overlap;
		double* h_mic          = mic + (chunk + startingY) * sizeOfHostChunk;
		CU_CHK(cudaMemcpy(d_mic, h_mic, xDim * pieceDim * sizeof(double), cudaMemcpyHostToDevice));

		for (size_t pieceNumChunk = 0;
				pieceNumChunk < piecesPerChunk && subpieceNumAbsolute < divNumber;
				++pieceNumChunk, subpieceNumAbsolute++) {
			tAvg.tic();
			// Extract piece
			size_t blocki = subpieceNumAbsolute / divNumberX;
			size_t blockj = subpieceNumAbsolute % divNumberX;
			size_t x0 = blockj * step + skipBorders * pieceDim;

			// ComputeAvgStdev
			const double avgF = 0.0, stddevF = 1.0;
			double avg = 0.0, stddev = 0.0;

			avgStd.getAvgStd(blocki, blockj, avg, stddev);

			double a, b; // Norm params
			if (stddev != 0.0)
				a = stddevF / stddev;
			else
				a = 0.0;

			b = avgF - a * avg;
			tAvg.toc("Time to avg std:\t\t");

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			// Aux pointers
			double* d_piecePtr            = d_pieces  + pieceNumChunk * pieceNumPixels;
			cuDoubleComplex* d_fourierPtr = d_fourier + pieceNumChunk * pieceFFTNumPixels;
			double* h_rPtr                = h_pieces  + pieceNumChunk * pieceNumPixels;

			// Normalize and smooth
			smooth<<<dimGridSmooth, dimBlockSmooth, 0, streams[pieceNumChunk]>>>(d_mic, d_pieceSmoother, d_piecePtr, pieceDim, x0, yDim, a, b);
			CU_CHK(cudaPeekAtLastError()); // test kernel was created correctly

			// FFT Execution
			FFT_CHK(cufftExecD2Z(plans[pieceNumChunk], d_piecePtr, d_fourierPtr));

			// Post process (expansion)
			naivePost<<<dimGridSmooth, dimBlockSmooth, 0, streams[pieceNumChunk]>>>(d_fourierPtr, d_piecePtr, XSIZE_FOURIER, XSIZE_REAL, YSIZE_REAL, iSize);
			CU_CHK(cudaPeekAtLastError()); // test kernel was created correctly

			// Read result
			CU_CHK(cudaMemcpyAsync(h_rPtr, d_piecePtr, pieceSize, cudaMemcpyDeviceToHost, streams[pieceNumChunk]));
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Sum reduction of FFT of every piece
		tPost.tic();
		size_t absPos = chunk * piecesPerChunk;
		for (size_t n = 0; n < piecesPerChunk && absPos < divNumber; ++n, absPos++) {
			// Wait for fft completed for this piece
			CU_CHK(cudaStreamSynchronize(streams[n]));

			for (size_t i = 0; i < pieceDim; ++i) {
				for (size_t j = 0; j < pieceDim; ++j) {
					psd[i * pieceDim + j] += h_pieces[n * pieceNumPixels + i * pieceDim + j];
				}
			}
		}
		tPost.toc("Time to post:\t\t\t");
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Average
	t.tic();
	double idiv_Number = 1.0 / (divNumber) * pieceDim * pieceDim;
	for(size_t i = 0; i < pieceDim * pieceDim; ++i)	{
		psd[i] *= idiv_Number;
	}
	t.toc("Final reduction:\t\t");
	tTotalCompute.toc("Total compute:\t\t\t");
	tTotal.toc("Total\t\t\t\t");
}
