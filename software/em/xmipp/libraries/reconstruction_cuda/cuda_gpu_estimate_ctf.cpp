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

//__device__ void CB_ConvolveAndStoreTransposedC(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPointer) {
//    double real = cuCreal(element);
//    double imag = cuCimag(element);
//    ((double*)dataOut)[offset] = real * real + imag * imag;
//}
//
//__device__ cufftCallbackStoreZ d_storeCallbackPtr = CB_ConvolveAndStoreTransposedC;

// Piece normalization values
const double avgF = 0.0, stddevF = 1.0;

void cudaRunGpuEstimateCTF(double* mic, size_t xDim, size_t yDim, double overlap, size_t pieceDim, int skipBorders, double* pieceSmoother, double* psd) {
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
//	in  = (double*) malloc(inSize);
//	out = (cuDoubleComplex*) malloc(outSize);

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
				it++;
				smoothIt++;
			}
		}

		double* inPtr           = d_in  + n * pieceNumPixels;
		cuDoubleComplex* outPtr = d_out + n * pieceFFTNumPixels;

//		 Execution
//		CU_CHK (cudaStreamCreate(streams + n));
//		FFT_CHK(cufftPlan2d(plans + n,pieceDim, pieceDim, CUFFT_D2Z));
//		FFT_CHK(cufftSetStream(plans[n], streams[n]));

//		FFT_CHK(cufftXtSetCallback(plans[n], (void **)&h_storeCallbackPtr, CUFFT_CB_ST_COMPLEX_DOUBLE, (void **)NULL));

//		CU_CHK(cudaMemcpyAsync(d_in, in, pieceSize, cudaMemcpyHostToDevice, streams[n]));
//		FFT_CHK(cufftExecD2Z(plans[n], inPtr, outPtr));
//		CU_CHK(cudaMemcpyAsync(out, d_out, pieceFFTSize, cudaMemcpyDeviceToHost, streams[n]));
	}

	// Expand + redux
	for (size_t n = 0; n < divNumber; ++n) {
//		CU_CHK(cudaStreamSynchronize(streams[n]));
//		CU_CHK(cudaStreamDestroy(streams[n]));

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

//				if (iterator >= outNumPixels) {
//					std::cerr << "i " << i << " j " << j << " " << out << " " << std::endl;
//					std::cerr << "n " << n << " it " << iterator << std::endl;
//					std::cerr << "outNumPixels " << outNumPixels << std::endl;
//				}

				val = *(out + iterator);
				double real = cuCreal(val);
				double imag = cuCimag(val);

				//*ptrDest += real * real + imag * imag;
				*ptrDest += in[n * pieceFFTNumPixels
							+ i * pieceDim + j];
			}
		}
	}

	// Free memory
	CU_CHK(cudaFreeHost(in));
	CU_CHK(cudaFreeHost(out));
	CU_CHK(cudaFree(d_in));
	CU_CHK(cudaFree(d_out));
	//free(in);
}
