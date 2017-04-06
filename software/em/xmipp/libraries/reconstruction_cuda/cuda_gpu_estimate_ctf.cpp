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



void gpuFFT(double* input, std::complex<double>* f, int pieceDim) {
	cufftDoubleComplex *fourier = (cufftDoubleComplex*) f;

	size_t inputSize     = pieceDim * pieceDim * sizeof(double);
	size_t transformSize = pieceDim * (pieceDim / 2 + 1) * sizeof(cufftDoubleComplex);
	// Half of elements because of Hermitian Symmetry of Real value FT

	// Device pointers
	double*             d_input;
	cufftDoubleComplex* d_fourier;

	// Device memory allocation
	CU_CHK(cudaMalloc((void**) &d_input,   inputSize));
	CU_CHK(cudaMalloc((void**) &d_fourier, transformSize));

	// Offload to device
	CU_CHK(cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice));

	// Fourier setup
	cufftHandle plan;
	FFT_CHK(cufftPlan2d(&plan, pieceDim, pieceDim, CUFFT_D2Z));

	// Fourier execution
	FFT_CHK(cufftExecD2Z(plan, d_input, d_fourier));
	CU_CHK(cudaDeviceSynchronize());

	// Read result from device
	CU_CHK(cudaMemcpy(fourier, d_fourier, transformSize, cudaMemcpyDeviceToHost));

	// Free device memory
	CU_CHK(cudaFree(d_input));
	CU_CHK(cudaFree(d_fourier));

	FFT_CHK(cufftDestroy(plan));
}
