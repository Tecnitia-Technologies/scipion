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
}

using namespace cuda_gpu_estimate_ctf_err;

#define TEST(x) __e = x;\
if (__e != 0) { \
std::cerr << "ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetErrorString(__e) << std::endl;\
exit(-1); }

#define TEST_FFT(x) __fftr = x;\
if (__fftr != 0) { \
std::cerr << "FFT ERROR, line: " << __LINE__ << " File: " << __FILE__ << " Error: " <<  cudaGetCuFFTResultString(__fftr) << std::endl;\
exit(-1); }

void cudaRunGpuEstimateCTF(double* mic, double* psd, int pieceDim,
		int div_Number, int div_NumberX, int div_NumberY) {

	int numElemsPerLine = pieceDim * pieceDim * div_NumberX;

	// Device pointers
	double *d_micLine, *d_resLine;
	cufftDoubleComplex* d_fourier; // Fourier intermediate result

	// Auxiliar host mem
	double *partialPsd;
	TEST(cudaMallocHost((void**) &partialPsd, numElemsPerLine * sizeof(double)));

	// Reservar espacio para una linea
	TEST(cudaMalloc((void**) &d_micLine, numElemsPerLine * sizeof(double)));

	// Reservar espacio auxiliar para FFT
	TEST(cudaMalloc((void**) &d_fourier, numElemsPerLine * sizeof(cufftDoubleComplex)));

	// Reservar resultado
	TEST(cudaMalloc((void**) &d_resLine, numElemsPerLine * sizeof(double)));

	for (int line = 0; line < div_NumberY; ++line) {
		// Subir a gpu la linea Imic_ptr[line * numElemsPerLine]
		TEST(cudaMemcpy(d_micLine, mic + line * numElemsPerLine, numElemsPerLine, cudaMemcpyHostToDevice));

		// Procesar linea

		// traer resultado
		TEST(cudaMemcpy(partialPsd, d_resLine, numElemsPerLine, cudaMemcpyDeviceToHost));

		// reducir resultado
	}

	// media
	// psd *= (double) 1.0 / div_Number;
	for (int i = 0; i < pieceDim * pieceDim; i++) {
		psd[i] *= (double) 1.0 / div_Number;
	}

	// Free device and auxiliar memory
	TEST(cudaFree(d_micLine));
	TEST(cudaFree(d_fourier));
	TEST(cudaFree(d_resLine));

	TEST(cudaFreeHost(partialPsd));
}

void cudaRunGpuEstimateCTFTest(double* mic, double* psd, int pieceDim,
		int div_Number, int div_NumberX, int div_NumberY, int size_x) {


	// Device pointers
	double *d_micLine, *d_resLine;
	cufftDoubleComplex* d_fourier; // Fourier intermediate result

	// Auxiliar host mem
	double *partialPsd;
	TEST(cudaMallocHost((void**) &partialPsd, pieceDim * pieceDim * sizeof(double)));

	// Reservar espacio para una linea
	TEST(cudaMalloc((void**) &d_micLine, pieceDim * pieceDim * sizeof(double)));

	// Reservar espacio auxiliar para FFT
	TEST(cudaMalloc((void**) &d_fourier, pieceDim * pieceDim * sizeof(cufftDoubleComplex)));

	// Reservar resultado
	TEST(cudaMalloc((void**) &d_resLine, pieceDim * pieceDim * sizeof(double)));



	// Subir a gpu la linea Imic_ptr[line * numElemsPerLine]
	for (int i = 0; i < pieceDim; i++) {
		TEST(cudaMemcpy(d_micLine + i * pieceDim, mic + i * size_x, pieceDim , cudaMemcpyHostToDevice));
	}



	// traer resultado
	TEST(cudaMemcpy(partialPsd, d_resLine, pieceDim * pieceDim, cudaMemcpyDeviceToHost));


	// media
	// psd *= (double) 1.0 / div_Number;
	for (int i = 0; i < pieceDim * pieceDim; i++) {
		psd[i] *= (double) 1.0 / div_Number;
	}

	// Free device and auxiliar memory
	TEST(cudaFree(d_micLine));
	TEST(cudaFree(d_fourier));
	TEST(cudaFree(d_resLine));

	TEST(cudaFreeHost(partialPsd));
}
