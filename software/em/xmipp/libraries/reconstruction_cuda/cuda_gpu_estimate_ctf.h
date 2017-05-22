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

#ifndef __CUDA_GPU_ESTIMATE_CTF__
#define __CUDA_GPU_ESTIMATE_CTF__

#define USE_PINNED

#include <cuda.h>
#include <cufft.h>

#include <cufftXt.h>

class CudaPsdCalculator {

	/* Psd calculator configuration ********************************************************************************************/
	size_t piecesPerChunk;
	size_t numChunks;

	size_t xDim;
	size_t yDim;

	double overlap;
	size_t pieceDim;

	int skipBorders;
	bool verbose;

	bool firstExecution;

	/* Size variables **********************************************************************************************************/
	size_t divNumberX;
	size_t divNumberY;
	size_t divNumber;

	size_t startingX;
	size_t startingY;

	size_t inNumPixels;
	size_t inSize;

	size_t outNumPixels;
	size_t outSize;

	size_t pieceNumPixels;
	size_t pieceSize;

	size_t pieceFFTNumPixels;
	size_t pieceFFTSize;

	size_t XSIZE_FOURIER;
	size_t XSIZE_REAL;
	size_t YSIZE_REAL;
	double iSize;

	size_t step;

	/* GPU and host memory addresses *******************************************************************************************/
	// Host page-locked memory
	double* h_pieces;
	double* pieceSmoother;

	// Device memory
	double* d_mic;
	double* d_pieces;
	double* d_pieceSmoother;
	cuDoubleComplex* d_fourier;


	/* GPU kernel sizes ********************************************************************************************************/
	dim3 dimBlockSmooth;
	dim3 dimGridSmooth;
	cudaStream_t* streams;
	cufftHandle* plans;



	/* GPU kernel sizes ********************************************************************************************************/

	// Lazy init of variales and memory allocation
	void firstExecutionConfiguration(size_t xDim, size_t yDim);

	void createPieceSmoother();

public:

	CudaPsdCalculator(size_t chunkSize, double overlap, size_t pieceDim, int skipBorders, bool verbose, double* pieceSmoother) :
		overlap(overlap), pieceDim(pieceDim), skipBorders(skipBorders), verbose(verbose), pieceSmoother(pieceSmoother), firstExecution(true) {

	}

	virtual ~CudaPsdCalculator() {
		for (size_t n = 0; n < piecesPerChunk; ++n) {
			cudaStreamDestroy(streams[n]);
			cufftDestroy(plans[n]);
		}

		delete [] streams;
		delete [] plans;

		// Free memory
		cudaFreeHost(h_pieces);
		cudaFree(d_pieces);
		cudaFree(d_fourier);
		cudaFree(d_pieceSmoother);
		cudaFree(d_mic);
	}

	void calculatePsd(double* mic, size_t xDim, size_t yDim, double* psd);

//	void calculatePsd(const Image<double>& mic, Image<double>& psd) {
//		calculatePsd(mic.data.data, mic.data.xdim, mic.data.ydim, psd.data.data);
//	}
//
//	void calculatePsd(const MultidimArray<double>& mic, MultidimArray<double>& psd) {
//		calculatePsd(mic.data, mic.xdim, mic.data.ydim, psd.data);
//	}

};

//void cudaRunGpuEstimateCTF(double* mic, size_t xDim, size_t yDim, double overlap, size_t pieceDim, int skipBorders, double* pieceSmoother, double* psd, bool verbose);
#endif
