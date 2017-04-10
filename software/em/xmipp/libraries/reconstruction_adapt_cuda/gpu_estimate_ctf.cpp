/***************************************************************************
 *
 * Authors:     Miguel Ascanio Gómez 	(miguel.ascanio@tecnitia.com)
 * 				Alberto Casas Ortiz 	(alberto.casas@tecnitia.com)
 * 				David Gómez Blanco 		(david.gomez@tecnitia.com)
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

#include "gpu_estimate_ctf.h"

#include <reconstruction_cuda/cuda_gpu_estimate_ctf.h>
#include <data/xmipp_fftw.h>
#include <data/xmipp_image.h>
#include <data/xmipp_program.h>
#include <data/multidim_array.h>

#include <TicTocHeaderOnly.h>

// Read arguments ==========================================================
void ProgGpuEstimateCTF::readParams()
{
	fnMic = getParam("-i");
	fnOut = getParam("-o");
    pieceDim = getIntParam("--pieceDim");
    overlap = getDoubleParam("--overlap");
}

// Show ====================================================================
void ProgGpuEstimateCTF::show()
{
	if (verbose==0)
		return;
    std::cout
	<< "Input micrograph:          " << fnMic    << std::endl
	<< "Piece dim:                 " << pieceDim << std::endl
	<< "Piece overlap:             " << overlap  << std::endl
	;
}

// usage ===================================================================
void ProgGpuEstimateCTF::defineParams()
{
    addUsageLine("Estimate Xmipp CTF model from micrograph with CUDA in GPU");
    addParamsLine("   -i <micrograph>        : Input micrograph");
    addParamsLine("   -o <micrograph>        : Output psd");
    addParamsLine("  [--pieceDim <d=512>]    : Size of the piece");
    addParamsLine("  [--overlap <o=0.5>]     : Overlap (0=no overlap, 1=full overlap)");
}

/* Construct piece smoother =============================================== */
template <typename T>
void constructPieceSmoother(const MultidimArray<T> &piece,
		MultidimArray<T> &pieceSmoother) {
	// Attenuate borders to avoid discontinuities
	pieceSmoother.resizeNoCopy(piece);
	pieceSmoother.initConstant(1);
	pieceSmoother.setXmippOrigin();
	T iHalfsize = 2.0 / YSIZE(pieceSmoother);
	const T alpha = 0.025;
	const T alpha1 = 1 - alpha;
	const T ialpha = 1.0 / alpha;
	for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother); i++) {
		T iFraction = fabs(i * iHalfsize);
		if (iFraction > alpha1) {
			T maskValue = 0.5 * (1 + cos(PI * ((iFraction - 1) * ialpha + 1)));
			for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother); j++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}

	for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother); j++) {
		T jFraction = fabs(j * iHalfsize);
		if (jFraction > alpha1) {
			T maskValue = 0.5 * (1 + cos(PI * ((jFraction - 1) * ialpha + 1)));
			for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother); i++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}

	STARTINGX(pieceSmoother) = STARTINGY(pieceSmoother) = 0;
}

template <typename T>
void ProgGpuEstimateCTF::extractPiece(const MultidimArray<T>& mic, int N,
		int div_NumberX, size_t Ydim, size_t Xdim, MultidimArray<T>& piece) {

	int step = (int) (((1 - overlap) * pieceDim));
	int blocki = (N - 1) / div_NumberX;
	int blockj = (N - 1) % div_NumberX;
	int piecei = blocki * step;
	int piecej = blockj * step;
	// test if the full piece is inside the micrograph
	if (piecei + pieceDim > Ydim)
		piecei = Ydim - pieceDim;

	if (piecej + pieceDim > Xdim)
		piecej = Xdim - pieceDim;

	piece(pieceDim, pieceDim);
	window2D(mic, piece, piecei, piecej, piecei + YSIZE(piece) - 1,
			piecej + XSIZE(piece) - 1);
}

template <typename T>
void ProgGpuEstimateCTF::computeDivisions(const Image<T>& mic,
		int& div_Number, int& div_NumberX, int& div_NumberY,
		size_t& Xdim, size_t& Ydim,	size_t& Zdim, size_t& Ndim) {
	mic.getDimensions(Xdim, Ydim, Zdim, Ndim);

	div_NumberX = CEIL((double)Xdim / (pieceDim *(1-overlap))) - 1;
	div_NumberY = CEIL((double)Ydim / (pieceDim *(1-overlap))) - 1;
	div_Number = div_NumberX * div_NumberY;

	if (verbose) {
		std::cout << "Xdim: " << Xdim << std::endl
				  << "Ydim: " << Ydim << std::endl
				  << "Zdim: " << Zdim << std::endl
				  << "Ndim: " << Ndim << std::endl
				  << std::endl
				  << "div_NumberX: " << div_NumberX << std::endl
				  << "div_NumberY: " << div_NumberY << std::endl
				  << "div_Number : " << div_Number << std::endl
				  << std::endl
				  << "pieceDim: " << pieceDim << std::endl
				  << "overlap:  " << overlap  << std::endl;
	}
}

void ProgGpuEstimateCTF::run() {
	// Input
	Image<real_t> mic;
	mic.read(fnMic);
	real_t *micPtr = mic().data;
	// Result
	Image<real_t> psd;
	psd().initZeros(pieceDim, pieceDim);
	real_t *psdPtr = psd().data;


	// Compute the number of divisions --------------------------------------
	size_t Xdim, Ydim, Zdim, Ndim;
	int div_Number;
	int div_NumberX, div_NumberY;
 	computeDivisions(mic, div_Number, div_NumberX, div_NumberY, Xdim, Ydim, Zdim, Ndim);

 	// Attenuate borders to avoid discontinuities
	MultidimArray<real_t> piece(pieceDim, pieceDim);
    MultidimArray<real_t> pieceSmoother;
    constructPieceSmoother(piece, pieceSmoother);


	// Calculate reduced input dim (exact multiple of pieceDim, without skipBorders)

	size_t inNumPixels   = div_Number * pieceDim * pieceDim;
	size_t inSize        = inNumPixels * sizeof(double);

	size_t outNumPixels  = pieceDim * pieceDim;
	size_t outSize       = outNumPixels * sizeof(double);

	double* test = (double*) malloc(inSize);

	cudaRunGpuEstimateCTF(micPtr, Xdim, Ydim, overlap, pieceDim, 0, pieceSmoother.data, test);

// 	for(int N = 1; N <= div_Number; N++) {
//		// Extract piece
//		extractPiece(mic.data, N, div_NumberX, Ydim, Xdim, piece);
//		// Normalize piece
//
//		piece.statisticsAdjust(0, 1);
//		STARTINGX(piece) = STARTINGY(piece) = 0;
//		piece *= pieceSmoother;
//
//		size_t it = (N-1) * pieceDim * pieceDim;
//		for (size_t i = 0; i < pieceDim * pieceDim; i++) {
//			if (std::abs(piece.data[i] - test[it]) > 10e-12) {
//				std::cout << "piece: " << N << " i " << i << ", CPU: " << piece.data[i] << " GPU: " << test[it] << std::endl;
//			}
//			it++;
//		}
// 	}

	return;


// 	for(int N = 1; N <= div_Number; N++) {
// 		TicToc t;
// 		t.tic();
//		// Extract piece
//		extractPiece(mic.data, N, div_NumberX, Ydim, Xdim, piece);
//		// Normalize piece
//
//		piece.statisticsAdjust(0, 1);
//		STARTINGX(piece) = STARTINGY(piece) = 0;
//		piece *= pieceSmoother;
//		t.toc();
//		std::cout << t << std::endl;
//
//		// Test fourier
//		Image<real_t> fourierMagCPU;
//		Image<real_t> fourierMagGPU;
//
//		fourierMagCPU().initZeros(pieceDim, pieceDim);
//		fourierMagGPU().initZeros(pieceDim, pieceDim);
//
//		FourierTransformer transformer;
//	//	transformer.setNormalizationSign(0);
//
//		// GPU FFT
//		complex_t *fourierGPUptr = (complex_t*) malloc(pieceDim * (pieceDim / 2 + 1) * sizeof(complex_t));
//		gpuFFT(piece.data, fourierGPUptr, pieceDim);
//
//		// CPU FFT
//		transformer.setReal(piece);
//		transformer.Transform(-1); // FFTW_FORWARD
//		complex_t *fourierCPUptr = transformer.fFourier.data;
//
//		// Normalize FFT
//		real_t isize = 1.0 / (pieceDim * pieceDim);
//		for (int i = 0; i < pieceDim * (pieceDim / 2 + 1); i++) {
//			fourierGPUptr[i].real(fourierGPUptr[i].real() * isize);
//			fourierGPUptr[i].imag(fourierGPUptr[i].imag() * isize);
//		}
//
//		// Check FFT
//		for (int i = 0; i < pieceDim * (pieceDim / 2 + 1); i++) {
//			if (std::abs(fourierGPUptr[i] - fourierCPUptr[i]) > 10e-12) {
//				std::cout << i << ", CPU: " << fourierCPUptr[i] << " GPU: " << fourierGPUptr[i] << std::endl;
//			}
//		}
//
//		fourierMagCPU.write("cpu_mag");
//		fourierMagGPU.write("gpu_mag");
// 	}
}
