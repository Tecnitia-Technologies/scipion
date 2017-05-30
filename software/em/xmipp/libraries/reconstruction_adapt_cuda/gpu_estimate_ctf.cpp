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
	fnMic       = getParam("-i");
	fnOut       = getParam("-o");
    pieceDim    = getIntParam("--pieceDim");
    overlap     = getDoubleParam("--overlap");
    skipBorders = getIntParam("--skipBorders");
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
	<< "Skip borders:              " << skipBorders  << std::endl
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
    addParamsLine("  [--skipBorders <s=0>]   : Skipped pieces from each side");
}

/* Construct piece smoother =============================================== */
template <typename T>
void constructPieceSmoother(const MultidimArray<T> &piece,
		MultidimArray<T> &pieceSmoother) {
    TicToc t(false);
	// Attenuate borders to avoid discontinuities
	pieceSmoother.resizeNoCopy(piece);
    t.tic();
	pieceSmoother.initConstant(1);
    t.toc("initConstant piece smoother\t");
	pieceSmoother.setXmippOrigin();

	T iHalfsize = 2.0 / YSIZE(pieceSmoother);
	const T alpha = 0.025;
	const T alpha1 = 1 - alpha;
	const T ialpha = 1.0 / alpha;

	t.tic();

	for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother); i++) {
		T iFraction = fabs(i * iHalfsize);
		if (iFraction > alpha1) {
			T maskValue = 0.5 * (1 + cos(PI * ((iFraction - 1) * ialpha + 1)));
			for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother); j++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}
    t.toc("for1 piece smoother\t\t");

    t.tic();
	for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother); j++) {
		T jFraction = fabs(j * iHalfsize);
		if (jFraction > alpha1) {
			T maskValue = 0.5 * (1 + cos(PI * ((jFraction - 1) * ialpha + 1)));
			for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother); i++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}
    t.toc("for2 piece smoother\t\t");

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

void ProgGpuEstimateCTF::run() {
	if (pieceDim % 2 != 0) {
		std::cerr << "ERROR, pieceDim must be even" << std::endl;
		exit(EXIT_FAILURE);
	}
	TicToc t, total;

	total.tic();
	// Input
	Image<real_t> mic;
	mic.read(fnMic);
	real_t *micPtr = mic().data;

	// Result
	Image<real_t> psd;
	psd().initZeros(pieceDim, pieceDim);
	real_t *psdPtr = psd().data;

	Image<real_t> psd2;
	psd2().initZeros(pieceDim, pieceDim);
	real_t *psdPtr2 = psd2().data;

	// Compute the number of divisions --------------------------------------
	size_t Xdim, Ydim, Zdim, Ndim;
//	int div_Number;
//	int div_NumberX, div_NumberY;
 	//computeDivisions(mic, div_Number, div_NumberX, div_NumberY, Xdim, Ydim, Zdim, Ndim);
	mic.getDimensions(Xdim, Ydim, Zdim, Ndim);

 	// Attenuate borders to avoid discontinuities
	MultidimArray<real_t> piece(pieceDim, pieceDim);
    t.tic();
    MultidimArray<real_t> pieceSmoother;
    constructPieceSmoother(piece, pieceSmoother);
    t.toc("piece smoother\t\t\t");

    // CU FFT
//	cudaRunGpuEstimateCTF(micPtr, Xdim, Ydim, overlap, pieceDim, 0, pieceSmoother.data, psdPtr, verbose);
	CudaPsdCalculator psdCalc(32, overlap, pieceDim, skipBorders, true, pieceSmoother.data);

	for (int var = 0; var < 10; ++var) {
		psdCalc.calculatePsd(micPtr, Xdim, Ydim, psdPtr);
		std::cout << std::endl;
	}
	psdCalc.calculatePsd(micPtr, Xdim, Ydim, psdPtr2);
	std::cout << std::endl;

	total.toc("Total\t\t\t\t");
	psd2.write(fnOut);
}
