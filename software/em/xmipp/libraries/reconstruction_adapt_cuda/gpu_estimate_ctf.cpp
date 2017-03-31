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

#include "gpu_estimate_ctf.h"

#include <reconstruction_cuda/cuda_gpu_estimate_ctf.h>
#include <data/xmipp_image.h>
#include <data/xmipp_program.h>
#include <data/normalize.h>

// Prototypes
void constructPieceSmoother(const MultidimArray<double> &piece,
		MultidimArray<double> &pieceSmoother);

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


void ProgGpuEstimateCTF::run() {
	// Input
	Image<double> mic;
	// Result
	Image<double> psd;
	psd().initZeros(pieceDim, pieceDim);

	mic.read(fnMic);

	size_t Xdim, Ydim, Zdim, Ndim;
	mic.getDimensions(Xdim, Ydim, Zdim, Ndim);

	double *micPtr = MULTIDIM_ARRAY(mic());
	double *psdPtr = MULTIDIM_ARRAY(psd());

	// Compute the number of divisions --------------------------------------
	int div_Number = 0;
	int div_NumberX = 1, div_NumberY = 1;
	div_NumberX = CEIL((double)Xdim / (pieceDim *(1-overlap))) - 1;
	div_NumberY = CEIL((double)Ydim / (pieceDim *(1-overlap))) - 1;
	div_Number  = div_NumberX * div_NumberY;

	if (verbose) {
		std::cout << "Xdim: " << Xdim << std::endl
				  << "Ydim: " << Ydim << std::endl
				  << "Zdim: " << Zdim << std::endl
				  << "Ndim: " << Ndim << std::endl
				  << std::endl
		          << "div_NumberX: " << div_NumberX << std::endl
				  << "div_NumberY: " << div_NumberY << std::endl
			      << "div_Number : " << div_Number  << std::endl;

		std::cout << "Computing model of the micrograph" << std::endl;
		init_progress_bar(div_Number);
	}

	// Attenuate borders to avoid discontinuities
//    MultidimArray<double> pieceSmoother;
//    constructPieceSmoother(mic(), pieceSmoother);

	// Normalize image
	mic().statisticsAdjust(0, 1);
	normalize_ramp(mic());
	//mic() *= pieceSmoother;



	psd.write(fnOut);

	std::cout << std::endl;
}

/* Construct piece smoother =============================================== */
void constructPieceSmoother(const MultidimArray<double> &piece,
		MultidimArray<double> &pieceSmoother) {
	// Attenuate borders to avoid discontinuities
	pieceSmoother.resizeNoCopy(piece);
	pieceSmoother.initConstant(1);
	pieceSmoother.setXmippOrigin();
	double iHalfsize = 2.0 / YSIZE(pieceSmoother);
	const double alpha = 0.025;
	const double alpha1 = 1 - alpha;
	const double ialpha = 1.0 / alpha;
	for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother);
			i++) {
		double iFraction = fabs(i * iHalfsize);
		if (iFraction > alpha1) {
			double maskValue = 0.5
					* (1 + cos(PI * ((iFraction - 1) * ialpha + 1)));
			for (int j = STARTINGX(pieceSmoother);
					j <= FINISHINGX(pieceSmoother); j++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}

	for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother);
			j++) {
		double jFraction = fabs(j * iHalfsize);
		if (jFraction > alpha1) {
			double maskValue = 0.5
					* (1 + cos(PI * ((jFraction - 1) * ialpha + 1)));
			for (int i = STARTINGY(pieceSmoother);
					i <= FINISHINGY(pieceSmoother); i++)
				A2D_ELEM(pieceSmoother,i,j) *= maskValue;
		}
	}
}
