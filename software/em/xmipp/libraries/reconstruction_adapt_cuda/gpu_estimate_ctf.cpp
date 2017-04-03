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
#include <data/multidim_array.h>

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


void transposeImage(Image<double> &im);

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
				  << "div_Number : " << div_Number  << std::endl
				  << std::endl
				  << "pieceDim: " << pieceDim << std::endl
				  << "overlap:  " << overlap  << std::endl;

		std::cout << "Computing model of the micrograph" << std::endl;
		//init_progress_bar(div_Number);
	}

	// Test one piece
	int N      = 42;
	int step   = (int) ((1 - overlap) * pieceDim);
	int blocki = (N-1) / div_NumberX;
	int blockj = (N-1) % div_NumberX;

	int piecei = blocki * step;
	int piecej = blockj * step;

	// test if the full piece is inside the micrograph
	if (piecei + pieceDim > Ydim)
		piecei = Ydim - pieceDim;
	if (piecej + pieceDim > Xdim)
		piecej = Xdim - pieceDim;

	std::cout << "N     : " << N       << std::endl
			  << "step  : " << step    << std::endl
			  << "blocki: " << blocki  << std::endl
			  << "blockj: " << blockj  << std::endl
			  << "piecei: " << piecei  << std::endl
			  << "piecej: " << piecej  << std::endl;

	Image<double> piece;
	piece().initZeros(pieceDim, pieceDim);
	window2D(mic(), piece(), piecei, piecej, piecei + YSIZE(piece()) - 1, piecej + XSIZE(piece()) - 1);

	// Normalize piece
	piece().statisticsAdjust(0, 1);
	normalize_ramp(piece());

	piece.write("gpu_1_piece");
	testOnePiece(piece().data, psdPtr, pieceDim);

	psd.write(fnOut);

	std::cout << std::endl;
}

void transposeImage(Image<double> &im) {
	Matrix2D<double> m(im().ydim, im().xdim);
	memcpy(m.mdata, im().data, im().ydim * im().xdim * sizeof(double));
	memcpy(im().data, m.transpose().mdata, im().ydim * im().xdim * sizeof(double));
}
