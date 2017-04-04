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

void ProgGpuEstimateCTF::toMagnitudeMatrix(std::complex<double>* f, double* mag) {
	// CPU Magnitude
	int fourierPos = 0;
	for (int i = 0; i < pieceDim; i++) {
		for (int j = i; j < pieceDim; j++) {
			double d = std::abs(f[i]);
			mag[i * pieceDim + j] = d * d * pieceDim * pieceDim;
			fourierPos++;
		}
	}
}

Image<double> ProgGpuEstimateCTF::extractPiece(const Image<double>& mic, int N,
		int div_NumberX, size_t Ydim, size_t Xdim) {

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

//	std::cout << "N     : " << N      << std::endl
//			  << "step  : " << step   << std::endl
//			  << "blocki: " << blocki << std::endl
//			  << "blockj: " << blockj << std::endl
//			  << "piecei: " << piecei << std::endl
//			  << "piecej: " << piecej << std::endl;

	Image<double> piece;
	piece().initZeros(pieceDim, pieceDim);
	window2D(mic(), piece(), piecei, piecej, piecei + YSIZE(piece()) - 1,
			piecej + XSIZE(piece()) - 1);
	return piece;
}

void ProgGpuEstimateCTF::computeDivisions(const Image<double>& mic,
		int& div_Number, int& div_NumberX, int& div_NumberY,
		size_t& Xdim, size_t& Ydim,	size_t& Zdim, size_t& Ndim) {
	div_Number = 0;
	div_NumberX = 1, div_NumberY = 1;

	mic.getDimensions(Xdim, Ydim, Zdim, Ndim);

	div_NumberX = CEIL((double)Xdim / (pieceDim *(1-overlap))) - 1;
	div_NumberY = CEIL((double)Ydim / (pieceDim *(1-overlap))) - 1;
	div_Number = div_NumberX * div_NumberY;

	if (verbose) {
		std::cout << "Xdim: " << Xdim << std::endl
				  << "Ydim: " << Ydim << std::endl
				  << "Zdim: " << Zdim << std::endl
				  << "Ndim: "	<< Ndim << std::endl
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
	Image<double> mic;
	double *micPtr = MULTIDIM_ARRAY(mic());
	mic.read(fnMic);
	// Result
	Image<double> psd;
	double *psdPtr = MULTIDIM_ARRAY(psd());
	psd().initZeros(pieceDim, pieceDim);

	// Compute the number of divisions --------------------------------------
	size_t Xdim, Ydim, Zdim, Ndim;
	int div_Number;
	int div_NumberX, div_NumberY;
 	computeDivisions(mic, div_Number, div_NumberX, div_NumberY, Xdim, Ydim, Zdim, Ndim);
	
 	for(int N = 1; N <= div_Number; N++) {
		// Extract piece
		Image<double> piece = extractPiece(mic, N, div_NumberX, Ydim, Xdim);
		// Normalize piece
		piece().statisticsAdjust(0, 1);
		normalize_ramp(piece());

		//piece.write("gpu_1_piece");

		// Test fourier
		Image<double> fourierMagCPU;
		Image<double> fourierMagGPU;

		fourierMagCPU().initZeros(pieceDim, pieceDim);
		fourierMagGPU().initZeros(pieceDim, pieceDim);

		FourierTransformer transformer;
	//	transformer.setNormalizationSign(0);

		// GPU FFT
		std::complex<double> *fourierGPUptr = (std::complex<double>*) malloc(pieceDim * (pieceDim / 2 + 1) * sizeof(std::complex<double>));
		gpuFFT(piece().data, fourierGPUptr, pieceDim);

		// CPU FFT
		transformer.setReal(piece());
		transformer.Transform(-1); // FFTW_FORWARD
		std::complex<double> *fourierCPUptr = transformer.fFourier.data;

		// Normalize FFT
		double isize = 1.0 / (pieceDim * pieceDim);
		for (int i = 0; i < pieceDim * (pieceDim / 2 + 1); i++) {
			fourierGPUptr[i].real(fourierGPUptr[i].real() * isize);
			fourierGPUptr[i].imag(fourierGPUptr[i].imag() * isize);
		}

		// Check FFT
		for (int i = 0; i < pieceDim * (pieceDim / 2 + 1); i++) {
			if (std::abs(fourierGPUptr[i] - fourierCPUptr[i]) > 10e-12) {
				std::cout << i << ", CPU: " << fourierCPUptr[i] << " GPU: " << fourierGPUptr[i] << std::endl;
			}
		}

		toMagnitudeMatrix(fourierCPUptr, fourierMagCPU().data);
		toMagnitudeMatrix(fourierGPUptr, fourierMagGPU().data);

		fourierMagCPU.write("cpu_mag");
		fourierMagGPU.write("gpu_mag");
 	}


	// psd.write(fnOut);
}
