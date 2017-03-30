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

//#include <reconstruction_cuda/cuda_gpu_rotate_image.h>
#include <data/xmipp_image.h>
#include "data/xmipp_program.h"

// Read arguments ==========================================================
void ProgGpuEstimateCTF::readParams()
{
	fnMic = getParam("-i");
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
    addParamsLine("  [--pieceDim <d=512>]    : Size of the piece");
    addParamsLine("  [--overlap <o=0.5>]     : Overlap (0=no overlap, 1=full overlap)");
}

//#define DEBUG
// Compute distance --------------------------------------------------------
void ProgGpuEstimateCTF::run()
{
	// Input
    Image<float> Imic;
    // Result
    Image<float> psd;

    Imic.read(fnMic);
    size_t Xdim, Ydim, Zdim, Ndim;
    Imic.getDimensions(Xdim, Ydim, Zdim, Ndim);
    float *Imic_ptr = MULTIDIM_ARRAY(Imic());

    // Compute the number of divisions --------------------------------------
    int div_Number = 0;
    int div_NumberX=1, div_NumberY=1;
    div_NumberX = CEIL((double)Xdim / (pieceDim *(1-overlap))) - 1;
	div_NumberY = CEIL((double)Ydim / (pieceDim *(1-overlap))) - 1;
	div_Number = div_NumberX * div_NumberY;

	if (verbose) {
		std::cout << "Computing model of the micrograph" << std::endl;
		init_progress_bar(div_Number);
	}


	// TODO
	//	piece.statisticsAdjust(0, 1);
	//	normalize_ramp(piece);

	// Reservar espacio para una linea div_NumberX * sizeof(float)
	// Reservar resultado 2 * div_NumberX * sizeof(float)
	for (int line = 0; line < div_NumberY; ++line) {
		// Subir a gpu la linea Imic_ptr[line * div_NumberX]
		// Procesar linea
		// traer resultado
		// reducir resultado
	}

	// media
	// psd *= (double) 1.0 / div_Number;



}

