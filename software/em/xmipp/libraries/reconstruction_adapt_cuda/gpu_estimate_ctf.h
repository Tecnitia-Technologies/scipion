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
#ifndef _PROG_GPU_ESTIMATE_CTF
#define _PROG_GPU_ESTIMATE_CTF


#include <data/xmipp_program.h>


/**@defgroup Estimate CTF through the GPU
   @ingroup ReconsLibrary */
//@{

//#define CU_ESTIMATE_CTF_USE_FLOAT

#ifdef CU_ESTIMATE_CTF_USE_FLOAT
typedef float real_t;
typedef std::complex_t<float> complex_t;
#else
typedef double real_t;
typedef std::complex<double> complex_t;
#endif

class ProgGpuEstimateCTF: public XmippProgram
{
public:
    /** Filename input micrograph */
	FileName fnMic;
    /** Filename output psd */
	FileName fnOut;
    /// Dimension of micrograph pieces
	size_t pieceDim;
    /** Overlap among pieces (0=No overlap, 1=Full overlap */
    double overlap;
public:
    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /** Run */
    void run();

private:
	template <typename T>
	void extractPiece(const MultidimArray<T>& mic, int N,
			int div_NumberX, size_t Ydim, size_t Xdim, MultidimArray<T>& piece);

	template <typename T>
	void computeDivisions(const Image<T>& mic,
			int& div_Number, int& div_NumberX, int& div_NumberY,
			size_t& Xdim, size_t& Ydim,	size_t& Zdim, size_t& Ndim);
};
//@}
#endif
