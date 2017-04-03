/***************************************************************************
 *
 * Authors:     Javier Angel Velazquez Muriel (javi@cnb.csic.es)
 *              Carlos Oscar Sanchez Sorzano
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

#include "ctf_estimate_from_micrograph.h"
#include "ctf_enhance_psd.h"

#include <data/args.h>
#include <data/micrograph.h>
#include <data/metadata.h>
#include <data/xmipp_image.h>
#include <data/xmipp_fft.h>
#include <data/xmipp_threads.h>
#include <data/basic_pca.h>
#include <data/normalize.h>

/* Read parameters ========================================================= */
ProgCTFEstimateFromMicrograph::ProgCTFEstimateFromMicrograph()
{
    psd_mode = OnePerMicrograph; 
    PSDEstimator_mode = Periodogram;
}

void ProgCTFEstimateFromMicrograph::readParams()
{
    fn_micrograph = getParam("--micrograph");
    fn_root = getParam("--oroot");
    if (fn_root == "")
        fn_root = fn_micrograph.withoutExtension();
    pieceDim = getIntParam("--pieceDim");
    skipBorders = getIntParam("--skipBorders");
    overlap = getDoubleParam("--overlap");
    String aux = getParam("--psd_estimator");
    if (aux == "periodogram")
        PSDEstimator_mode = Periodogram;
    else
    {
        PSDEstimator_mode = ARMA;
        ARMA_prm.readParams(this);
    }
    Nsubpiece = getIntParam("--Nsubpiece");

    String mode = getParam("--mode");
    if (mode == "micrograph")
        psd_mode = OnePerMicrograph;
    else if (mode == "regions")
    {
        psd_mode = OnePerRegion;
        fn_pos = getParam("--mode", 1);
    }
    else if (mode == "particles")
    {
        psd_mode = OnePerParticle;
        fn_pos = getParam("--mode", 1);
    }
    estimate_ctf = !checkParam("--dont_estimate_ctf");
    if (estimate_ctf)
        prmEstimateCTFFromPSD.readBasicParams(this);
    bootstrapN = getIntParam("--bootstrapFit");
}

void ProgCTFEstimateFromMicrograph::defineParams()
{
    addUsageLine("Estimate the CTF from a micrograph.");
    addUsageLine("The PSD of the micrograph is first estimated using periodogram averaging or ");
    addUsageLine("ARMA models ([[http://www.ncbi.nlm.nih.gov/pubmed/12623169][See article]]). ");
    addUsageLine("Then, the PSD is enhanced ([[http://www.ncbi.nlm.nih.gov/pubmed/16987671][See article]]). ");
    addUsageLine("And finally, the CTF is fitted to the PSD, being guided by the enhanced PSD ");
    addUsageLine("([[http://www.ncbi.nlm.nih.gov/pubmed/17911028][See article]]).");
    addParamsLine("   --micrograph <file>         : File with the micrograph");
    addParamsLine("  [--oroot <rootname=\"\">]    : Rootname for output");
    addParamsLine("                               : If not given, the micrograph without extensions is taken");
    addParamsLine("                               :++ rootname.psd or .psdstk contains the PSD or PSDs");
    addParamsLine("==+ PSD estimation");
    addParamsLine("  [--psd_estimator <method=periodogram>] : Method for estimating the PSD");
    addParamsLine("         where <method>");
    addParamsLine("                  periodogram");
    addParamsLine("                  ARMA");
    addParamsLine("  [--pieceDim <d=512>]       : Size of the piece");
    addParamsLine("  [--overlap <o=0.5>]        : Overlap (0=no overlap, 1=full overlap)");
    addParamsLine("  [--skipBorders <s=2>]      : Number of pieces around the border to skip");
    addParamsLine("  [--Nsubpiece <N=1>]        : Each piece is further subdivided into NxN subpieces.");
    addParamsLine("                              : This option is useful for small micrographs in which ");
    addParamsLine("                              : not many pieces of size pieceDim x pieceDim can be defined. ");
    addParamsLine("                              :++ Note that this is not the same as defining a smaller pieceDim. ");
    addParamsLine("                              :++ Defining a smaller pieceDim, would result in a small PSD, while ");
    addParamsLine("                              :++ subdividing the piece results in a large PSD, although smoother.");
    addParamsLine("  [--mode <mode=micrograph>]  : How many PSDs are to be estimated");
    addParamsLine("         where <mode>");
    addParamsLine("                  micrograph  : Single PSD for the whole micrograph");
    addParamsLine("                  regions <file=\"\"> : The micrograph is divided into a region grid ");
    addParamsLine("                              : and a PSD is computed for each one.");
    addParamsLine("                              : The file is metadata with the position of each particle within the micrograph");
    addParamsLine("                  particles <file> : One PSD per particle.");
    addParamsLine("                              : The file is metadata with the position of each particle within the micrograph");
    addParamsLine("==+ CTF fit");
    addParamsLine("  [--dont_estimate_ctf]       : Do not fit a CTF to PSDs");
    ARMA_parameters::defineParams(this);
    ProgCTFEstimateFromPSD::defineBasicParams(this);
    addExampleLine("Estimate PSD", false);
    addExampleLine("xmipp_ctf_estimate_from_micrograph --micrograph micrograph.mrc --dont_estimate_ctf");
    addExampleLine("Estimate a single CTF for the whole micrograph", false);
    addExampleLine("xmipp_ctf_estimate_from_micrograph --micrograph micrograph.mrc --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5");
    addExampleLine("Estimate a single CTF for the whole micrograph providing a starting point for the defocus",false);
    addExampleLine("xmipp_ctf_estimate_from_micrograph --micrograph micrograph.mrc --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --defocusU -15000");
    addExampleLine("Estimate a CTF per region", false);
    addExampleLine("xmipp_ctf_estimate_from_micrograph --micrograph micrograph.mrc --mode regions micrograph.pos --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --defocusU -15000");
    addExampleLine("Estimate a CTF per particle", false);
    addExampleLine("xmipp_ctf_estimate_from_micrograph --micrograph micrograph.mrc --mode particles micrograph.pos --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --defocusU -15000");
}

/* Construct piece smoother =============================================== */
void constructPieceSmoother(const MultidimArray<double> &piece,
                            MultidimArray<double> &pieceSmoother)
{
    // Attenuate borders to avoid discontinuities
    pieceSmoother.resizeNoCopy(piece);
    pieceSmoother.initConstant(1);
    pieceSmoother.setXmippOrigin();
    double iHalfsize = 2.0 / YSIZE(pieceSmoother);
    const double alpha = 0.025;
    const double alpha1 = 1 - alpha;
    const double ialpha = 1.0 / alpha;
    for (int i = STARTINGY(pieceSmoother); i <= FINISHINGY(pieceSmoother);
         i++)
    {
        double iFraction = fabs(i * iHalfsize);
        if (iFraction > alpha1)
        {
            double maskValue = 0.5
                               * (1 + cos(PI * ((iFraction - 1) * ialpha + 1)));
            for (int j = STARTINGX(pieceSmoother);
                 j <= FINISHINGX(pieceSmoother); j++)
                A2D_ELEM(pieceSmoother,i,j)*=maskValue;
        }
    }

    for (int j = STARTINGX(pieceSmoother); j <= FINISHINGX(pieceSmoother);
         j++)
    {
        double jFraction = fabs(j * iHalfsize);
        if (jFraction > alpha1)
        {
            double maskValue = 0.5
                               * (1 + cos(PI * ((jFraction - 1) * ialpha + 1)));
            for (int i = STARTINGY(pieceSmoother);
                 i <= FINISHINGY(pieceSmoother); i++)
                A2D_ELEM(pieceSmoother,i,j)*=maskValue;
        }
    }
}

/* Main ==================================================================== */
//#define DEBUG
void ProgCTFEstimateFromMicrograph::run()
{
    // Open input files -----------------------------------------------------
    // Open coordinates
    MetaData posFile;
    if (fn_pos != "")
        posFile.read(fn_pos);
    MDIterator iterPosFile(posFile);

    // Open the micrograph --------------------------------------------------
    Image<double> M_in;
    size_t Ndim, Zdim, Ydim , Xdim; // Micrograph dimensions

    M_in.read(fn_micrograph);
    M_in.getDimensions(Xdim, Ydim, Zdim, Ndim);

    // Compute the number of divisions --------------------------------------
    int div_Number = 0;
    int div_NumberX=1, div_NumberY=1;

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
				  << "div_Number : " << div_Number  << std::endl
				  << std::endl
				  << "pieceDim: " << pieceDim << std::endl
				  << "overlap:  " << overlap  << std::endl;

		std::cout << "Computing model of the micrograph" << std::endl;
		//init_progress_bar(div_Number);
	}

    // Process each piece ---------------------------------------------------
    Image<double> psd_avg, psd;
    MultidimArray<std::complex<double> > Periodogram;
    MultidimArray<double> piece(pieceDim, pieceDim);
    psd().resizeNoCopy(piece);
    MultidimArray<double> &mpsd = psd();
    double pieceDim2 = pieceDim * pieceDim;

    // Attenuate borders to avoid discontinuities
    MultidimArray<double> pieceSmoother;
    constructPieceSmoother(piece, pieceSmoother);

    if (verbose) {
        std::cerr << "Computing models of each piece ...\n";
        init_progress_bar(div_Number);
    }

    size_t piecei = 0, piecej = 0; // top-left corner of the current piece
    FourierTransformer transformer;
    int actualDiv_Number = 0;

	std::cout << div_Number << std::endl;
	for (int N = 1; N <= div_Number; N++)
	{
		bool skip = false;
		int blocki=0, blockj=0;
		// Compute the top-left corner of the piece ..........................

		int step = pieceDim;
		if (psd_mode == OnePerMicrograph)
			step = (int) ((1 - overlap) * step);
		blocki = (N - 1) / div_NumberX;
		blockj = (N - 1) % div_NumberX;
		if (blocki < skipBorders || blockj < skipBorders
				|| blocki > (div_NumberY - skipBorders - 1)
				|| blockj > (div_NumberX - skipBorders - 1))
			skip = true;
		piecei = blocki * step;
		piecej = blockj * step;

		// test if the full piece is inside the micrograph
		if (piecei + pieceDim > Ydim)
			piecei = Ydim - pieceDim;
		if (piecej + pieceDim > Xdim)
			piecej = Xdim - pieceDim;

		if (!skip)
		{
			std::cout << N << std::endl;
			// Extract micrograph piece ..........................................
			window2D( M_in(), piece, piecei, piecej, piecei + YSIZE(piece) - 1, piecej + XSIZE(piece) - 1);
			piece.statisticsAdjust(0, 1);
			normalize_ramp(piece);
			//piece *= pieceSmoother;

			// Estimate the power spectrum .......................................
			transformer.completeFourierTransform(piece, Periodogram);
			FFT_magnitude(Periodogram, mpsd);
			FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mpsd)
				DIRECT_MULTIDIM_ELEM(mpsd,n) *= DIRECT_MULTIDIM_ELEM(mpsd, n) * pieceDim2;

			actualDiv_Number += 1;
			// Compute average and standard deviation
			if (XSIZE(psd_avg()) != XSIZE(mpsd))
				psd_avg() = mpsd;
			else
				psd_avg() += mpsd;

			if (N == 42) {
				Image<double> p;

				p() = mpsd;
				p.write("cpu_1_psd");

				p() = piece;
				p.write("cpu_1_piece");
			}
		}
//		if (verbose)
//			progress_bar(N+1);
	}

//    if (verbose)
//        progress_bar(div_Number);

    // Average
    const MultidimArray<double> &mpsd_avg = psd_avg();
	double idiv_Number = 1.0 / actualDiv_Number;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mpsd_avg)
	{
		DIRECT_MULTIDIM_ELEM(mpsd_avg,n)*=idiv_Number;
	}

	psd_avg.write("psdAveraged.psd");
}

/* Fast estimate of PSD --------------------------------------------------- */
class ThreadFastEstimateEnhancedPSDParams
{
public:
    ImageGeneric *I;
    MultidimArray<double> *PSD, *pieceSmoother;
    MultidimArray<int> *pieceMask;
    Mutex *mutex;
    int Nprocessed;
};

void threadFastEstimateEnhancedPSD(ThreadArgument &thArg)
{
    ThreadFastEstimateEnhancedPSDParams *args =
        (ThreadFastEstimateEnhancedPSDParams*) thArg.workClass;
    int Nthreads = thArg.getNumberOfThreads();
    int id = thArg.thread_id;
    ImageGeneric &I = *(args->I);
    const MultidimArrayGeneric& mI = I();
    size_t IXdim, IYdim, IZdim;
    I.getDimensions(IXdim, IYdim, IZdim);
    MultidimArray<double> &pieceSmoother = *(args->pieceSmoother);
    MultidimArray<int> &pieceMask = *(args->pieceMask);
    MultidimArray<double> localPSD, piece;
    MultidimArray<std::complex<double> > Periodogram;
    piece.initZeros(pieceMask);
    localPSD.initZeros(*(args->PSD));

    FourierTransformer transformer;
    transformer.setReal(piece);

    int pieceNumber = 0;
    int Nprocessed = 0;
    double pieceDim2 = XSIZE(piece) * XSIZE(piece);
    for (size_t i = 0; i < (IYdim - YSIZE(piece)); i+=YSIZE(piece))
        for (size_t j = 0; j < (IXdim - XSIZE(piece)); j+=XSIZE(piece), pieceNumber++)
        {
            if ((pieceNumber + 1) % Nthreads != id)
                continue;
            Nprocessed++;

            // Extract micrograph piece ..........................................
            for (size_t k = 0; k < YSIZE(piece); k++)
                for (size_t l = 0; l < XSIZE(piece); l++)
                    DIRECT_A2D_ELEM(piece, k, l)= mI(i+k, j+l);
            piece.statisticsAdjust(0, 1);
            normalize_ramp(piece, &pieceMask);
            piece *= pieceSmoother;

            // Estimate the power spectrum .......................................
            transformer.FourierTransform();
            transformer.getCompleteFourier(Periodogram);
            FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localPSD)
            {
                double *ptr = (double*) &DIRECT_MULTIDIM_ELEM(Periodogram, n);
                double re=*ptr;
                double im=*(ptr+1);
                double magnitude2=re*re+im*im;
                DIRECT_MULTIDIM_ELEM(localPSD,n)+=magnitude2*pieceDim2;
            }
        }

    // Gather results
    args->mutex->lock();
    args->Nprocessed += Nprocessed;
    *(args->PSD) += localPSD;
    args->mutex->unlock();
}

void fastEstimateEnhancedPSD(const FileName &fnMicrograph, double downsampling,
                             MultidimArray<double> &enhancedPSD, int numberOfThreads)
{
    size_t Xdim, Ydim, Zdim, Ndim;
    getImageSizeFromFilename(fnMicrograph, Xdim, Ydim, Zdim, Ndim);
    int minSize = 2 * (std::max(Xdim, Ydim) / 10);
    minSize = (int)std::min((double) std::min(Xdim, Ydim), NEXT_POWER_OF_2(minSize));
    minSize = std::min(1024, minSize);

    /*
     ProgCTFEstimateFromMicrograph prog1;
     prog1.fn_micrograph=fnMicrograph;
     prog1.fn_root=fnMicrograph.withoutExtension()+"_tmp";
     prog1.pieceDim=(int)(minSize*downsampling);
     prog1.PSDEstimator_mode=ProgCTFEstimateFromMicrograph::Periodogram;
     prog1.Nsubpiece=1;
     prog1.psd_mode=ProgCTFEstimateFromMicrograph::OnePerMicrograph;
     prog1.estimate_ctf=false;
     prog1.bootstrapN=-1;
     prog1.verbose=1;
     prog1.overlap=0;
     prog1.run();
     */
    // Prepare auxiliary variables
    ImageGeneric I;
    I.read(fnMicrograph);

    MultidimArray<double> PSD;
    PSD.initZeros(minSize, minSize);

    MultidimArray<int> pieceMask;
    pieceMask.resizeNoCopy(PSD);
    pieceMask.initConstant(1);

    MultidimArray<double> pieceSmoother;
    constructPieceSmoother(PSD, pieceSmoother);

    // Prepare thread arguments
    Mutex mutex;
    ThreadFastEstimateEnhancedPSDParams args;
    args.I = &I;
    args.PSD = &PSD;
    args.pieceMask = &pieceMask;
    args.pieceSmoother = &pieceSmoother;
    args.Nprocessed = 0;
    args.mutex = &mutex;
    ThreadManager *thMgr = new ThreadManager(numberOfThreads, &args);
    thMgr->run(threadFastEstimateEnhancedPSD);
    if (args.Nprocessed != 0)
        *(args.PSD) /= args.Nprocessed;

    ProgCTFEnhancePSD prog2;
    prog2.filter_w1 = 0.02;
    prog2.filter_w2 = 0.2;
    prog2.decay_width = 0.02;
    prog2.mask_w1 = 0.005;
    prog2.mask_w2 = 0.5;

    prog2.applyFilter(*(args.PSD));
    enhancedPSD = *(args.PSD);

    int downXdim = (int) (XSIZE(enhancedPSD) / downsampling);
    int firstIndex = FIRST_XMIPP_INDEX(downXdim);
    int lastIndex = LAST_XMIPP_INDEX(downXdim);
    enhancedPSD.setXmippOrigin();
    enhancedPSD.selfWindow(firstIndex, firstIndex, lastIndex, lastIndex);
}
