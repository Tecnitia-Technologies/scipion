# **************************************************************************
# *
# * Authors:         Javier Vargas (jvargas@cnb.csic.es) (2016)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'jmdelarosa@cnb.csic.es'
# *
# **************************************************************************


from pyworkflow.object import Float, String
from pyworkflow.protocol.params import (PointerParam, FloatParam, STEPS_PARALLEL,
                                        StringParam, BooleanParam, IntParam, LEVEL_ADVANCED)
from pyworkflow.em.data import Volume
from pyworkflow.em import Viewer
import pyworkflow.em.metadata as md
from pyworkflow.em.packages.xmipp3.protocol_directional_classes import XmippProtDirectionalClasses
from pyworkflow.em.packages.xmipp3.protocol_ctf_correct_wiener2d import XmippProtCTFCorrectWiener2D

from pyworkflow.em.protocol import ProtClassify3D
from pyworkflow.utils.path import moveFile, makePath
from pyworkflow.gui.plotter import Plotter
import xmipp
import math
import random



from pyworkflow.em.packages.xmipp3.convert import (writeSetOfParticles,
                                                   writeSetOfVolumes,
                                                   getImageLocation)

class XmippProtClass3DRansac(ProtClassify3D, XmippProtDirectionalClasses, XmippProtCTFCorrectWiener2D):
    """    
    Performs 3D classification of input particles with previous alignment
    """
    _label = 'classify3D_ransac'
    
    def __init__(self, *args, **kwargs):
        ProtClassify3D.__init__(self, *args, **kwargs)
        #XmippProtDirectionalClasses.__init__(self, *args, **kwargs)
        
    #--------------------------- DEFINE param functions --------------------------------------------   
    def _defineParams(self, form):

        form.addSection(label='Input')
        form.addParam('inputVolume', PointerParam, pointerClass='Volume',
                      label="Input volume",  
                      help='Select the input volume.')     
        form.addParam('inputParticles', PointerParam,
                      pointerClass='SetOfParticles', pointerCondition='hasAlignment',
                      label="Input particles", important=True,
                      help='Select the input projection images.')
        form.addParam('backRadius', IntParam, default=-1,
                      label='Mask radius',
                      help='Pixels outside this circle are assumed to be noise')
        form.addParam('targetResolution', FloatParam, default=10, label='Target resolution (A)', expertLevel=LEVEL_ADVANCED,
                      help='Expected Resolution of the initial 3D classes obtained by the 2D classes. You should have a good' 
                      'reason to modify the 10 A value')
        form.addParam('numClasses', IntParam, default=2, label='Number of 3D classes')
        form.addParam('symmetryGroup', StringParam, default='c1',
                      label="Symmetry group", 
                      help='See [[Xmipp Symmetry][http://www2.mrc-lmb.cam.ac.uk/Xmipp/index.php/Conventions_%26_File_formats#Symmetry]] page '
                           'for a description of the symmetry format accepted by Xmipp') 

        form.addSection(label='CTF')
        form.addParam('doWiener', BooleanParam, default='True',
                      label="CTF correction",
                      help='Perform CTF correction by Wiener filtering.') 
        form.addParam('isIsotropic', BooleanParam, default='True',
                      label="Isotropic Correction",condition='doWiener',
                      help='If true, Consider that there is not astigmatism and then it is performed an isotropic correction.') 
        form.addParam('padding_factor', IntParam, default=2,expertLevel=LEVEL_ADVANCED,
                      label="Padding factor",condition='doWiener',
                      help='Padding factor for Wiener correction ')
        form.addParam('wiener_constant', FloatParam, default=-1,expertLevel=LEVEL_ADVANCED,
                      label="Wiener constant",condition='doWiener',
                      help=' Wiener-filter constant (if < 0: use FREALIGN default)')
        form.addParam('correctEnvelope', BooleanParam, default='False',expertLevel=LEVEL_ADVANCED,
                      label="Correct for CTF envelope",condition='doWiener',
                      help=' Only in cases where the envelope is well estimated correct for it')
        
        form.addSection(label='Directional Classes')        
        form.addParam('directionalSamples', IntParam, default=5, label='Number of directional samples',
                      help="Number of random samples of the angular directions to obtain 3D reconstructions")
        form.addParam('directionalTrials', IntParam, default=100, label='Number of directional trials', expertLevel=LEVEL_ADVANCED, 
                      help="Number of random combinations of the angular directions to select good orientations in which perform 2D classification")
        form.addParam('angularSampling', FloatParam, default=5, label='Angular sampling', expertLevel=LEVEL_ADVANCED, help="In degrees")
        form.addParam('angularDistance', FloatParam, default=10, label='Angular distance', expertLevel=LEVEL_ADVANCED,
                      help="In degrees. An image belongs to a group if its distance is smaller than this value")
        

        form.addParallelSection(threads=1, mpi=1)


        
    def _insertAllSteps(self):
        
        convertId = self._insertFunctionStep('convertInputStep',
                                             self.inputParticles.get().getObjId(), self.inputVolume.get().getObjId(), 
                                             self.targetResolution.get())
        
        self._insertFunctionStep('constructGroupsStep', self.inputParticles.get().getObjId(),
                                 self.angularSampling.get(), self.angularDistance.get(), self.symmetryGroup.get())
        
        self._insertFunctionStep('selectDirections', self.symmetryGroup.get())
        

        #self._insertFunctionStep('wienerStep')

        #deps = [] # store volumes steps id to use as dependencies for last step
        
        #consGS = self._insertFunctionStep('constructGroupsStep', self.inputParticles.get().getObjId(),
                                 
        #commonParams    = self._getCommonParams()
        #deps.append(convertId)
        
    def convertInputStep(self, particlesId, volId, targetResolution):
        #XmippProtDirectionalClasses.convertInputStep(self, particlesId, volId, targetResolution)
        """ 
        Write the input images as a Xmipp metadata file. 
        particlesId: is only need to detect changes in
        input particles and cause restart from here.
        """
        writeSetOfParticles(self.inputParticles.get(), self._getPath('input_particles.xmd'))
        
        if (self.doWiener == True):
            params  =  '  -i %s' % self._getPath('input_particles.xmd')
            params +=  '  -o %s' % self._getExtraPath('corrected_ctf_particles.stk')
            params +=  '  --save_metadata_stack %s' % self._getExtraPath('corrected_ctf_particles.xmd')
            params +=  '  --pad %s' % self.padding_factor.get()
            params +=  '  --wc %s' % self.wiener_constant.get()
            params +=  '  --sampling_rate %s' % self.inputParticles.get().getSamplingRate()

            if (self.inputParticles.get().isPhaseFlipped()):
                params +=  '  --phase_flipped '
            
            if (self.correctEnvelope):
                params +=  '  --correct_envelope '
                
            nproc = self.numberOfMpi.get()
            nT=self.numberOfThreads.get() 
    
            self.runJob('xmipp_ctf_correct_wiener2d',
                        params)
        
        Xdim = self.inputParticles.get().getDimensions()[0]
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get()*0.4
        newTs = max(Ts,newTs)
        newXdim = Xdim*Ts/newTs
        
        if (self.doWiener == True):
            params =  '  -i %s' % self._getExtraPath('corrected_ctf_particles.xmd')
        else :
            params =  '  -i %s' % self._getPath('input_particles.xmd')
        params +=  '  -o %s' % self._getExtraPath('scaled_particles.stk')
        params +=  '  --save_metadata_stack %s' % self._getExtraPath('scaled_particles.xmd')
        params +=  '  --dim %d' % newXdim
        
        self.runJob('xmipp_image_resize',params)
        from pyworkflow.em.convert import ImageHandler
        img = ImageHandler()
        img.convert(self.inputVolume.get(), self._getExtraPath("volume.vol"))
        Xdim = self.inputVolume.get().getDim()[0]
        if Xdim!=newXdim:
            self.runJob("xmipp_image_resize","-i %s --dim %d"%\
                        (self._getExtraPath("volume.vol"),
                        newXdim), numberOfMpi=1)

    def constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup):
        XmippProtDirectionalClasses.constructGroupsStep(self, particlesId, angularSampling, angularDistance, symmetryGroup)
        
    def selectDirections(self,symmetryGroup):
        fnNeighbours = self._getExtraPath("neighbours.xmd")
        fnGallery=self._getExtraPath("gallery.doc")
        listOfBlocks = xmipp.getBlocksInMetaDataFile(fnNeighbours)
        
        Xdim = self.inputParticles.get().getDimensions()[0]
        Ts = self.inputParticles.get().getSamplingRate()
        newTs = self.targetResolution.get()*0.4
        newTs = max(Ts,newTs)
        self.newRadius=(self.backRadius.get())*(Ts/newTs)
        normFreq = 0.25*(self.targetResolution.get()/Ts)

        md = xmipp.MetaData()
        for i in range(self.directionalSamples):
            randBlock =random.randint(0, len(listOfBlocks))
            block = listOfBlocks[randBlock]
            fnBlock="%s@%s"%(block,fnNeighbours)
            fnDir = self._getExtraPath("direction_%s"%i)

            ''' the gallery give is a good reference'''
            galleryImgNo = int(block.split("_")[1])
            mdRef = xmipp.MetaData(fnGallery)
            rot  = mdRef.getValue(xmipp.MDL_ANGLE_ROT,galleryImgNo)
            tilt = mdRef.getValue(xmipp.MDL_ANGLE_TILT,galleryImgNo)
            psi = 0.0
            
            self.runJob("xmipp_image_align","-i %s  --oroot %s --iter 5 --ref %s"
                        %(fnBlock,fnDir,mdRef.getValue(xmipp.MDL_IMAGE,galleryImgNo)),numberOfMpi=1)
            
            self.runJob("xmipp_transform_mask","-i %s  -o %s --mask circular -%f"
                        %(self._getExtraPath("direction_%s_ref.xmp"%i),self._getExtraPath("direction_%s_ref.xmp"%i),self.newRadius)
                          ,numberOfMpi=1)
            
            objId = md.addObject()
            md.setValue(xmipp.MDL_IMAGE,self._getExtraPath("direction_%s_ref.xmp"%i),objId)
            md.setValue(xmipp.MDL_ANGLE_ROT,rot,objId)
            md.setValue(xmipp.MDL_ANGLE_TILT,tilt,objId) 
            md.setValue(xmipp.MDL_ANGLE_PSI,psi,objId)
            md.setValue(xmipp.MDL_SHIFT_X,0.0,objId)
            md.setValue(xmipp.MDL_SHIFT_Y,0.0,objId)

        fnRecons = self._getExtraPath("guess_%s"%0)
        md.write(fnRecons+'.xmd')
        
        self.runJob("xmipp_reconstruct_fourier","-i %s.xmd -o %s.vol --sym %s --max_resolution %f" %(fnRecons,fnRecons,self.symmetryGroup.get(),normFreq))
        self.runJob("xmipp_transform_filter",   "-i %s.vol -o %s.vol --fourier low_pass %f --bad_pixels outliers 0.5" %(fnRecons,fnRecons,normFreq))
        self.runJob("xmipp_transform_mask","-i %s.vol  -o %s.vol --mask circular -%f" %(fnRecons,fnRecons,self.newRadius))
        md.clear()
        
        #objId = mdCorr.addObject()
        #mdCorr.setValue(xmipp.MDL_WEIGHT,self.corrThresh.get(),objId)
        #mdCorr.write("corrThreshold@"+fnCorr,xmipp.MD_APPEND)
        #print "Correlation threshold: "+str(self.corrThresh.get())
            

        self.runJob("xmipp_image_align","-i %s  --oroot %s --iter 5"
                    %(fnBlock,fnDir,fnDir),numberOfMpi=1)
            


        
        #print xmipp.getBlocksInMetaDataFile(fnNeighbours)[0]
            #for block in xmipp.getBlocksInMetaDataFile(fnNeighbours):
                
                
        
    def createOutputStep(self):
        pass
    #--------------------------- INFO functions -------------------------------------------- 
    def _validate(self):
        pass
    
    def _summary(self):
        pass
    
    def _methods(self):
        messages = []
        return messages
    
    def _citations(self):
        return ['Vargas2014a']
    
    #--------------------------- UTILS functions -------------------------------------------- 
    def _updateLocation(self, item, row):
        index, filename = xmippToLocation(row.getValue(md.MDL_IMAGE))
        item.setLocation(index, filename)
