# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
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
"""
This sub-package contains classes to use in common processing operations of SetOfParticles, Volume or SetOfVolumes
"""

from pyworkflow.em import *  
from pyworkflow.utils import *  
import xmipp
import xmipp3
from convert import createXmippInputImages, readSetOfParticles, getImageLocation

from pyworkflow.em.constants import *
from constants import *


class XmippProcess(EMProtocol):
    """ Class to create a base template for Xmipp protocol that share
    a common structure: build a commmand line and call a program. """
    def __init__(self):
        self._args = "-i %(inputFn)s"
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        self._defineFilenames()
        self._insertFunctionStep("convertStep")
        self._insertProcessStep(self.inputFn, self.outputStk, self.outputMd)
        self._insertFunctionStep('createOutputStep')
        
    def _insertProcessStep(self, inputFn, outputFn, outputMd):
        args = self._getCommand(inputFn)
        
        if outputFn != inputFn:
            args += " -o " + outputFn
        
        if outputMd is not None:
            args += (" --save_metadata_stack %(outputMd)s --keep_input_columns --track_origin") % locals()
        
        self._insertRunJobStep(self._program, args)


class XmippProcessParticles(XmippProcess):
    """ Class to create a base template for Xmipp protocols that process SetOfParticles """
    def __init__(self):
        XmippProcess.__init__(self)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        """ convert if necessary"""
        pass
    
    def createOutputStep(self):
        imgSet = self._createSetOfParticles()
        imgSet.copyInfo(self.inputParticles.get())
        readSetOfParticles(self.outputMd, imgSet)
        self._processOutput(imgSet)
        self._defineOutputs(outputParticles=imgSet)
        self._defineTransformRelation(self.inputParticles.get(), imgSet)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _defineFilenames(self):
        self.inputFn = createXmippInputImages(self, self.inputParticles.get())
        self.outputMd = self._getPath('output_images.xmd')
        self.outputStk = self._getPath('output_images.stk')
    
    def _processOutput(self, outputPcts):
        """ This function should be implemented
        if some additional modifications needs to be done
        on output particles.
        """
        pass


class XmippProcessVolumes(XmippProcess):
    """ Class to create a base template for Xmipp protocols that process both volume or a SetOfVolumes objects """
    def __init__(self):
        XmippProcess.__init__(self)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def convertStep(self):
        """ convert if necessary"""
        volInput = self.inputVolumes.get()
        
        # Check volInput is a volume or a stack
        if isinstance(volInput, Volume):
            ImageHandler().convert(volInput, (1, self.outputStk))
        else:
            volInput.writeStack(self.outputStk)
    
    def createOutputStep(self):
        volInput = self.inputVolumes.get()
        
        if self._isSingleInput():
            vol = Volume()
            vol.copyInfo(volInput)
            vol.setLocation(1, self.outputStk)
            self._defineOutputs(outputVol=vol)
        else:
            volumes = self._createSetOfVolumes()
            volumes.copyInfo(volInput)
            
            for i, vol in enumerate(volInput):
                j = i + 1 
                vol.setLocation(j, self.outputStk)
                volumes.append(vol)
            self._defineOutputs(outputVol=volumes)

        self._defineTransformRelation(volInput, self.outputVol)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _isSingleInput(self):
        return isinstance(self.inputVolumes.get(), Volume)
        
    def _defineFilenames(self):
        """ Prepare the files to process """
        if self._isSingleInput():
            self.outputStk = self._getPath("output_volume.vol")
        else:
            self.outputStk = self._getPath("output_volumes.stk")
        self.inputFn = self.outputStk
        self.outputMd = None
    
    def _processOutput(self, outputPcts):
        """ This function should be implemented
        if some additional modifications needs to be done
        on output particles.
        """
        pass
    
