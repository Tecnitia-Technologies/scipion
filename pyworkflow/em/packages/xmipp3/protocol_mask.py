# **************************************************************************
# *
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *              Josue Gomez Blanco     (jgomez@cnb.csic.es)
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
This sub-package contains protocol for masks operations
"""

from pyworkflow.em import *
from pyworkflow.utils import *  
import xmipp
from geometrical_mask import XmippGeometricalMask3D, XmippGeometricalMask2D
from protocol_process import XmippProcessParticles, XmippProcessVolumes
from convert import createXmippInputImages, readSetOfParticles

from pyworkflow.em.constants import *
from constants import *


class XmippProtMask():
    """ This class implement the common features for applying a mask with Xmipp either SetOfParticles, Volume or SetOfVolumes objects.
    """
    
    def __init__(self, **args):
        self._program = "xmipp_transform_mask"
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        """ Add common mask parameters that can be used
        in several protocols definitions.
        Params:
            form: the Definition instance.
        """
        
        form.addParam('source', EnumParam,
                      label='Mask source',
                      default=SOURCE_GEOMETRY, choices=['Geometry','Created mask'], 
                      help='Select which type of mask do you want to apply. \n ')
        
        self._defineProtParams(form)
        
        form.addParam('fillType', EnumParam, 
                      choices=['value', 'min', 'max', 'avg'], 
                      default=MASK_FILL_VALUE,
                      label="Fill with ", display=EnumParam.DISPLAY_COMBO,
                      help='Select how are you going to fill the pixel values outside the mask. ')
        
        form.addParam('fillValue', IntParam, default=0, 
                      condition='fillType == %d' % MASK_FILL_VALUE,
                      label='Fill value',
                      help='Value to fill the pixel values outside the mask. ')
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertProcessStep(self, inputFn, outputFn, outputMd):
        if self.source.get() == SOURCE_MASK:
            self._insertFunctionStep('copyFileStep', self.inputMask.get().getLocation())
        self._insertProtStep(self, inputFn, outputFn, outputMd)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _getCommand(self, inputFn):
        if self.fillType == MASK_FILL_VALUE:
            fillValue = self.fillValue.get()
        
        self._args += " --substitute %(fillValue)f "
        
        if self.source == SOURCE_GEOMETRY:
            self._args += self._getGeometryCommand()
        elif self.source == SOURCE_MASK:
            self._args += "--mask binary_file %s" % self.maskFn
        else:
            raise Exception("Unrecognized mask type: %d" % self.source)

        return self._args % locals()


class XmippProtMaskParticles(ProtMaskParticles, XmippProcessParticles, XmippProtMask, XmippGeometricalMask2D):
    """ Apply mask to a set of particles """
    _label = 'mask particles'
    
    def __init__(self, **args):
        ProtMaskParticles.__init__(self, **args)
        XmippProcessParticles.__init__(self)
        XmippProtMask.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippProtMask._defineProcessParams(self, form)
    
    def _defineProtParams(self, form):
        form.addParam('inputMask', PointerParam, pointerClass="Mask", label="Input mask",condition='source==%d' % SOURCE_MASK)
        XmippGeometricalMask2D.defineParams(self, form, isGeometry='source==%d' % SOURCE_GEOMETRY, addSize=False)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertProtStep(prot, inputFn, outputFn, outputMd):
        XmippProcessParticles._insertProcessStep(prot, inputFn, outputFn, outputMd)
    
    #--------------------------- STEPS functions ---------------------------------------------------
    def copyFileStep(self, *args):
        """ Convert the input mask to file. """
        ImageHandler().convert(self.inputMask.get().getLocation(), (None, self.maskFn))
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _defineFilenames(self):
        XmippProcessParticles._defineFilenames(self)
        self.maskFn = self._getPath('mask.spi')
    
    def _getGeometryCommand(self):
        Xdim = self.inputParticles.get().getDimensions()[0]
        self.ndim = self.inputParticles.get().getSize()
        args = XmippGeometricalMask2D.argsForTransformMask(self, Xdim)
        return args


class XmippProtMaskVolumes(ProtMaskVolumes, XmippProcessVolumes, XmippProtMask, XmippGeometricalMask3D):
    """ Apply mask to a volume """
    _label = 'apply mask'
    
    def __init__(self, **args):
        ProtMaskVolumes.__init__(self, **args)
        XmippProcessVolumes.__init__(self)
        XmippProtMask.__init__(self, **args)
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineProcessParams(self, form):
        XmippProtMask._defineProcessParams(self, form)
    
    def _defineProtParams(self, form):
        form.addParam('inputMask', PointerParam, pointerClass="VolumeMask", label="Input mask",condition='source==%d' % SOURCE_MASK)
        XmippGeometricalMask3D.defineParams(self, form, isGeometry='source==%d' % SOURCE_GEOMETRY, addSize=False)
    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertProtStep(inputFn, outputFn, outputMd):
        XmippProcessVolumes._insertProcessStep(inputFn, outputFn, outputMd)
    
    #--------------------------- UTILS functions ---------------------------------------------------
    def _defineFilenames(self):
        XmippProcessVolumes._defineFilenames(self)
        self.maskFn = self._getPath('mask.spi')
    
    def _getGeometryCommand(self):
        if isinstance(self.inputVolumes.get(), Volume):
            Xdim = self.inputVolumes.get().getDim()[0]
        else:
            Xdim = self.inputVolumes.get().getDimensions()[0]
        args = XmippGeometricalMask3D.argsForTransformMask(self,Xdim)
        return args

