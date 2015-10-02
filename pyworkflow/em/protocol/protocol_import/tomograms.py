# **************************************************************************
# *
# * Author:     Josue Gomez Blanco (jgomez@cnb.csic.es)
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
In this module are protocol base classes related to EM imports of Tomograms
"""


from pyworkflow.protocol import params

from micrographs import ProtImportMicBase


class ProtImportTomograms(ProtImportMicBase):
    """Protocol to import a set of tomograms to the project"""
    _label = 'import tomograms'
    _outputClassName = 'SetOfTomograms'
#     _checkStacks = False    
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        ProtImportMicBase._defineParams(self, form)    
        
        form.addParam('bfactor', params.FloatParam, default=4.0,
                      label='Provide B-factor:',
                      help= '3D CTF model weighting B-factor per e-/A2')
        form.addParam('totalDose', params.FloatParam, default=40.0,
                      label='Provide acummulated dose:',
                      help= 'Total dose for the whole tomogram.')
    
    def setSamplingRate(self, tomoSet):
        ProtImportMicBase.setSamplingRate(self, tomoSet)
        tomoSet.setBfactor(self.bfactor.get())
        tomoSet.setDose(self.totalDose.get())
        