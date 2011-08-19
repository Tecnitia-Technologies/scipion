#!/usr/bin/env python
'''
/***************************************************************************
 * Authors:     J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
 *
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
 '''

#------------------------------------------------------------------------------------------------
# Generic protocol for all Xmipp programs


from protlib_base import XmippProtocol, protocolMain
from config_protocols import protDict
from protlib_utils import greenStr, redStr, blueStr

class ProtXmippProgram(XmippProtocol):
    def __init__(self, scriptname, project):
        XmippProtocol.__init__(self, protDict.xmipp_program.name, scriptname, project)
        self.Import = 'from protocol_program import *'

    def validate(self):
        return []
        #return ["Protocol not implemented yet..."]
    
    def summary(self):
        return ["This is a test summary",
                "Need a real summary here..."
                ]
        
    def buildCommandLine(self):
        paramsLine = ""
        lastListValue = None
        for k in dir(self):
            if k.startswith('_K_'):
                value = self.__dict__[k]
                if '_P_' in k:
                    if '_L_' in k: sep = '_P_L_'
                    else: sep = '_P_'
                    myLine = ""
                    key, suffix = k.split(sep)
                    if '_A_' in suffix:
                        args = suffix.split('_A_')
                        paramName = getParamName(args[0])
                        if  value != "":
                            if paramName not in paramsLine:
                                myLine = paramName
                            myLine += ' ' + value
                    else: #Param without args, True or False value
                        paramName = getParamName(suffix)
                        if value:
                            myLine = paramName
                    if sep != '_P_L_' or paramName == lastListValue:
                        paramsLine += " " + myLine
                elif '_L_' in k: #exclusive list of some params
                    lastListValue = value
                    
        print 'PARAMS LINE: ', greenStr(paramsLine)
        
        return paramsLine
    
    def defineSteps(self):
        program = self.ProgramName
        params = self.buildCommandLine()
        self.NumberOfMpi = 1
        self.NumberOfThreads = 1
        self.Db.insertStep('printCommandLine', programname=program, params=params)
#        self.Db.insertStep('runJob', 
#                             programname=program, 
#                             params=params,
#                             NumberOfMpi = self.NumberOfMpi,
#                             NumberOfThreads = self.NumberOfThreads)

def getParamName(paramId):
    if len(paramId) > 1: return '--' + paramId
    return '-' + paramId

def printCommandLine(log, programname, params):
    print "Running program: ", programname
    print "         params: ", params
