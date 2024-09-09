# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 20:12:58 2022

@author: Morishita and Yasuno
"""

import os

class octFilePath:
    def __init__(self, rawVolumeFilePath, outRootDir = ""):
        (self.inRootDir, self.inFileName) = os.path.split(rawVolumeFilePath)
        (self.inFileNameCore, self.inFileNameExt) = os.path.splitext(self.inFileName)

        # Set output root directory.
        if outRootDir == "":
            self.outRootDir = self.inRootDir
        else:
            self.setOutRootDir(outRootDir, append = False)
            # (theDir, theFile) = os.path.split(outRootDir)
            # # Remove "\" at the end of the path
            # if(theFile != ''):
            #     theDir = theDir+theFile
            # self.outRootDir = theDir
            
    def makePath(self, tag = "", ext = ""):
        outPath = self.outRootDir + '\\' + self.inFileNameCore + tag + ext
        return(outPath)
    
    def setOutRootDir(self, outRootDir, append = True):
        """
        outRootDir : string
            The full path of the output folder or subfolder name appended to the input folder to creat the output folder.
        append : Bool
            If true, a subfolder of "outRootDir" is appended to the current output root directory.
            If False, the outRootDir (full path) is set as the output root directory.

        Returns
        -------
        None.

        """
        if append == True:
            self.outRootDir = self.outRootDir + '\\' + outRootDir
            # Make the output sub-directory if it does not exist
            if os.path.exists(self.outRootDir) == False:
                os.makedirs(self.outRootDir)
        else:
            (theDir, theFile) = os.path.split(outRootDir)
            # Remove "\" at the end of the path
            if(theFile != ''):
                theDir = theDir+theFile
            self.outRootDir = theDir
