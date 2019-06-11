#-------------------------------------------------
#-------------------------------------------------
#
#   Class interferogram
#
#-------------------------------------------------
#-------------------------------------------------

# Import all
from .triplet import triplet
from .readFiles import readFiles
import os,sys
import numpy as np
import pdb

#--------------------
# Class definition
#--------------------

class interferogram(object):
    '''
    A class to:
        - get interferogram mean closure in the triplet network sense
        - correct the interferogram

    Written by A. Benoit, B. Pinel-Puyssegur and R. Jolivet 2019
    
    Licence:
        PhaCo: Phase unwrapping errors Correction
        Copyright (C) 2019 <Angelique Benoit, Beatrice Pinel-Puyssegur and Romain Jolivet>

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
        
        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.
        
        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <https://www.gnu.org/licenses/>.
    '''

#--------------------
# Initialize
#--------------------

    def __init__(self):
        '''
        Initialize the class.
        Nothing to do.
        '''

        # All done    
        return
        
#--------------------
# Get igram score
#--------------------

    def meanClosureIgram(self, tripletsList, dates, filesDir, igramName, proc, length_max, minSize):
        '''
        It computes interferogram mean closure on the triplets network 
        and can detect an unwrapping error.

        Args:
            * tripletsList      : triplets list (list)
            * dates             : dates of the interferogram (list)
            * filesDir          : Files directory (str)
            * igramName         : Interferogram filename (str)
        '''

        # Initialize
        tripletsIgram = []
        tripletsIgram_signs = []
        tripletsIgram_masks = []
        tripletsIgram_closures = []
        tripletsIgram_mean_tmp = []

        # Get triplets that contain the igram
        for elements in tripletsList:
            if dates[0] in elements \
            and dates[1] in elements:
                tripletsIgram.append(elements)
        
        # Iterate over igram triplets
        for t in tripletsIgram:

            # Read files
            filesTmp1 = readFiles()
            filesTmp2 = readFiles()
            filesTmp3 = readFiles()
            if proc == 'isce':
                filesTmp1.readIsce([t[0],t[1]], filesDir, igramName)
                filesTmp2.readIsce([t[1],t[2]], filesDir, igramName)
                filesTmp3.readIsce([t[0],t[2]], filesDir, igramName)
            if proc == "roipac":
                filesTmp1.readRoiPac([t[0],t[1]], filesDir, igramName, length_max)
                filesTmp2.readRoiPac([t[1],t[2]], filesDir, igramName, length_max)
                filesTmp3.readRoiPac([t[0],t[2]], filesDir, igramName, length_max)

            # Get interferogram args (useless, just to run triplet)
            igram1 = None 
            igram2 = None 
            igram3 = None 

            # Get triplet mask and closure
            tripTmp = triplet(filesTmp1, filesTmp2, filesTmp3, igram1, igram2, igram3, minSize)
            tripTmp.makeTotMask()
            tripTmp.getclosureTriplet()

            # Append to lists
            tripletsIgram_masks.append(tripTmp.mask)
            tripletsIgram_closures.append(tripTmp.closure)

            # Get igram sign in the triplet
            shape = tripletsIgram_closures[0].shape
            if t.index(dates[0]) == 0 and t.index(dates[1]) == 1:
                tripletsIgram_signs.append(np.ones((shape[0], shape[1])))
            if t.index(dates[0]) == 1 and t.index(dates[1]) == 2:
                tripletsIgram_signs.append(np.ones((shape[0], shape[1])))
            if t.index(dates[0]) == 0 and t.index(dates[1]) == 2:
                tripletsIgram_signs.append(np.ones((shape[0],shape[1])) *-1)

        # Get the number of triplets containing each pixel
        den_list = [sum(x) for x in zip(*tripletsIgram_masks)]

        # Convert lists to arrays
        closures_ar = np.array(tripletsIgram_closures)
        masks_ar = np.array(tripletsIgram_masks)
        den_ar = np.array(den_list)
        signs_ar = np.array(tripletsIgram_signs)

        # Mask pixels that are only in 1 triplet
        loc_pixtrip = np.where(den_ar == 1.)
        for x_pixtrip,y_pixtrip in zip(loc_pixtrip[0],loc_pixtrip[1]):
            for n_igram in range(0, closures_ar.shape[0]):
                masks_ar[n_igram, x_pixtrip, y_pixtrip] = 0

        # Get igram mean closure
        num_ar = closures_ar * masks_ar * signs_ar
        sum_num = sum(num_ar)
        den_ar = np.array(den_list)
        np.seterr(divide='ignore', invalid='ignore')
        igram_meanClosure = sum_num/den_ar
        igram_meanClosure = np.nan_to_num(igram_meanClosure)

        # Save things
        self.tripletsIgram = tripletsIgram
        self.igram_meanClosure = igram_meanClosure

        # All done
        return

#--------------------
# Correct igram
#--------------------

    def correctIgram(self, trip, error_nb, ph, ifgFile):
        '''
        Correct unwrapping error in an interferogram.

        Args:
            * trip          : triplet informations
            * error_nb      : error to process
            * ifgFile       : corrected interferogram filename
        '''

        # Pixels positions of the unw error
        err_pos = np.where(trip.labels_errors == trip.z_errors_labels_ok[error_nb])
        err_pos_x = err_pos[0]
        err_pos_y = err_pos[1]

        # Sign of the unw error
        err_sign = np.sign(trip.signs2correct[error_nb])

        # Correct the unw error in the phase signal (2 or 4pi)
        for x,y in zip(err_pos_x, err_pos_y):
            ph[x,y] = ph[x,y] + (err_sign * trip.pi2correct[error_nb]*np.pi)

        # Save the number of pixels that were corrected
        self.pix_corr_nb = len(err_pos_x)

        print("Interferogram corrected from unw error of {}pi: {}".format(trip.pi2correct[error_nb], ifgFile))

        # All done
        return
