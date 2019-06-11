#--------------------------------------------------------
#--------------------------------------------------------
#
#   Build a triplet network
#
#--------------------------------------------------------
#--------------------------------------------------------

# Import python
import os
import h5py
import numpy as np
from datetime import datetime
import pdb

#--------------------
# Class definition
#--------------------

class buildNetwork(object):
    '''
    A class to build a triplet network 
    and save informations into h5 file.

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
# Initialization
#--------------------

    def __init__(self):
        '''
        Initialize the class.
        '''

        # Initialize lists
        self.all_datesList = []
        self.all_tripletsList = []
        self.all_closuresList = []
        self.all_iterationList = []
        self.all_correctedpix = []
        self.all_notcorrectedpix = []

        # All done
        return

#--------------------
# Build from ISCE
#--------------------

    def buildIsce(self, igramsDir, igramName, proc):
        '''
        Build the network from ISCE architecture.

        Args:
            * igramsDir     : isce interferograms directory (str)
            * igramName     : interferogram name (str)
            * proc          : processing software (str)
        '''

        # Initialize lists of triplets
        triplets_new_tmp = []
        triplets_new = []

        # Get all dates from isce igram directory
        if proc == 'isce':
            all_dates_tmp = [d for d in os.listdir(igramsDir) if os.path.isdir(os.path.join(igramsDir,d)) \
                                                          and os.path.exists(os.path.join(igramsDir,d,igramName))]
            self.length_max = None

        if proc == 'roipac':
            all_dates_tmp = []
            for d in os.listdir(igramsDir):
                if os.path.isdir(os.path.join(igramsDir,d)):
                    d1 = d.split('_')[0]
                    d2 = d.split('_')[1]
                    b = igramName.split('_')[:]
                    b.insert(2, "{}-{}".format(d1,d2))
                    igramName_tmp = "_".join(b)
                    if os.path.exists(os.path.join(igramsDir,d,igramName_tmp)):
                        all_dates_tmp.append(d)
        all_dates = [d.split('_') for d in all_dates_tmp]
        unique_dates = list(set(x for l in all_dates for x in l))
        unique_dates = sorted(unique_dates, key=lambda x: datetime.strptime(x,'%Y%m%d'))

        # Get the maximum length to reshape igrams at the same length when readFiles
        if proc == 'roipac':
            from osgeo import gdal
            length_list = []
            for igrams in all_dates:
                b = igramName.split('_')[:]
                b.insert(2, "{}-{}".format(igrams[0],igrams[1]))
                igramName_tmp = "_".join(b)
                ifgFile = '{}/{}_{}/{}'.format(igramsDir,igrams[0],igrams[1],igramName_tmp)
                ds = gdal.Open(r'{}'.format(ifgFile))
                length_list.append(int(ds.RasterYSize))
            self.length_max = max(length_list)

        # Search triplets
        for du in unique_dates:
            for d in all_dates:
                triplet = []
                # Sort the three dates
                d_arr = [du,d[0],d[1]]
                d_arr_sort=sorted(d_arr, key=lambda x: datetime.strptime(x,'%Y%m%d'))
                # Search igrams that form triplet with the unique date
                if ([d_arr_sort[0],d_arr_sort[1]]) in all_dates \
                and ([d_arr_sort[1],d_arr_sort[2]]) in all_dates \
                and ([d_arr_sort[0],d_arr_sort[2]]) in all_dates:
                    triplet.append(d_arr_sort[0])
                    triplet.append(d_arr_sort[1])
                    triplet.append(d_arr_sort[2])
                    if triplet not in triplets_new_tmp:
                        triplets_new_tmp.append(triplet)

        # Get triplets not already processed
        self.triplets_new = [t for t in triplets_new_tmp if t not in self.all_tripletsList]

        # Save new data to lists of all (processed and not processed)
        self.all_tripletsList.extend(self.triplets_new)
        self.all_iterationList.extend([0] * len(self.triplets_new))

        # All done
        return

#--------------------
# Create h5 file
#--------------------

    def createh5(self, h5file):
        '''
        Create h5 file to save informations.

        Args:
            * h5file    : h5 file (str)
        '''

        # Create the file
        h5in = h5py.File(h5file, 'w')
        h5in.attrs['help'] = 'All informations about unwCorrector.'

        # Initialize datasets
        triplets = h5in.create_dataset('triplets', (1,3), dtype='i8', chunks=True, maxshape=(None, None))
        iteration = h5in.create_dataset('iteration', (1,1), dtype='i8', chunks=True, maxshape=(None, None))
        corrected_pix = h5in.create_dataset('corrected_pix', (1,1), dtype='i8', chunks=True, maxshape=(None, None))
        notcorrected_pix = h5in.create_dataset('notcorrected_pix', (1,1), dtype='i8', chunks=True, maxshape=(None, None))

        # Add attributes
        triplets.attrs['help'] = 'Triplets processed (int).'
        iteration.attrs['help'] = 'Number of iteration per triplet (int).'
        corrected_pix.attrs['help'] = 'Number of pixel corrected automatically per triplet (int).'
        notcorrected_pix.attrs['help'] = 'Number of pixel not corrected automatically per triplet (int).'

        # Close file
        h5in.close()

        # All done
        return

#--------------------
# Read h5
#--------------------

    def readh5(self, h5file):
        '''
        Read triplets already processed and stored in h5 file.

        Args:
            * h5file    : h5 file where triplets are saved (str)
        '''

        # Open h5 file
        h5in = h5py.File(h5file, 'r+')

        # Read infos
        self.triplets = h5in['triplets']
        self.iteration = h5in['iteration']
        self.corrected_pix = h5in['corrected_pix']
        self.notcorrected_pix = h5in['notcorrected_pix']

        # Add triplets already processed to lists of all
        if self.triplets.value[0,1] != 0:

            # Convert to string and append triplets infos to lists
            for elts in range(0,self.triplets.value.shape[0]):
                self.all_tripletsList.append(list(map(str, self.triplets.value[elts])))
                # Get latest elements
                self.all_iterationList.append(self.iteration[elts, -1])
                self.all_correctedpix.append(self.corrected_pix[elts, -1])
                self.all_notcorrectedpix.append(self.notcorrected_pix[elts, -1])

        # Save
        self.h5in = h5in

        # All done
        return

#--------------------
# Save new infos in h5 file
#--------------------

    def saveh5(self, triplet, iteration, corr_pix_nb, notcorr_pix_nb):
        '''
        Save triplet informations in h5 file.

        Args:
            * triplet           : triplet dates (list of str)
            * iteration         : triplet iteration step (int)
            * corr_pix_nb       : number of corrected pixels (int)
            * notcorr_pix_nb    : number of not corrected pixels (int)
        '''

        # If no previous data, overwrite empty datasets
        if self.triplets.value[0,1] == 0:
            print('No previous data, overwriting the empty data in h5 file.')
        
        # Previous data, add new data in datasets
        else:
            # Resize datasets
            self.triplets.resize((self.triplets.shape[0] + 1, 3))
            self.iteration.resize((self.iteration.shape[0] + 1, 1))
            self.corrected_pix.resize((self.corrected_pix.shape[0] + 1, 1))
            self.notcorrected_pix.resize((self.notcorrected_pix.shape[0] + 1, 1))

        # Add triplet dates to h5
        self.triplets[self.triplets.shape[0]-1,] = list(map(int,triplet))

        # Add triplet iteration number (add an iteration)
        self.iteration[self.iteration.shape[0]-1,] = iteration +1

        # Add triplet corrected pixels
        self.corrected_pix[self.corrected_pix.shape[0]-1,] = corr_pix_nb

        # Add triplet not corrected pixels
        self.notcorrected_pix[self.notcorrected_pix.shape[0]-1,] = notcorr_pix_nb

        # All done
        return

#--------------------
# Update infos in h5 file
#--------------------

    def updateh5(self, triplet, iteration, corr_pix_nb, notcorr_pix_nb):
        '''
        Update triplet informations in h5 file if it is already processed.

        Args:
            * triplet           : triplet dates (list of str)
            * iteration         : triplet iteration step (int)
            * corr_pix_nb       : number of corrected pixels (int)
            * notcorr_pix_nb    : number of not corrected pixels (int)
        '''

        # Find triplet position in h5 datasets
        for elts in range(0,self.triplets.value.shape[0]):
            if list(map(str, self.triplets.value[elts])) == triplet:
                pos_t = elts

        # Check if resize necessary
        if self.iteration[pos_t,-1] != self.iteration[pos_t, self.iteration.shape[1]-2] or \
        self.iteration.shape[1] == 1:

            # Resize datasets
            self.iteration.resize((self.iteration.shape[0], self.iteration.shape[1] + 1))
            self.corrected_pix.resize((self.corrected_pix.shape[0], self.corrected_pix.shape[1] + 1))
            self.notcorrected_pix.resize((self.notcorrected_pix.shape[0], self.notcorrected_pix.shape[1] + 1))

            # Copy previous iteration for triplets not corrected here
            for elts in range(0,self.triplets.value.shape[0]):
                self.iteration[elts,self.iteration.shape[1]-1] = int(self.iteration.value[elts,self.iteration.shape[1]-2])

        # Update iteration for the triplet
        self.iteration[pos_t,-1] = iteration + 1
        self.corrected_pix[pos_t,-1] = corr_pix_nb
        self.notcorrected_pix[pos_t,-1] = notcorr_pix_nb

        # All done
        return

#--------------------
# Close h5 file
#--------------------

    def closeh5(self):
        '''
        Close h5 file to save informations.

        Args:
            None
        '''
    
        # Close the file
        self.h5in.flush()
        self.h5in.close()
    
        # All done
        return
