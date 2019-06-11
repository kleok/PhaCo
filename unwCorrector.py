#--------------------------------------------------------
#--------------------------------------------------------
#
#   Main script
#
#--------------------------------------------------------
#--------------------------------------------------------

# Import python
import os,sys
import argparse
import numpy as np
import pdb

# Import my classes
from UnwErrCor.buildNetwork import buildNetwork
from UnwErrCor.readFiles import readFiles
from UnwErrCor.interferogram import interferogram
from UnwErrCor.triplet import triplet
from UnwErrCor.plot import plot

'''
Main script to run the PhaCo algorithm.

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
# Parser
#--------------------

def createParser():
    '''
    Create command line parser.
    '''

    parser = argparse.ArgumentParser(description='Correct interferograms from unwrapping errors')

    parser.add_argument('--h5file', dest='h5file', type=str, default='./unwCorrector.h5',
                        help='Path for h5 storage file')
    parser.add_argument('--filesDir', dest='filesDir', type=str,
                        help='Files directory')
    parser.add_argument('--igramName', dest='igramName', type=str,
                        help='Interferogram filename')
    parser.add_argument('--verbose', dest='verbose', type=bool, default=False,
                        help='Verbose mode')
    parser.add_argument('--proc', dest='proc', type=str, required=True, default='isce',
                        help='Software used (isce, roipac)')
    parser.add_argument('--counterOn', dest='counterOn', type=bool, default=False,
                        help='Show bar for processing (set Off if you redirecting to a file')
    parser.add_argument('--proc_iterations', dest='proc_iterations', type=int, default=0,
                        help='Process triplet if its iteration is equal or lesser than proc_iterations')
    # Thresholds
    parser.add_argument('--minIgramPix', dest='minIgramPix', type=int, default=20,
                        help='Minimum percentage of pixels in interferogram to start processing')
    parser.add_argument('--minSize', dest='minSize', type=int, default=200,
                        help='Minimum pixel size for an unwrapping error')
    parser.add_argument('--t1', dest='t1', type=float, default=0.5,
                        help='Pi threshold to detect an unwrapping error (2pi +/- t1)')
    parser.add_argument('--p_flux', dest='p_flux', type=float, default=0.30,
                        help='Minimum proportion of flux vectors to decide if it is an unwrapping error, flux method')
    parser.add_argument('--t1_prop', dest='t1_prop', type=float, default=1.5,
                        help='Flux ratio between 2 igrams to decide to correct')
    parser.add_argument('--p_mc', dest='p_mc', type=float, default=0.5,
                        help='Minimum proportion of values to decide if it is an unwrapping error, mean closure method')
    parser.add_argument('--r_mc', dest='r_mc', type=float, default=2.,
                        help='Ratio between 2 igrams to decide which igram to correct, mean closure method')
    parser.add_argument('--minFlux4MC', dest='minFlux4MC', type=float, default=0.5,
                        help='Minimum flux to agree with MC method')
    return parser


def cmdLineParse(iargs = None):
    '''
    Command line parser.
    '''

    parser = createParser()
    return parser.parse_args(args=iargs) 

#--------------------
# Main driver
#--------------------

def main(iargs=None):
    '''
    The main driver.
    '''

    #--------------------
    # INITIALIZE

    # Get input arguments
    inps = cmdLineParse(iargs)

    print('----------------------------------------')
    print('----------------------------------------')
    print('STARTING UNWRAPPING CORRECTOR:\n')

    #--------------------
    # BUILD TRIPLETS NETWORK

    # Create object
    buildtrip = buildNetwork()

    # Initialize h5 file for storage if not exists
    if not os.path.exists(inps.h5file):
        print("... Creating h5 file: {}".format(inps.h5file))
        buildtrip.createh5(inps.h5file)

    # Read h5 file triplets
    print("... Reading triplets already processed from: {}".format(inps.h5file))
    buildtrip.readh5(inps.h5file)

    # Search new triplets not yet corrected
    print("... Searching new triplets not yet corrected")
    if inps.proc == "isce" or inps.proc == "roipac":
        buildtrip.buildIsce(inps.filesDir, inps.igramName, inps.proc)
    else:
        print("Not yet implemented for this type of data, add your own.")
        sys.exit()

    #--------------------
    # START PROCESSING

    # Get index of chosen iterations
    pos_iterations = [index for index,value in enumerate(buildtrip.all_iterationList) if value <= inps.proc_iterations]

    # Print
    print('----------------------------------------')
    print('Total number of triplets: {}'.format(len(buildtrip.all_tripletsList)))
    print('Triplets already processed: {}'.format(len(buildtrip.all_tripletsList)-len(buildtrip.triplets_new)))
    print('Number of triplets to process: {}'.format(len(buildtrip.triplets_new)))
    print('Number of total triplets to process: {}'.format(len(pos_iterations)))
    print('----------------------------------------\n')

    # Iterate over triplets chosen by their iteration number
    n = 0
    for it_nb in pos_iterations:
        t = buildtrip.all_tripletsList[it_nb]
        n+=1
        print("\n... Processing triplet: {} --> {}/{}".format(t, n, len(pos_iterations)))

        # Create objects
        files1 = readFiles()
        files2 = readFiles()
        files3 = readFiles()
        igram1 = interferogram()
        igram2 = interferogram()
        igram3 = interferogram()
        trip = triplet(files1, files2, files3, igram1, igram2, igram3, inps.minSize)
        graph = plot(trip)

        # Read igrams and masks
        if inps.proc == "isce":
            files1.readIsce([t[0],t[1]], inps.filesDir, inps.igramName) 
            files2.readIsce([t[1],t[2]], inps.filesDir, inps.igramName)
            files3.readIsce([t[0],t[2]], inps.filesDir, inps.igramName)
        if inps.proc == "roipac":
            files1.readRoiPac([t[0],t[1]], inps.filesDir, inps.igramName, buildtrip.length_max)
            files2.readRoiPac([t[1],t[2]], inps.filesDir, inps.igramName, buildtrip.length_max)
            files3.readRoiPac([t[0],t[2]], inps.filesDir, inps.igramName, buildtrip.length_max)
        
        # Check if there is enough pixels in each igrams
        if np.count_nonzero(files1.phase) > ((files1.length * files1.width)*inps.minIgramPix)/100 \
        and np.count_nonzero(files2.phase) > ((files1.length * files1.width)*inps.minIgramPix)/100 \
        and np.count_nonzero(files3.phase) > ((files1.length * files1.width)*inps.minIgramPix)/100:

            # Build total mask
            trip.makeTotMask()

            # Compute triplet misclosure
            trip.getclosureTriplet()
            
            # Label regions
            trip.labelRegions(counterOn=inps.counterOn)

            # Plots
            #graph.plotMisclosures()
            #graph.plotRegions()

            # Continue if unw error are detected
            if len(trip.z_errors_labels_ok) > 0:

                # Fill holes
                trip.fillHoles()

                # METHOD: mean misclosure
                igram1.meanClosureIgram(buildtrip.all_tripletsList, [t[0],t[1]], inps.filesDir, inps.igramName, inps.proc, buildtrip.length_max, inps.minSize)
                igram2.meanClosureIgram(buildtrip.all_tripletsList, [t[1],t[2]], inps.filesDir, inps.igramName, inps.proc, buildtrip.length_max, inps.minSize)
                igram3.meanClosureIgram(buildtrip.all_tripletsList, [t[0],t[2]], inps.filesDir, inps.igramName, inps.proc, buildtrip.length_max, inps.minSize)

                # METHOD: flux
                trip.getFlux(pi_thr=inps.t1, min_flux=inps.p_flux, t1_prop=inps.t1_prop, t2_prop=inps.p_mc, t3_prop=inps.r_mc, minFlux4MC=inps.minFlux4MC)

                # Correct unw errors
                pixels_corrected_nb = 0
                for correction in range(len(trip.igrams2correct)):
                    if trip.igrams2correct[correction] == 0:
                        igram1.correctIgram(trip, correction, files1.phase, files1.ifgFile_corrected)
                        pixels_corrected_nb = pixels_corrected_nb + igram1.pix_corr_nb
                    if trip.igrams2correct[correction] == 1:
                        igram2.correctIgram(trip, correction, files2.phase, files2.ifgFile_corrected)
                        pixels_corrected_nb = pixels_corrected_nb + igram2.pix_corr_nb
                    if trip.igrams2correct[correction] == 2:
                        igram3.correctIgram(trip, correction, files3.phase, files3.ifgFile_corrected)
                        pixels_corrected_nb = pixels_corrected_nb + igram3.pix_corr_nb
                    else:
                        pass

                print("Pixels corrected: {}".format(pixels_corrected_nb))

                # Print how many pixels were not corrected
                pixels_notcorrected_nb = trip.pixels_tocorrect_nb - pixels_corrected_nb
                print("Pixels not corrected: {}".format(pixels_notcorrected_nb))

            # No unw errors detected
            else:
                pixels_corrected_nb = 0
                pixels_notcorrected_nb = 0

        # Not enough pixels to start processing
        else:
            print("Not enough points in interferograms, pass this triplet.")
            pixels_corrected_nb = 0
            pixels_notcorrected_nb = 0

        # Iteration step for the triplet
        it_triplet = buildtrip.all_iterationList[it_nb]

        # Save/update h5 file
        if t in buildtrip.triplets_new:
            buildtrip.saveh5(t, it_triplet, pixels_corrected_nb, pixels_notcorrected_nb)
        else:
            buildtrip.updateh5(t, it_triplet, pixels_corrected_nb, pixels_notcorrected_nb)

    # Close h5 file after processing
    buildtrip.closeh5()

    print('\n----------------------------------------')
    print('----------------------------------------')

    # All done
    return

# Run main
if __name__ == '__main__':
    main()
