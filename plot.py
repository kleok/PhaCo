#-------------------------------------------------
#-------------------------------------------------
#
#   Class plot
#
#-------------------------------------------------
#-------------------------------------------------

# Import all
import matplotlib.pyplot as plt
import numpy as np

#--------------------
# Class definition
#--------------------

class plot(object):
    '''
    A class to plot.

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

    def __init__(self, trip):
        '''
        Initialize the class.

        Args:
            * trip      : triplet
        '''

        # Get triplet class arguments
        self.trip = trip
        
        # All done
        return

#--------------------
# Plot triplet misclosures
#--------------------

    def plotMisclosures(self):
        '''
        Plot misclosures of the triplet.
        '''

        # Plot 0 with NaNs
        self.trip.closure_unw[self.trip.closure_unw ==0.] = 'nan'
        self.trip.closure_int[self.trip.closure_int ==0.] = 'nan'
        self.trip.closure[self.trip.closure ==0.] = 'nan'
    
        # Misclosures of the triplet
        fig = plt.figure()
        ax1=fig.add_subplot(1,4,1)
        plt.imshow(self.trip.closure_int)
        ax1.set_title('Misclosure from wrap igram')
        ax2=fig.add_subplot(1,4,2, sharex=ax1, sharey=ax1)
        plt.imshow(self.trip.closure_unw)
        ax2.set_title('Closure of unwrap igram')
        ax3=fig.add_subplot(1,4,3, sharex=ax1, sharey=ax1)
        plt.imshow(self.trip.closure)
        ax3.set_title('Total closure without misclosure')
        #ax4=fig.add_subplot(1,4,4)
        #plt.hist(self.trip.closure_int.flatten(), bins=200, log=True)
        #ax4.set_title('Distrib of misclosure')
        plt.show()

        ## Misclosure before and after regions size selection
        #self.trip.closure_mod2pi_errors[abs(self.trip.closure_mod2pi_errors)==0] = np.nan
        #fig = plt.figure()
        #ax1=fig.add_subplot(1,2,1)
        #plt.imshow(self.trip.closure_mod2pi_errors)
        #ax1.set_title('Closure before size selection')
        #ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        #plt.imshow(self.trip.closure_mod2pi_errors_ok)
        #ax2.set_title('Closure after size selection')
        #plt.show()

        # Redo NaN to 0
        #self.trip.closure_mod2pi_errors_ok = np.nan_to_num(self.trip.closure_mod2pi_errors_ok)
        self.trip.closure_unw = np.nan_to_num(self.trip.closure_unw)
        self.trip.closure_int = np.nan_to_num(self.trip.closure_int)
        self.trip.closure = np.nan_to_num(self.trip.closure)

        # All done
        return

#--------------------
# Plot error ref regions
#--------------------

    def plotRegions(self):
        '''
        Plot error and reference regions.
        '''

        # Make the plot
        fig = plt.figure()
        ax1=fig.add_subplot(1,2,1)
        plt.imshow(self.trip.labels_errors)
        ax1.set_title('All error regions labels')
        ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        plt.imshow(self.trip.labels_ref)
        ax2.set_title('All reference regions labels')
        plt.show()

        # All done
        return

#--------------------
# Plot triplet mean closure
#--------------------

   # def plotMeanClosure(self):

        #plt.figure()
        #plt.imshow(self.igram_meanClosure)
        #plt.title('igram mean closure')
        #plt.show()

