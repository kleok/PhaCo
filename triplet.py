#-------------------------------------------------
#-------------------------------------------------
#
#   Class triplet
#
#-------------------------------------------------
#-------------------------------------------------

# Import all
import sys
import numpy as np
import scipy as sc
import math
from scipy.ndimage import measurements
from scipy import stats
import matplotlib.pyplot as plt
import pdb

#--------------------
# Class definition
#--------------------

class triplet(object):
    '''
    A class to:
        - build the triplet total mask
        - compute the triplet misclosure
        - label regions (errors and references)
        - launch flux method for correcting unw errors

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

    def __init__(self, files1, files2, files3, igram1, igram2, igram3, minSize):
        '''
        Initialize the class.

        Args:
            * files1        : phase and mask for the igram number 1 
            * files2        : phase and mask for the igram number 2
            * files3        : phase and mask for the igram number 3
            * igram1        : interferogram number 1
            * igram2        : interferogram number 2
            * igram3        : interferogram number 3
            * minSize       : minimum pixels in the unwrapping error (int)
        '''

        # Get readFiles class arguments
        self.files = [files1, files2, files3]

        # Get interferogram class arguments
        self.igrams = [igram1, igram2, igram3]

        # Minimum pixels in the unwrapping error (int)
        self.minSize = minSize

        # Initialize the referencing
        self.ref_nb = 0

        # All done
        return

#--------------------
# Build total mask
#--------------------

    def makeTotMask(self):
        '''
        Build the total mask, intersection of the 
        three interferograms masks.

        Args:
            None
        '''

        # Builf total mask with unwrapped igrams
        np.seterr(divide='ignore', invalid='ignore')
        mask_unw = (self.files[0].phase/self.files[0].phase) \
                   * (self.files[1].phase/self.files[1].phase) \
                   * (self.files[2].phase/self.files[2].phase)
        self.mask = np.nan_to_num(mask_unw)

        # All done
        return 

#--------------------
# Get closure
#--------------------

    def getclosureTriplet(self):
        '''
        Remove phase misclosure due to 
        phase loss in multilooking and compute 
        interferograms triplet phase closure.

        Args:
            None
        '''

        # Closure.unw
        self.closure_unw = (self.files[0].phase + self.files[1].phase - self.files[2].phase) * self.mask

        # Test on closure.int distribution
        closure_int_complex = (self.files[0].phase_int * self.files[1].phase_int * np.conj(self.files[2].phase_int))
        closure_int_test = (np.angle(closure_int_complex)) * self.mask
        closure_int_test[closure_int_test == 0.] = 'nan'
        alpha = 2
        p = np.nanmean(np.abs((closure_int_test).flatten()))
        # Case 1: centered on 0
        if p < alpha:
            flagBimodal = False
            self.closure_int = (np.angle(closure_int_complex)) * self.mask
        # Case 2: bimodal, centered on -pi and +pi TODO: optimize loop
        else:
            flagBimodal = False
            #for x in range(0,closure_int_test.shape[0]):
            #    for y in range(0,closure_int_test.shape[1]):
            #        if 1 < closure_int_test[x,y] < 9:
            #            closure_int_test[x,y] = closure_int_test[x,y] - 3.14
            #        if -9 < closure_int_test[x,y] < -1:
            #            closure_int_test[x,y] = closure_int_test[x,y] + 3.14
            #self.closure_int = closure_int_test * self.mask
            
            # NENLEVE PAS DE MISCLOSURE
            phaseint_1 = np.angle(self.files[0].phase_int)
            phaseint_2 = np.angle(self.files[1].phase_int)
            phaseint_3 = np.angle(self.files[2].phase_int)
            self.closure_int = ((phaseint_1 + phaseint_2 - phaseint_3) % (2*np.pi)) * self.mask

        ##----------- DIFFERENT BUT WE CAN TEST

        #    closure_int_complex = closure_int_complex * np.conj(np.pi)

        #-----------
        
        # Closures
        self.closure = self.closure_unw - self.closure_int
        if flagBimodal and self.ref_nb == 0:
            self.closure_mod2pi = np.round(self.closure/(np.pi)) * 2*np.pi
            #plt.figure()
            #plt.imshow(self.closure_mod2pi)
            #plt.show()
        else:
            self.closure_mod2pi = np.round(self.closure/(2*np.pi)) * 2*np.pi
    
        #fig = plt.figure()
        #ax1=fig.add_subplot(1,2,1)
        #plt.imshow(self.closure_mod2pi)
        #ax1.set_title('Closure tot')
        #ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        #plt.imshow(self.closure_int)
        #ax2.set_title('OLD WITH TEST')

        #ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        #plt.imshow(closure_mod2pi_OLD)
        #ax2.set_title('OLD WITH TEST')
        plt.show()

        # All done
        return 

#--------------------
# Label regions
#--------------------

    def labelRegions(self, counterOn):
        '''
        Label unwrapping errors regions and reference regions.

        Kwargs:
            * counterOn     : Counter on/off
        '''

        # Initialize lists
        z_errors_sizes = []
        z_errors_labels_ok = []
        z_errors_sizes_ok = []
        z_ref_labels = []
        z_ref_sizes = []
        z_ref_labels_assoc = []
        z_error_sign = []
        
        # Prepare closures for labeling
        closure_mod2pi_errors = self.closure_mod2pi * self.mask
        closure_mod2pi_errors[np.isnan(closure_mod2pi_errors)] = 0

        #-----------------------------------------------------
        # UNW ERRORS REGIONS

        # Group and label unwrapping errors pixels by regions
        labels_errors, num_errors = measurements.label(closure_mod2pi_errors)
        #fig = plt.figure()
        #ax1=fig.add_subplot(1,2,1)
        #plt.imshow(closure_mod2pi_errors)
        #ax1.set_title('Closure')
        #ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        #plt.imshow(labels_errors)
        #ax2.set_title('All errors regions labels')
        #plt.show()

        # Count unw errors sizes (in pixels) and remove the count of 0 values
        z_errors_sizes = np.bincount(labels_errors.flatten())
        z_errors_sizes = z_errors_sizes[1:]

        print('Size criterion: minimum {} pixels'.format(self.minSize))
        print('Number of unwrapping errors: {}'.format(len(z_errors_sizes)))

        # Size selection
        n = 0
        for i in range(len(z_errors_sizes)):

            # Counter
            if counterOn:
                n = n + 1
                sys.stdout.write("\rSelection of errors: {} / {} \r".format(n, len(z_errors_sizes)))
                sys.stdout.flush()

            # Check if error is greater than minSize
            if (z_errors_sizes[i] > self.minSize):
                # If a big unw error is detected due to different igrams prior referencing
                if z_errors_sizes[i] > (np.count_nonzero(self.files[0].phase.flatten())/2):
                    # Reference igram only one time
                    if self.ref_nb == 0:
                        self.ref_nb = self.ref_nb + 1
                        print("WARNING: Bad prior referencing")
                        # Get the reference step
                        a = np.where(labels_errors == i+1)
                        step = closure_mod2pi_errors[a[0][0]][a[1][0]]
                        # Search igram to reference
                        newclos_ref1 = np.round((self.files[0].phase[a[0][0]][a[1][0]] + step) + \
                                       self.files[1].phase[a[0][0]][a[1][0]] - self.files[2].phase[a[0][0]][a[1][0]])
                        newclos_ref2 = np.round(self.files[0].phase[a[0][0]][a[1][0]] + \
                                       (self.files[1].phase[a[0][0]][a[1][0]] + step) - self.files[2].phase[a[0][0]][a[1][0]])
                        newclos_ref3 = np.round(self.files[0].phase[a[0][0]][a[1][0]] + \
                                       self.files[1].phase[a[0][0]][a[1][0]] - (self.files[2].phase[a[0][0]][a[1][0]] + step))
                        newclos = [newclos_ref1, newclos_ref2, newclos_ref3]
                        # Reference the bad igram
                        for closure_idx in range(0, len(newclos)):
                            if 0. <= np.abs(newclos[closure_idx]) <= 6.:
                                # Print igram to reference
                                if closure_idx == 0:
                                    print("Referencing the igram: {}-{}".format(self.files[0].dates[0],self.files[0].dates[1]))
                                if closure_idx == 1:
                                    print("Referencing the igram: {}-{}".format(self.files[1].dates[0],self.files[1].dates[1]))
                                if closure_idx == 2:
                                    print("Referencing the igram: {}-{}".format(self.files[2].dates[0],self.files[2].dates[1]))
                                # Referencing the igram
                                self.files[closure_idx].phase += step
                                self.files[closure_idx].phase[self.files[closure_idx].phase == step] = 0.0
                                print("Restarting to compute closure and label regions")
                                self.getclosureTriplet()
                                self.labelRegions(counterOn=False)
                                return
                        else:
                            print("DO SOMETHING BEC NO IGRAM TO REFERENCE")
                            labels_errors[labels_errors == i+1] = False
                    else:
                        continue

                # Keep unwrapping errors greater than minSize
                else:
                    z_errors_labels_ok.append(i+1)
                    z_errors_sizes_ok.append(z_errors_sizes[i])

            # Remove small errors in labels array
            else:
                labels_errors[labels_errors == i+1] = False

        # Continue if errors are detected
        if len(z_errors_labels_ok) > 0:

            #-----------------------------------------------------
            # REFERENCE AREAS

            # Binary mask of the inverse of closure
            labels_errors_thr = labels_errors/labels_errors
            labels_errors_thr = np.nan_to_num(labels_errors_thr)
            mask_inv = (1 - labels_errors_thr)

            # Group and label reference pixels by regions
            labels_ref, num_ref = measurements.label(mask_inv)
            labels_ref = labels_ref * self.mask
            #fig = plt.figure()
            #ax1=fig.add_subplot(1,3,1)
            #plt.imshow(self.mask)
            #ax1.set_title('mask')
            #ax2=fig.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
            #plt.imshow(labels_ref)
            #ax2.set_title('All ref')
            #ax2=fig.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
            #plt.imshow(labels_errors)
            #ax2.set_title('Errors')
            #plt.show()

            # Count reference region pixels
            for regionNb in range(1,(num_ref+1)):
                z_ref_labels.append(regionNb)
                z_ref_sizes.append(len(np.where(labels_ref==regionNb)[0]))

            # Sort reference regions by size (bigger to smaller)
            if len(z_ref_sizes)>1:
                z_ref_sizes,z_ref_labels = (list(t) for t in zip(*sorted(zip(z_ref_sizes,z_ref_labels),reverse=True)))

            # Convert 0 to NaN in labels
            labels_errors = labels_errors.astype('float')
            labels_errors[labels_errors == 0.] = 'nan'
            labels_ref = labels_ref.astype('float')
            labels_ref[labels_ref == 0.] = 'nan'

            print('Number of unwrapping errors after size selection: {}'.format(len(z_errors_labels_ok)))

            #-----------------------------------------------------
            # ASSOCIATE REFERENCE AREA TO AN UNW ERROR

            # Iterate over unw errors
            n = 0
            for errors in z_errors_labels_ok:
                # Counter
                if counterOn:
                    n = n + 1
                    sys.stdout.write("\rAssociating references to errors: {} / {} \r".format(n, len(z_errors_labels_ok)))
                    sys.stdout.flush()
                flag_ref = False
                # Search pixels in error border
                err_eroded = sc.ndimage.binary_erosion(labels_errors==errors).astype(int)
                err_dilated = sc.ndimage.binary_dilation(labels_errors==errors).astype(int)
                err_border = (err_dilated - err_eroded) * self.mask
                #fig = plt.figure()
                #ax1=fig.add_subplot(1,4,1)
                #plt.imshow(err_eroded)
                #ax1.set_title('Eroded')
                #ax2=fig.add_subplot(1,4,2, sharex=ax1, sharey=ax1)
                #plt.imshow(err_dilated)
                #ax2.set_title('Dilated')
                #ax3=fig.add_subplot(1,4,3, sharex=ax1, sharey=ax1)
                #plt.imshow(err_border)
                #ax3.set_title('Error border')
                #ax4=fig.add_subplot(1,4,4, sharex=ax1, sharey=ax1)
                #plt.imshow(labels_ref==1)
                #ax4.set_title('Ref')
                #plt.show()

                # Iterate over reference regions
                try:
                    if len(z_ref_labels) > 5:
                        n = 5
                    else:
                        n = len(z_ref_labels)
                    # Test only with big references regions
                    for references in z_ref_labels[:n]:
                        # Binary mask for intersection of reference and error regions
                        intersection = (labels_ref==references) * err_border
                        #fig = plt.figure()
                        #ax1=fig.add_subplot(1,3,1)
                        #plt.imshow(labels_ref==references)
                        #ax1.set_title('Ref')
                        #ax2=fig.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
                        #plt.imshow(intersection)
                        #ax2.set_title('Intersection')
                        #ax3=fig.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
                        #plt.imshow(err_border)
                        #ax3.set_title('Error border')
                        #plt.show()
                        # Define as a reference if 1% common pixels with the error
                        if (np.count_nonzero(intersection)/np.count_nonzero(err_border)) > 0.01:
                        #if np.count_nonzero(intersection) > 10: # in pixel nb
                            z_ref_labels_assoc.append(references)
                            flag_ref = True
                            raise StopIteration
                except StopIteration: pass
                # Cannot associate a reference area
                if not flag_ref:
                    z_ref_labels_assoc.append(None)

            # Remove unw error if there is no reference area
            indices = [i for i,x in enumerate(z_ref_labels_assoc) if x == None]
            for index in sorted(indices, reverse=True):
                ind_to_remove = np.where(labels_errors==z_errors_labels_ok[index])
                for xtmp,ytmp in zip(ind_to_remove[0],ind_to_remove[1]):
                    labels_errors[xtmp,ytmp] = 0
                del z_ref_labels_assoc[index]
                del z_errors_labels_ok[index]

            print('Number of unwrapping errors after referencing regions: {}'.format(len(z_errors_labels_ok)))

            #-----------------------------------------------------

            # Get the new closure array (without small unw errors)
            closure_mod2pi_errors_ok = closure_mod2pi_errors * (labels_errors/labels_errors)

            # Get the sign of each error with closure
            for reg_lab in z_errors_labels_ok:
                ind_reg = np.where(labels_errors==reg_lab)
                reg_sign = np.sign(np.median(closure_mod2pi_errors_ok[ind_reg]))
                z_error_sign.append(reg_sign)

            # Save things
            self.z_errors_labels_ok = z_errors_labels_ok
            self.labels_errors = labels_errors
            self.labels_ref = labels_ref
            self.z_ref_labels_assoc = z_ref_labels_assoc
            self.closure_mod2pi_errors_ok = closure_mod2pi_errors_ok
            self.z_error_sign = z_error_sign
        
        else:
            self.z_errors_labels_ok = []
            print('Number of unwrapping errors after size selection: 0. End of process')

        #fig = plt.figure()
        #ax1=fig.add_subplot(1,2,1)
        #plt.imshow(self.labels_errors)
        #ax1.set_title('All error regions labels')
        #ax2=fig.add_subplot(1,2,2, sharex=ax1, sharey=ax1)
        #plt.imshow(self.labels_ref)
        #ax2.set_title('All reference regions labels')
        #plt.show()

        # All done
        return

#--------------------
# Fill holes
#--------------------

    def fillHoles(self):
        '''
        Fill the holes in unwrapping errors.
        Used to calculate more efficiently the flux.

        Args:
            * None
        '''

        # Make sure there are no NaN
        self.labels_errors = np.nan_to_num(self.labels_errors)

        # Fill holes
        self.labels_filled = sc.ndimage.binary_fill_holes(self.labels_errors).astype(int)

        # All done
        return
    
#--------------------
# Compute flux
#--------------------

    def getFlux(self, pi_thr, min_flux, t1_prop, t2_prop, t3_prop, minFlux4MC):
        '''
        Compute flux vectors between external and internal pixels
        of the unwrapping error.

        Args:
            * pi_thr        : Pi threshold to detect an unwrapping error (float)
            * min_flux      : Minimum proportion of flux vectors
                              to decide if it is an unwrapping error (float)
            * t1_prop       : Flux ratio between 2 igrams to decide to correct (float)
            * t2_prop       : Verify if there are enough non zero values for the mean closure method
            * t3_prop       : Ratio between 2 igrams mean closures to decide which igram to correct
            * minFlux4MC    : Minimum flux to agree with MC method (float)
        '''

        # Initialize
        self.igrams2correct = []
        self.signs2correct = []
        self.pi2correct = []

        # Label filled errors
        labels_filled_errors, num_filled_errors = measurements.label(self.labels_filled)

        # Count the total number of pixels to correct
        self.pixels_tocorrect_nb = np.count_nonzero((labels_filled_errors).flatten())
        print("Pixels to correct: {}".format(self.pixels_tocorrect_nb))

        # Iterate over unwrapping errors
        i = 0
        for error in self.z_errors_labels_ok:
            i = i+1

            # Initialize
            self.vectors_flux_igram1 = []
            self.vectors_flux_igram2 = []
            self.vectors_flux_igram3 = []
            score_val_igram1 = []
            score_val_igram2 = []
            score_val_igram3 = []
            counter_list1 = 0
            counter_list2 = 0
            counter_list3 = 0
            counter_list1_4pi = 0
            counter_list2_4pi = 0
            counter_list3_4pi = 0

            # Erode and dilate contours
            labels_eroded = sc.ndimage.binary_erosion(labels_filled_errors == i).astype(int)
            struct2 = sc.ndimage.generate_binary_structure(2, 2)
            labels_dilated = sc.ndimage.binary_dilation(labels_filled_errors == i, structure=struct2, iterations=2).astype(int)

            # Get internal and external regions between which to compute flux
            int_region = np.not_equal(labels_filled_errors == i, labels_eroded).astype(int)
            ext_region = np.not_equal(labels_filled_errors == i, labels_dilated).astype(int)

            # Get reference region associated to the error
            ref_assoc = (self.labels_ref == self.z_ref_labels_assoc[self.z_errors_labels_ok.index(error)]).astype(int)

            # Get only externals pixels that are in the reference region
            ext_region_ok = ref_assoc * ext_region

            # Get positions of internal and external pixels
            ext_region_ok_pos = np.where(ext_region_ok)
            int_region_pos = np.where(int_region)

            # Test avec meshgrid pour trouver distances entre pixels
            Xext,Xint = np.meshgrid(ext_region_ok_pos[1],int_region_pos[1])
            Yext,Yint = np.meshgrid(ext_region_ok_pos[0],int_region_pos[0])
            dist = np.sqrt((Xext-Xint)**2 + (Yext-Yint)**2)
            dist_1 = np.where(dist == 1)
            dist_1_x = dist_1[1]
            dist_1_y = dist_1[0]
            if len(dist_1_x) > 1:
                ex_x = dist_1_x[1]
                ex_y = dist_1_y[1]
                #print("Compute example: {} - {} inside / {} - {} outside".format(Xint[ex_y,ex_x],Yint[ex_y,ex_x],Xext[ex_y,ex_x],Yext[ex_y,ex_x]))
                for ind in range(0, len(dist_1_y)):
                    self.computeVectorsFlux(Xext[dist_1_y[ind],dist_1_x[ind]],Yext[ dist_1_y[ind],dist_1_x[ind]],\
                                            Xint[dist_1_y[ind],dist_1_x[ind]],Yint[dist_1_y[ind],dist_1_x[ind]])

            #fig = plt.figure()
            #ax1=fig.add_subplot(1,3,1)
            #plt.imshow(ref_assoc)
            #ax1.set_title('Ref')
            #ax2=fig.add_subplot(1,3,2, sharex=ax1, sharey=ax1)
            #plt.imshow(int_region)
            #ax2.set_title('int region')
            #ax3=fig.add_subplot(1,3,3, sharex=ax1, sharey=ax1)
            #plt.imshow(ext_region_ok)
            #ax3.set_title('Pix in the ref')
            #plt.show()

            ## Iterate over external pixels to compute flux vectors
            #for x,y in zip(ext_region_ok_pos[1],ext_region_ok_pos[0]):
            #    
            #    # Binary mask for dilatation of external pixels
            #    pix_reference_mask = [[0 for x in range(ext_region.shape[1])] \
            #                             for y in range(ext_region.shape[0])]
            #    pix_reference_mask[y][x] = 1
            #    pix_reference_mask = np.asarray(pix_reference_mask)
            #    ext_pix_dilated = sc.ndimage.binary_dilation(pix_reference_mask, structure=struct2, iterations=2).astype(int)
            #    
            #    # Intersection with internal pixels
            #    pix_nearest_mask = ext_pix_dilated * (int_region * self.mask)
            #    pos_intersect = np.where(pix_nearest_mask)

            #    # Compute flux vector
            #    for x2, y2 in zip(pos_intersect[1],pos_intersect[0]):
            #        self.computeVectorsFlux(x,y,x2,y2)

            # Verify if there are flux vectors computed for each igram
            if (len(self.vectors_flux_igram1) != 0) \
            and (len(self.vectors_flux_igram2) != 0) \
            and (len(self.vectors_flux_igram3) != 0):

                # Get proportions of flux vectors that correspond to an unw error of k*pi (2pi or 4pi)
                for flux in self.vectors_flux_igram1:
                    if np.sign(flux) * (2*np.pi) - pi_thr < flux < np.sign(flux) * (2*np.pi) + pi_thr:
                        counter_list1 += 1
                    if np.sign(flux) * (4*np.pi) - pi_thr < flux < np.sign(flux) * (4*np.pi) + pi_thr:
                        counter_list1_4pi += 1
                for flux in self.vectors_flux_igram2:
                    if np.sign(flux) * (2*np.pi) - pi_thr < flux < np.sign(flux) * (2*np.pi) + pi_thr:
                        counter_list2 += 1
                    if np.sign(flux) * (4*np.pi) - pi_thr < flux < np.sign(flux) * (4*np.pi) + pi_thr:
                        counter_list2_4pi += 1
                for flux in self.vectors_flux_igram3:
                    if np.sign(flux) * (2*np.pi) - pi_thr < flux < np.sign(flux) * (2*np.pi) + pi_thr:
                        counter_list3 += 1
                    if np.sign(flux) * (4*np.pi) - pi_thr < flux < np.sign(flux) * (4*np.pi) + pi_thr:
                        counter_list3_4pi += 1

                # Define if it is a 2pi or 4pi error
                sum_2pi = counter_list1 + counter_list2 + counter_list3
                sum_4pi = counter_list1_4pi + counter_list2_4pi + counter_list3_4pi
                if sum_2pi >= sum_4pi:
                    k = 2
                    proportions = [round(float(counter_list1)/len(self.vectors_flux_igram1),2),
                                   round(float(counter_list2)/len(self.vectors_flux_igram2),2),
                                   round(float(counter_list3)/len(self.vectors_flux_igram3),2)]
                if sum_4pi > sum_2pi:
                    k = 4
                    proportions = [round(float(counter_list1_4pi)/len(self.vectors_flux_igram1),2),
                                   round(float(counter_list2_4pi)/len(self.vectors_flux_igram2),2),
                                   round(float(counter_list3_4pi)/len(self.vectors_flux_igram3),2)]

                print('Proportions: {} around pixel (c, l): {},{}'.format(proportions,int_region_pos[1][0],int_region_pos[0][0]))

                # Flux proportions to choose which igram to correct
                x = [i_tmp for i_tmp in proportions]
                x_arr = np.array(x) 

                #------------ CASE 1: Detection in only one igram, correct it 

                # Only one proportion is greater than min_flux
                flag_fluxmethodok = False
                if sum(x_arr > min_flux) == 1: 
                    # Get the position of igram to correct in list 
                    pos_xok = int(np.where((x_arr > min_flux))[0]) 
                    flag_fluxmethodok = True  
                # Two proportions are greater than min_flux
                if sum(x_arr > min_flux) == 2: 
                    pos_xok_tmp = np.where((x_arr > min_flux))[0] 
                    # Check if they are really different to decide which igram to correct
                    if x[pos_xok_tmp[0]]/x[pos_xok_tmp[1]] > t1_prop \
                    or x[pos_xok_tmp[1]]/x[pos_xok_tmp[0]] > t1_prop: 
                        # Get the position of igram to correct in list
                        pos_xok = x.index(max(x))  
                        flag_fluxmethodok = True
                # Correct the igram 
                if flag_fluxmethodok:
                    print('Correction with flux method for error {}: {}'.format(i,x))
                    self.igrams2correct.append(pos_xok) 
                    self.pi2correct.append(k) 
                    if pos_xok == 0: 
                        self.signs2correct.append(np.median(self.vectors_flux_igram1))
                    if pos_xok == 1: 
                        self.signs2correct.append(np.median(self.vectors_flux_igram2))
                    if pos_xok == 2: 
                        self.signs2correct.append(np.median(self.vectors_flux_igram3))

                #------------ CASE 2: Detection not robust, apply mean closure method to detect which igram to correct

                # Three proportions are greater or lesser than min_flux
                # or two proportions but not so different so we cannot decide
                if not flag_fluxmethodok:
                #if sum(x_arr > min_flux) == 3 or sum(x_arr <= min_flux) == 3:
                    # Check if the three proportions are not equal
                    if x[0] == x[1] == x[2]:
                        print("No correction for error: {}, all flux vectors null".format(i))
                        self.igrams2correct.append(None)
                        self.signs2correct.append(None)
                        self.pi2correct.append(None)
                    # Perform mean closure method
                    else:
                        # Check if igram in min 2 triplets
                        if len(self.igrams[0].tripletsIgram) > 1 \
                        and len(self.igrams[1].tripletsIgram) > 1 \
                        and len(self.igrams[2].tripletsIgram) > 1:
                            # Pixel positions of the unw err
                            positions_err = np.where(self.labels_errors == error)
                            # Get mean closure for each error pixels and for each igram
                            for x_pos,y_pos in zip(positions_err[0], positions_err[1]):
                                score_val_igram1.append(self.igrams[0].igram_meanClosure[x_pos,y_pos])
                                score_val_igram2.append(self.igrams[1].igram_meanClosure[x_pos,y_pos])
                                score_val_igram3.append(self.igrams[2].igram_meanClosure[x_pos,y_pos])
                            # Convert to array
                            score_val_igram1_ar = (np.asarray(score_val_igram1)).flatten() 
                            score_val_igram2_ar = (np.asarray(score_val_igram2)).flatten()
                            score_val_igram3_ar = (np.asarray(score_val_igram3)).flatten()
                            # Keep values greater than 2*pi 
                            #for val in range(0, len(score_val_igram1_ar)):
                            #    if not (k*np.pi) - pi_thr < abs(score_val_igram1_ar[val]) < (k*np.pi) + pi_thr:
                            #        score_val_igram1_ar[val] = 0.
                            #    if not (k*np.pi) - pi_thr < abs(score_val_igram2_ar[val]) < (k*np.pi) + pi_thr:
                            #        score_val_igram2_ar[val] = 0.
                            #    if not (k*np.pi) - pi_thr < abs(score_val_igram3_ar[val]) < (k*np.pi) + pi_thr:
                            #        score_val_igram3_ar[val] = 0.
                            for val in range(0, len(score_val_igram1_ar)):
                                if (2*np.pi) - pi_thr >= score_val_igram1_ar[val] * self.z_error_sign[self.z_errors_labels_ok.index(error)]:
                                    score_val_igram1_ar[val] = 0.
                                if (2*np.pi) - pi_thr >= score_val_igram2_ar[val] * self.z_error_sign[self.z_errors_labels_ok.index(error)]:
                                    score_val_igram2_ar[val] = 0.
                                if (2*np.pi) - pi_thr >= score_val_igram3_ar[val] * self.z_error_sign[self.z_errors_labels_ok.index(error)] * (-1.):
                                    score_val_igram3_ar[val] = 0.

                            # Count non zero values
                            nonzero_igrams = [np.count_nonzero(score_val_igram1_ar),\
                                              np.count_nonzero(score_val_igram2_ar),\
                                              np.count_nonzero(score_val_igram3_ar)]
                            print("Score: {}".format(nonzero_igrams))
                            # Verify if min 50% pixels are non zero values
                            perc_igrams = [float(nonzero_igrams[0]) / len(score_val_igram1_ar),\
                                           float(nonzero_igrams[1]) / len(score_val_igram2_ar),\
                                           float(nonzero_igrams[2]) / len(score_val_igram3_ar)]
                            for perc in range(0, len(perc_igrams)):
                                if perc_igrams[perc] < t2_prop:
                                    nonzero_igrams[perc] = 0.
                            nonzero_igrams_clean = [value for value in nonzero_igrams if value != 0.]
                            # Get the igram of highest mean closure
                            if len(nonzero_igrams_clean) >= 1:
                                score_sort = sorted(nonzero_igrams_clean, reverse=True)
                                # MC method and flux seem to agree
                                if ((len(nonzero_igrams_clean) ==1) or (score_sort[0]/score_sort[1] >= t3_prop)):
                                    if(proportions[nonzero_igrams.index(max(score_sort))] >= minFlux4MC):
                                        print('Correction with MC method for error: {}'.format(i))
                                        self.igrams2correct.append(nonzero_igrams.index(max(score_sort)))
                                        self.pi2correct.append(k)
                                        #if nonzero_igrams.index(max(score_sort)) == 0:
                                        #    self.signs2correct.append(np.median(self.vectors_flux_igram1))
                                        #if nonzero_igrams.index(max(score_sort)) == 1:
                                        #    self.signs2correct.append(np.median(self.vectors_flux_igram2))
                                        #if nonzero_igrams.index(max(score_sort)) == 2:
                                        #    self.signs2correct.append(np.median(self.vectors_flux_igram3))
                                        if nonzero_igrams.index(max(score_sort)) == 0:
                                            self.signs2correct.append(-self.z_error_sign[self.z_errors_labels_ok.index(error)])
                                        if nonzero_igrams.index(max(score_sort)) == 1:
                                            self.signs2correct.append(-self.z_error_sign[self.z_errors_labels_ok.index(error)])
                                        if nonzero_igrams.index(max(score_sort)) == 2:
                                            self.signs2correct.append(self.z_error_sign[self.z_errors_labels_ok.index(error)])
                                    else:
                                        print("No correction for error: {}, discrepancy between MC and flux methods".format(i))
                                        self.igrams2correct.append(None)
                                        self.signs2correct.append(None)
                                        self.pi2correct.append(None)

                                # Cannot perform score method
                                else:
                                    print("No correction for error: {}, not enough contrast between MC values".format(i))
                                    self.igrams2correct.append(None)
                                    self.signs2correct.append(None)
                                    self.pi2correct.append(None)
                            else:
                                print("No correction for error: {}, no MC sufficiently high".format(i))
                                self.igrams2correct.append(None)
                                self.signs2correct.append(None)
                                self.pi2correct.append(None)
                        else:
                            print("No correction for error: {}, not enough loops to compute MC".format(i))
                            self.igrams2correct.append(None)
                            self.signs2correct.append(None)
                            self.pi2correct.append(None)

            # No flux vectors computed for igram, pass the error
            else:
                print("No correction for error: {}".format(i))
                self.igrams2correct.append(None)
                self.signs2correct.append(None)
                self.pi2correct.append(None)

        # All done
        return

#--------------------
# Compute flux vectors
#--------------------

    def computeVectorsFlux(self, x1, y1, x2, y2):
        '''
        Compute a flux vector and add it to a list.

        Args:
            * x1        : x position of external pixel
            * y1        : y position of external pixel
            * x2        : x position of internal pixel
            * y2        : y position of internal pixel
        '''

        # Check if not NaNs
        if not math.isnan(x1) \
        and not math.isnan(x2) \
        and not math.isnan(y1) \
        and not math.isnan(y2):

            # Get igram phase values
            ph1_int = self.files[0].phase[y2, x2]
            ph1_ext = self.files[0].phase[y1, x1]
            ph2_int = self.files[1].phase[y2, x2]
            ph2_ext = self.files[1].phase[y1, x1]
            ph3_int = self.files[2].phase[y2, x2]
            ph3_ext = self.files[2].phase[y1, x1]

            # Compute the flux
            ph1_flux = ph1_ext - ph1_int
            ph2_flux = ph2_ext - ph2_int
            ph3_flux = ph3_ext - ph3_int

            # Save into lists
            self.vectors_flux_igram1.append(ph1_flux)
            self.vectors_flux_igram2.append(ph2_flux)
            self.vectors_flux_igram3.append(ph3_flux)

        # All done
        return
