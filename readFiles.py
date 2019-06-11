#-------------------------------------------------
#-------------------------------------------------
#
#   Class readFiles
#
#-------------------------------------------------
#-------------------------------------------------

# Import all
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided

#--------------------
# Class definition
#--------------------

class readFiles(object):
    '''
    A class to read:
        - interferogram
        - mask

    Implemented for:
        - ISCE
        - Roi_pac

    Feel free to add your own reader.

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
        '''

        # All done
        return


#--------------------
# Read ISCE files
#--------------------

    def readIsce(self, dates, filesDir, igramName):
        '''
        Read files from ISCE software.

        Args:
            * dates             : Dates of the interferogram (list)
            * filesDir          : Files directory (str)
            * igramName         : Interferogram filename (str)
        '''

        # Import ISCE stuff
        import isce
        import isceobj
        from isceobj.Util.ImageUtil import ImageLib as IML

        #-------------------------------------------------
        # READ IGRAM
        
        # Igram filenames (wrapped, unwrapped and corrected)
        ifgInt = '{}/{}_{}/{}.int'.format(filesDir,dates[0],dates[1],os.path.splitext(os.path.basename(igramName))[0])
        ifgFile = '{}/{}_{}/{}'.format(filesDir,dates[0],dates[1],igramName)
        ifgFile_corrected = '{}_corrected{}'.format(os.path.splitext(ifgFile)[0],os.path.splitext(ifgFile)[1])

        # Igram.unw xml infos
        imgIfgFile = isceobj.createImage()
        imgIfgFile.load(ifgFile + '.xml')
        bands = imgIfgFile.bands
        length = imgIfgFile.getLength()
        width = imgIfgFile.getWidth()
        scheme = imgIfgFile.scheme

        # Igram.int xml infos
        imgInt = isceobj.createImage()
        imgInt.load(ifgInt + '.xml')
        bandsInt = imgInt.bands
        schemeInt = imgInt.scheme
        datatypeInt = IML.NUMPY_type(imgInt.dataType)

        # Igram not yet corrected
        if not os.path.exists(ifgFile_corrected):

            # Read igram.unw
            data = IML.memmap(ifgFile, mode='readonly', nchannels=bands, nyy=length, nxx=width, scheme=scheme)

            # Create the corrected igram
            data_corrected = IML.memmap(ifgFile_corrected, mode='write', nchannels=bands, nyy=length, nxx=width, scheme=scheme)

            # Copy igram.unw in the new file
            data_corrected.bands[0][:,:] = data.bands[0][:,:]
            data_corrected.bands[1][:,:] = data.bands[1][:,:]

            # New xml infos
            data_correctedXml = isceobj.Image.createUnwImage()
            data_correctedXml.bands = 2
            data_correctedXml.setWidth(width)
            data_correctedXml.setLength(length)
            data_correctedXml.setFilename(ifgFile_corrected)
            data_correctedXml.renderHdr()
            data_correctedXml.renderVRT()

            # Flag
            flag_corr = True

        # Igram already corrected, start from it
        else:

            # Read igram.unw already corrected
            data = IML.memmap(ifgFile_corrected, mode='readwrite', nchannels=bands, nyy=length, nxx=width, scheme=scheme)

            # Flag 
            flag_corr = False

        # Read wrap igram where to compute misclosure
        data_int = IML.memmap(ifgInt, mode='readonly', nchannels=bandsInt, dataType=datatypeInt, nyy=length, nxx=width, scheme=schemeInt)

        #-------------------------------------------------
        # READ MASK

        # Initialize the mask
        mask = np.zeros((length, width))

        #-------------------------------------------------
        # SAVE

        # Save things
        if flag_corr:
            self.phase = data_corrected.bands[1]
        else:
            self.phase = data.bands[1]
        self.phase_int = data_int.bands[0]
        self.dates = dates
        self.igramsDir = filesDir
        self.igramName = igramName
        self.width = width
        self.length = length
        self.ifgFile_corrected = ifgFile_corrected
        
        # Close files
        del data
        del data_int

        # All done
        return
        
#--------------------
# Read ROI_PAC files
#--------------------

    def readRoiPac(self, dates, filesDir, igramName, length_max):
        '''
        Read files from roi_pac software.

        Args:
            * dates             : Dates of the interferogram (list)
            * filesDir          : Files directory (str)
            * igramName         : Interferogram filename, without the date (str)
            * length_max        : Maximum length of igrams (int)
        '''
    
        from osgeo import gdal

        #-------------------------------------------------
        # READ IGRAM

        # Format of the igramName (add the date in the name)
        b = igramName.split('_')[:]
        b.insert(2, "{}-{}".format(dates[0],dates[1]))
        igramName_tmp = "_".join(b)
        igramName_tmp_int = "_".join(b[:-1])

        # Igram filenames (wrapped, unwrapped and corrected)
        ifgInt = '{}/{}_{}/{}.int'.format(filesDir,dates[0],dates[1],os.path.splitext(os.path.basename(igramName_tmp_int))[0])
        ifgFile = '{}/{}_{}/{}'.format(filesDir,dates[0],dates[1],igramName_tmp)
        ifgFile_corrected = '{}_corrected{}'.format(os.path.splitext(ifgFile)[0],os.path.splitext(ifgFile)[1])

        # CREATING EVEN INDICES
        even_ind = np.arange(length_max) * 2

        # List of memmap objects (amplitude and phase)
        acc = []

        # Igram not yet corrected
        if not os.path.exists(ifgFile_corrected):

            # Read .unw and copy into the new file
            ds = gdal.Open(r'{}'.format(ifgFile))
            # The 2 following lines create the .rsc file associated to the corrected interferogram
            driver = ds.GetDriver()
            outDs = driver.Create(ifgFile_corrected,ds.RasterXSize,length_max, 2, gdal.GDT_Float32)
            # Reshape with max length
            newrows = np.zeros((2*(length_max - ds.RasterYSize), ds.RasterXSize))
            data = np.memmap(ifgFile, dtype='float32', mode='r', shape=(2 * ds.RasterYSize, ds.RasterXSize)) # lecture du BIL

            # Create the corrected igram
            nshape = 2*length_max*ds.RasterXSize
            data_corrected = np.memmap(ifgFile_corrected, dtype='float32', mode='readwrite', shape = (nshape,))
            data_corrected[2*ds.RasterYSize*ds.RasterXSize:2*length_max*ds.RasterXSize] = np.zeros(2*(length_max - ds.RasterYSize)*ds.RasterXSize)
            data_corrected[0:2*ds.RasterYSize*ds.RasterXSize] = np.reshape(data, 2*ds.RasterYSize*ds.RasterXSize)

            # Cut data_corrected in 2 bands for np.memmap
            fsize = np.zeros(1, dtype='float32').itemsize
            nstrides = (2*ds.RasterXSize*fsize, fsize)
            for band in range(2):
                noffset = band*ds.RasterXSize
                tmap = data_corrected[noffset:]
                fmap = as_strided(tmap, shape=(length_max,ds.RasterXSize), strides=nstrides)
                acc.append(fmap)

            # Close files
            del data

        else:
            # Read igram.unw already corrected
            ds = gdal.Open(r'{}'.format(ifgFile_corrected))
            nshape = 2*length_max*ds.RasterXSize
            data_corrected = np.memmap(ifgFile_corrected, dtype='float32', mode='readwrite', shape = (nshape,))

            # Cut data_corrected in 2 bands for np.memmap
            fsize = np.zeros(1, dtype='float32').itemsize
            nstrides = (2*ds.RasterXSize*fsize, fsize)
            for band in range(2):
                noffset = band*ds.RasterXSize
                tmap = data_corrected[noffset:]
                fmap = as_strided(tmap, shape=(length_max,ds.RasterXSize), strides=nstrides)
                acc.append(fmap)

        # Read .int
        ds_int = gdal.Open(r'{}'.format(ifgInt))
        data_int = ds_int.GetRasterBand(1).ReadAsArray()
        if ds_int.RasterYSize != length_max:
            newrows = np.zeros(((length_max - ds_int.RasterYSize), ds_int.RasterXSize))
            data_int = np.vstack([data_int, newrows])
        
        #-------------------------------------------------
        # CREATE MASK

        # Initialize the mask
        mask = np.zeros((length_max, ds.RasterXSize))

        # Mask constructed from the .unw
        positions_1 = np.where(acc[1] != 0.)
        for x,y in zip(positions_1[0],positions_1[1]):
            mask[x,y] = 1

        #-------------------------------------------------
        # SAVE

        # Save things
        self.phase = acc[1]
        self.phase_int = data_int
        self.dates = dates
        self.igramsDir = filesDir
        self.igramName = igramName
        self.mask = mask
        self.width = ds.RasterXSize
        self.length = length_max
        self.ifgFile_corrected = ifgFile_corrected

        # Close files
        del data_int

        # All done
        return
