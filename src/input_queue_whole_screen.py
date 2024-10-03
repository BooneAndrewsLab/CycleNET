"""
Author: Oren Kraus (https://github.com/okraus, 2013)
Edited by: Myra Paz Masinas (Andrews and Boone Lab, July 2023)
"""

import preprocess_images as procIm
import tensorflow as tf
import numpy as np
import threading
import glob
import os


def getSpecificChannels(flatImageData,channels,imageSize=64):
    return np.hstack(([flatImageData[:,c*imageSize**2:(c+1)*imageSize**2] for c in channels]))


class ScreenQueue:
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """

    def __init__(self, screen):
        self.localizationTerms = ['Actin', 'Bud', 'Bud Neck', 'Bud Periphery', 'Bud Site',
                                   'Cell Periphery', 'Cytoplasm', 'Cytoplasmic Foci', 'Eisosomes',
                                   'Endoplasmic Reticulum', 'Endosome', 'Golgi', 'Lipid Particles',
                                   'Mitochondria', 'None', 'Nuclear Periphery', 'Nucleolus', 'Nucleus',
                                   'Peroxisomes', 'Punctate Nuclear', 'Vacuole', 'Vacuole Periphery']

        self.data_intense_names = ['cellSize (pixel)','nucSize (pixel)',
                       'gfpIntegrated_cell','gfpIntegrated_nuc','gfpIntegrated_cyt',
                       'gfpMean_cell','gfpMean_nuc','gfpMean_cyt',
                       'gfpStd_cell','gfpStd_nuc','gfpStd_cyt',
                       'gfpMin_cell','gfpMin_nuc','gfpMin_cyt',
                       'gfpMax_cell','gfpMax_nuc','gfpMax_cyt',
                       'gfpMedian_cell','gfpMedian_nuc',
                       'gfpMedian_cyt']
                       
        self.basePath = screen + '/'

        #self.wells = np.unique([seq[:-2] for seq in GFP_images])

        self.wells = sorted([os.path.basename(x) for x in glob.glob(self.basePath+'*_labeled.npy')])
        #self.test_count = 8
        #self.wells = self.wells[:self.test_count]

        self.cropSize = 60
        self.imSize = 64
        self.numClasses = 22
        self.numChan = 3

        self.data_image_cyc = tf.placeholder(dtype=tf.float32, shape=[None, 5,
                                                                  self.cropSize,
                                                                  self.cropSize,
                                                                  self.numChan])
        self.data_image_loc = tf.placeholder(dtype=tf.float32, shape=[None, 5,
                                                                  self.cropSize,
                                                                  self.cropSize,
                                                                  self.numChan])
        self.data_coord = tf.placeholder(dtype=tf.float32, shape=[None, 2])
        self.data_intense = tf.placeholder(dtype=tf.float32, shape=[None, len(self.data_intense_names)])
        self.data_well = tf.placeholder(dtype=tf.string, shape=[None])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.
        self.queue = tf.FIFOQueue(shapes=[[5, self.cropSize, self.cropSize, self.numChan],
                                          [5, self.cropSize, self.cropSize, self.numChan],
                                          [2],
                                          [len(self.data_intense_names)],
                                          []],
                                  dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.string],
                                  capacity=2500)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.data_image_cyc,
                                                   self.data_image_loc,
                                                   self.data_coord,
                                                   self.data_intense,
                                                   self.data_well])

    def well_iterator(self, wells):
        for i in range(len(wells)):
            yield wells[i]


    def get_inputs(self, batch_size):
        """
        Return's tensors containing a batch of images and labels
        """
        data_image_cyc, data_image_loc, data_coord, data_intense, data_well = self.queue.dequeue_many(batch_size)
        return data_image_cyc, data_image_loc, data_coord, data_intense, data_well


    def thread_main(self, sess, coord, wells):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        # with coord.stop_on_exception():

        print('enqueue ',wells[0],wells[-1],len(wells))

        # while not coord.should_stop():
        for well in self.well_iterator(wells):

            try:
                print('enqueue ',well)
                # check if any cells
                coord_npy = np.load(os.path.join(self.basePath, well.replace('.npy', '_coords.npy'))) # np.load(self.basePath+'CellCoord/'+well[:-6]+'.npy')
                # print(os.path.join(self.basePath, well.replace('.npy', '_coords.npy')))
                if len(coord_npy)>0:
                    data_image_cyc, data_image_loc, data_coord, data_intense, well_frame = self.processWell(well)
                    sess.run(self.enqueue_op, feed_dict={self.data_image_cyc: data_image_cyc,
                                                         self.data_image_loc: data_image_loc,
                                                         self.data_coord: data_coord,
                                                         self.data_intense: data_intense,
                                                         self.data_well: well_frame})
                else:
                    print('no cells:',well,coord_npy.shape)

            except:
                # Report exceptions to the coordinator.
                coord.request_stop()
                sess.run(self.queue.close(cancel_pending_enqueues=True))
                raise


    def start_threads(self, sess, coord, input_wells=None, n_threads=1):
        """ Start background threads to feed queue """

        wells = input_wells or self.wells
        threads = []
        n = len(wells) // n_threads
        print('wells & n', wells, n)

        well_chunks = [wells[i:i + n] for i in range(0, len(wells), n)]
        for i in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, coord, well_chunks[i]))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


    def stopQueue(self, threads, coordinator, session):
        coordinator.request_stop()
        session.run(self.queue.close(cancel_pending_enqueues=True))
        coordinator.join(threads)


    def processWell(self, well):

        GFP_CHAN = 1 #gfp protein
        SEG_CHAN = 3 #binary seg
        FAR_RED_CHAN = 0 #septin/nuc
        RED_CHAN = 2 #cyto

        ### load from jpeg instead because HOwt flex files were stored in 8bit ###
        ###### switch back to flex, rescale to 0-1 by stretching

        # load data from previous segmentation
        #### correct dtype np.float16->np.float64
        cropped_npy = np.float64(np.load(self.basePath+well))
        #non_cropped_npy = np.float64(np.load(self.basePath+'CellData_non_cropped_fullcell/'+well))
        coord_npy = np.load(os.path.join(self.basePath, well.replace('.npy', '_coords.npy')))
        coord_npy_corrected = np.array([a[-2:] for a in coord_npy])
        wellNames = [well[:12]]*len(coord_npy)

        cycImageData = getSpecificChannels(cropped_npy, [FAR_RED_CHAN, RED_CHAN, RED_CHAN])
        locImageData = getSpecificChannels(cropped_npy, [FAR_RED_CHAN, GFP_CHAN, RED_CHAN])
        
        # get GFP and size stats
        gfp_chan = getSpecificChannels(cropped_npy, [GFP_CHAN])
        seg_chan = getSpecificChannels(cropped_npy, [SEG_CHAN])

        cellMask = seg_chan > .5  # cyto + nuc mask
        nucMask = seg_chan > 1.5  # nuc mask
        cytoMask = np.logical_and(seg_chan>0.5,seg_chan < 1.5) # cyto only mask
        cellSize = cellMask.sum(1,keepdims=True)
        nucSize = nucMask.sum(1,keepdims=True)
        # masked arrays of gfp_chan
        greenChanCell = np.ma.masked_array(gfp_chan,np.logical_not(cellMask))
        greenChanNuc = np.ma.masked_array(gfp_chan,np.logical_not(nucMask))
        greenChanCyt = np.ma.masked_array(gfp_chan,np.logical_not(cytoMask))
        # gfp_chan stats
        gfpIntegrated_cell = greenChanCell.sum(1,keepdims=True).data
        gfpIntegrated_nuc = greenChanNuc.sum(1,keepdims=True).data
        gfpIntegrated_cyt = greenChanCyt.sum(1,keepdims=True).data
        gfpMean_cell = greenChanCell.mean(1,keepdims=True).data
        gfpMean_nuc = greenChanNuc.mean(1,keepdims=True).data
        gfpMean_cyt = greenChanCyt.mean(1,keepdims=True).data
        gfpStd_cell = greenChanCell.std(1,keepdims=True).data
        gfpStd_nuc = greenChanNuc.std(1,keepdims=True).data
        gfpStd_cyt = greenChanCyt.std(1,keepdims=True).data
        gfpMin_cell = np.ma.min(greenChanCell,1,keepdims=True).data
        gfpMin_nuc =  np.ma.min(greenChanNuc,1,keepdims=True).data
        gfpMin_cyt =  np.ma.min(greenChanCyt,1,keepdims=True).data
        gfpMax_cell = np.ma.max(greenChanCell,1,keepdims=True).data
        gfpMax_nuc = np.ma.max(greenChanNuc,1,keepdims=True).data
        gfpMax_cyt = np.ma.max(greenChanCyt,1,keepdims=True).data
        gfpMedian_cell = np.ma.median(greenChanCell,1,keepdims=True).data
        gfpMedian_nuc = np.ma.median(greenChanNuc,1,keepdims=True).data
        gfpMedian_cyt = np.ma.median(greenChanCyt,1,keepdims=True).data

        intensityUsed = np.hstack((cellSize,nucSize,
                                   gfpIntegrated_cell,gfpIntegrated_nuc,gfpIntegrated_cyt,
                                   gfpMean_cell,gfpMean_nuc,gfpMean_cyt,
                                   gfpStd_cell,gfpStd_nuc,gfpStd_cyt,
                                   gfpMin_cell,gfpMin_nuc,gfpMin_cyt,
                                   gfpMax_cell,gfpMax_nuc,gfpMax_cyt,
                                   gfpMedian_cell,gfpMedian_nuc,gfpMedian_cyt))
        #intensityUsed = []


        ### stretch flex files to be between 0  - 1
        stretchLow = 0.1  # stretch channels lower percentile
        stretchHigh = 99.9  # stretch channels upper percentile
        processedBatch_Cyc = procIm.preProcessTestImages(cycImageData,
                                                     self.imSize, self.cropSize, self.numChan,
                                                     rescale=False, stretch=True,
                                                     means=None, stds=None,
                                                     stretchLow=stretchLow, stretchHigh=stretchHigh)

        processedBatch_Loc = procIm.preProcessTestImages(locImageData,
                                                     self.imSize, self.cropSize, self.numChan,
                                                     rescale=False, stretch=True,
                                                     means=None, stds=None,
                                                     stretchLow=stretchLow, stretchHigh=stretchHigh)

        return processedBatch_Loc, processedBatch_Cyc, coord_npy_corrected, intensityUsed, wellNames

