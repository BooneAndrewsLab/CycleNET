"""
Author: Oren Kraus (https://github.com/okraus, 2013)

"""

from input_queue_whole_screen import ScreenQueue
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import copy
import time
import os


parser = argparse.ArgumentParser(description='Evaluate Screens, cell cycle and localization models')
parser.add_argument("-l", "--LOC_CPKT", help="Path to model/checkpoint for localization network to use")
parser.add_argument("-c", "--CYC_CPKT", help="Path to model/checkpoint for cell cycle network to use")
parser.add_argument("-i", "--INPATH", help="Path to input folder containing labeled images")
parser.add_argument("-o", "--OUTPATH", help="Where to store output csv files")
# parser.add_argument("-n", action="store_true", dest="USE_NON_CROPPED", help="Use non cropped cells")
args = parser.parse_args()

locNetCpkt = args.LOC_CPKT
cycNetCpkt = args.CYC_CPKT
outputPath = args.OUTPATH
screens = [args.INPATH]
# use_non_cropped = args.USE_NON_CROPPED


def proccessCropsLoc(processedBatch,predicted_y,inputs,is_training,sess,keep_prob):
    crop_list = np.zeros((len(processedBatch), 5, 22))
    for crop in range(5):
        images = processedBatch[:, crop, :, :, :]
        tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False, keep_prob:1.0}))
        # print(tmp)
        crop_list[:, crop, :] = tmp[0]

    mean_crops = np.mean(crop_list, 1)
    return mean_crops


def proccessCropsCyc(processedBatch,predicted_y,inputs,is_training,sess,keep_prob):
    crop_list = np.zeros((len(processedBatch), 5, 9))
    for crop in range(5):
        images = processedBatch[:, crop, :, :, :]
        tmp = copy.copy(sess.run([predicted_y], feed_dict={inputs: images, is_training: False, keep_prob:1.0}))
        # print(tmp)
        crop_list[:, crop, :] = tmp[0]

    mean_crops = np.mean(crop_list, 1)
    return mean_crops


def eval():

    ###################################################################################################################
    ### LOAD NETWORKS ###
    #####################
    #LOCALIZATION
    loc = tf.Graph()
    with loc.as_default():
        loc_saver = tf.train.import_meta_graph(locNetCpkt+'.meta')
    locSession = tf.Session(graph=loc)
    loc_saver.restore(locSession, locNetCpkt)

    pred_loc = loc.get_tensor_by_name(u'softmax:0')
    input_loc = loc.get_tensor_by_name(u'input:0')
    is_training_loc = loc.get_tensor_by_name(u'is_training:0')
    keep_prob_loc = loc.get_tensor_by_name(u'Placeholder:0')

    #CELL CYCLE
    cyc = tf.Graph()
    with cyc.as_default():
        cyc_saver = tf.train.import_meta_graph(cycNetCpkt+'.meta')
    cycSession = tf.Session(graph=cyc)
    cyc_saver.restore(cycSession, cycNetCpkt)

    pred_cyc = cyc.get_tensor_by_name(u'softmax:0')
    input_cyc = cyc.get_tensor_by_name(u'input:0')
    is_training_cyc = cyc.get_tensor_by_name(u'is_training:0')
    keep_prob_cyc = cyc.get_tensor_by_name(u'Placeholder:0')


    ###################################################################################################################

    #initialize tf session for Queue
    sess = tf.Session()
    coord = tf.train.Coordinator()

    #global_step = tf.Variable(0, trainable=False)
    dequeueSize = 256

    localizationTerms = ['Actin', 'Bud', 'Bud Neck', 'Bud Periphery', 'Bud Site',
                         'Cell Periphery', 'Cytoplasm', 'Cytoplasmic Foci', 'Eisosomes',
                         'Endoplasmic Reticulum', 'Endosome', 'Golgi', 'Lipid Particles',
                         'Mitochondria', 'None', 'Nuclear Periphery', 'Nucleolus', 'Nucleus',
                         'Peroxisomes', 'Punctate Nuclear', 'Vacuole', 'Vacuole Periphery']
    cycleTerms = ['Early G1', 'Late G1', 'S/G2', 'Metaphase', 'Anaphase', 'Telophase',
                 'Abberent', 'Over_seg', 'Anaphase_defect']

    col_names_output = ['x_loc', 'y_loc', 'cellSize','nucSize',
                       'gfpIntegrated_cell','gfpIntegrated_nuc','gfpIngtegrated_cyt',
                       'gfpMean_cell','gfpMean_nuc','gfpMean_cyt',
                       'gfpStd_cell','gfpStd_nuc','gfpStd_cyt',
                       'gfpMin_cell','gfpMin_nuc','gfpMin_cyt',
                       'gfpMax_cell','gfpMax_nuc','gfpMax_cyt',
                       'gfpMedian_cell','gfpMedian_nuc',
                       'gfpMedian_cyt'] + localizationTerms + cycleTerms
    #col_names_output = ['x_loc', 'y_loc'] + localizationTerms + cycleTerms

    allPred = None
    MAX_CELLS_PER_SCREEN = 5000000

    for screen in screens:
        print("screen", screen)
        # start queue runner
        with tf.device("/cpu:0"):
            queue_runner = ScreenQueue(screen)
            data_image_loc,data_image_cyc, data_coord, data_intense, well_frame = queue_runner.get_inputs(dequeueSize)

        sess.run(tf.global_variables_initializer())

        #start queue
        tf.train.start_queue_runners(sess=sess)
        threads = queue_runner.start_threads(sess=sess, coord=coord, n_threads=4)

        # print('populating queue', sess.run(queue_runner.queue.size()))
        # time.sleep(60)
        # print('populated queue', sess.run(queue_runner.queue.size()))
        # time.sleep(60)
        # print('populated queue', sess.run(queue_runner.queue.size()))


        #global_step = checkpoint_file.split('/')[-1].split('-')[-1]

        del allPred
        allPred = pd.DataFrame(np.zeros((MAX_CELLS_PER_SCREEN,
                                         len(col_names_output))), columns=col_names_output)
        allPred_ind = 0

        wellNamesAll = []

        print('populating queue', sess.run(queue_runner.queue.size()))
        time.sleep(60)
        print('populated queue', sess.run(queue_runner.queue.size()))

        #run through queue with dequeue_size
        while sess.run(queue_runner.queue.size()) > dequeueSize:
            processedBatch_Loc, processedBatch_Cyc, coordUsed, intensityUsed, wellNames = sess.run([data_image_loc,data_image_cyc, data_coord,
                                                                            data_intense, well_frame])
            if len(wellNames)>0:
                print(wellNames[-1],'queue_size',sess.run(queue_runner.queue.size()))
            wellNamesAll.append(wellNames)

            predictedBatch_Loc = proccessCropsLoc(processedBatch=processedBatch_Loc, predicted_y=pred_loc,
                                           inputs=input_loc,is_training=is_training_loc, sess=locSession,keep_prob=keep_prob_loc)
            predictedBatch_Cyc = proccessCropsCyc(processedBatch=processedBatch_Cyc, predicted_y=pred_cyc,
                                               inputs=input_cyc, is_training=is_training_cyc, sess=cycSession,keep_prob=keep_prob_cyc)

            allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack((
                coordUsed, intensityUsed, predictedBatch_Loc,predictedBatch_Cyc))
            #allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack((
            #    coordUsed, predictedBatch_Loc,predictedBatch_Cyc))
            allPred_ind += len(predictedBatch_Loc)

        # process last batch
        with tf.device("/cpu:0"):
            remainingBatch = sess.run(queue_runner.queue.size())
            data_image_cyc, data_image_loc , data_coord, data_intense, well_frame = queue_runner.get_inputs(remainingBatch)

        processedBatch_Loc,processedBatch_Cyc, coordUsed, intensityUsed, wellNames = sess.run([data_image_loc, data_image_cyc, data_coord,
                                                                        data_intense, well_frame])
        if len(wellNames)>0:
            print(wellNames[-1],' last_batch','queue_size',sess.run(queue_runner.queue.size()))
        wellNamesAll.append(wellNames)
        predictedBatch_Loc = proccessCropsLoc(processedBatch=processedBatch_Loc, predicted_y=pred_loc,
                                           inputs=input_loc, is_training=is_training_loc, sess=locSession,keep_prob=keep_prob_loc)
        predictedBatch_Cyc = proccessCropsCyc(processedBatch=processedBatch_Cyc, predicted_y=pred_cyc,
                                       inputs=input_cyc, is_training=is_training_cyc, sess=cycSession,keep_prob=keep_prob_cyc)



        allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack((
            coordUsed, intensityUsed, predictedBatch_Loc, predictedBatch_Cyc))
        #allPred.iloc[allPred_ind:allPred_ind + len(predictedBatch_Loc), :] = np.hstack((
        #    coordUsed, predictedBatch_Loc, predictedBatch_Cyc))
        allPred_ind += len(predictedBatch_Loc)

        allPred = allPred.iloc[:allPred_ind, :]

        #add well data
        allWellNames = np.hstack(wellNamesAll)
        rows = []
        cols = []
        wellIDs = []
        frames =[]
        for well in allWellNames:
            wellName = well
            rows.append(int(wellName[1:3]))
            cols.append(int(wellName[4:6]))
            wellIDs.append(wellName[:6])
            frames.append(int(wellName[7:9]))
        allPred['wellPath'] = allWellNames
        allPred['wellId'] = np.hstack(wellIDs)
        allPred['row'] = np.hstack(rows)
        allPred['col'] = np.hstack(cols)
        allPred['frame'] = np.hstack(frames)


        #add max loc and cyc classes
        maxLocNames = allPred[localizationTerms].idxmax(1)
        maxCycNames = allPred[cycleTerms].idxmax(1)
        allPred['maxLocalization'] = maxLocNames
        allPred['maxCycle'] = maxCycNames

        locCpktBasename = os.path.basename(locNetCpkt)
        cycCpktBasename = os.path.basename(cycNetCpkt)
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        allPred.to_csv(os.path.join(outputPath, '%s_%s_cyc_loc_pred_v1.csv' % (locCpktBasename, cycCpktBasename)),
                       index=False)

        queue_runner.stopQueue(threads, coord, sess)

    sess.close()
    cycSession.close()
    locSession.close()


def main(_):
    eval()


if __name__ == '__main__':
    tf.app.run()

