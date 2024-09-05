"""
Author: Oren Kraus (https://github.com/okraus, 2013)
"""

import h5py
import numpy as np


class Data:
    def __init__(self,
                 folder,
                 keys2fetch,
                 batchSize,
                 ):
        self.numData = 0
        self.batchSize = batchSize
        self.folder = folder
        self.h5data = h5py.File(self.folder, 'r')
        self.keys2fetch = keys2fetch
        h5keys = list(self.h5data.keys())

        self.groupedData = {}
        for key in keys2fetch: self.groupedData[key] = []
        for key in h5keys:
            if any(x in key for x in keys2fetch):
                curInd = [x in key for x in keys2fetch]
                if curInd[0]:
                    self.numData += len(self.h5data[key])
                curKey = keys2fetch[curInd.index(True)]
                self.groupedData[curKey].append(int(key[len(curKey):]))
        for key in keys2fetch: self.groupedData[key].sort()

        self.startInd = 0
        self.stopInd = self.numData
        self.curInd = self.startInd

        assert batchSize<self.numData, "batchSize larger than dataset; batchSize: "+str(batchSize)+" dataSize: "+str(self.numData)

        self.h5chunkSize = len(self.h5data[keys2fetch[0] + '1'])
        self.keySizes = {}
        for key in keys2fetch: self.keySizes[key] = self.h5data[key + '1'].shape[1]

        self.returnArrays = {}
        for key in keys2fetch:
            self.returnArrays[key] = np.zeros((self.batchSize, self.keySizes[key]), dtype=np.float32)

    def getBatch(self):

        if (self.curInd + self.batchSize) >= self.stopInd:
            self.curInd = self.startInd

        startDsetNum = self.curInd / self.h5chunkSize + 1
        startDsetInd = self.curInd % self.h5chunkSize
        endDsetNum = (self.curInd + self.batchSize) / self.h5chunkSize + 1

        for key in self.keys2fetch:
            curInd = 0
            curDset = int(startDsetNum)
            curDsetInd = startDsetInd
            while curInd < self.batchSize:
                dsetShape = self.h5data[key + str(curDset)].shape
                self.returnArrays[key][curInd:min(dsetShape[0] - curDsetInd, self.batchSize + curDsetInd), :] = \
                    self.h5data[key + str(curDset)][curDsetInd:min(dsetShape[0], self.batchSize + curDsetInd), :]
                curDset += 1
                curDsetInd = 0
                curInd += min(dsetShape[0] - curDsetInd, self.batchSize)

        self.curInd += self.batchSize

        return self.returnArrays
