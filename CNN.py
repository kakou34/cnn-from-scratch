import copy
import math
from collections import defaultdict

import numpy as np

from Errors import FilterSizeError
from Errors import InputImageError
from FFFCN import FFFCN


class CNN:

    # initialization, assume stride = 1
    def __init__(self, lr, img_s, op_s, f_size, f_number, mp_size, mp_stride):
        # Calculating the size of FC input (flattened vector)
        c_s = math.floor((img_s - f_size) + 1)
        mp_s = math.floor(((c_s - mp_size) / mp_stride) + 1)
        fc_ins = mp_s * mp_s * f_number
        # initializing the filters and biases randompy
        self.filters = 1 * np.random.rand(f_size, f_size, f_number) - 0.5
        self.biases = 2 * np.random.rand(f_number)
        self.lr = lr
        self.output_size = op_s
        self.mp_size = mp_size
        self.mp_stride = mp_stride
        self.img_size = img_s
        self.filter_size = f_size
        self.filter_number = f_number
        self.fffcn = FFFCN(lr, fc_ins, op_s)
        self.initial = ()

    # Function for convolution
    def conv(self, img, filters, biases):
        # Check if the image is gray scaled and square
        if len(img.shape) > 2 or img.shape[0] != img.shape[1]:
            raise InputImageError
        # Check if filter dimensions are equal and filters have one channel.
        if filters.shape[0] != filters.shape[1] or len(filters.shape) > 3:
            raise FilterSizeError

        # An empty feature map to hold the convolved images.
        ncr = math.floor((img.shape[0] - filters.shape[0]) + 1)
        ncc = math.floor((img.shape[1] - filters.shape[0]) + 1)
        filter_no = filters.shape[2]
        feature_maps = np.zeros(shape=(ncr, ncc, filter_no))

        # Convolving the image by the filter(s).
        for n in range(filter_no):
            # get the nth filter and bias
            current_filter = filters[:, :, n]
            current_bias = biases[n]
            # get the convolution of the input image using the current filter and bias
            conv_map = self.convolution(img, current_filter, current_bias)
            # Holding feature map with the current filter.
            feature_maps[:, :, n] = conv_map

        return feature_maps

    # Helper function for Convolution
    def convolution(self, img, filter, bias):
        # Finding filter size
        filter_size = filter.shape[0]
        # Finding the size of the resulting convolved image
        ncr = math.floor((img.shape[0] - filter_size) + 1)
        ncc = math.floor(img.shape[1] - filter_size + 1)
        # initializing the resulting convolved image
        result = np.zeros(shape=(ncr, ncc))
        # Looping through the image to apply the convolution operation.
        r_idx = 0
        for r in range(img.shape[0] - filter_size + 1):
            c_idx = 0
            for c in range(img.shape[1] - filter_size + 1):
                # Getting the current region to get multiplied with the filter.
                current_region = img[r:r + filter_size, c:c + filter_size]
                # Element-wise multiplication between the current region and the filter.
                current_result = current_region * filter
                # Summing the result of multiplication and adding the bias.
                conv_sum = np.sum(current_result) + bias
                # Saving the resulting value in the corresponding pixel of the convolved image
                result[r_idx, c_idx] = conv_sum
                # proceed to the next column
                c_idx += 1
            # proceed to the next row
            r_idx += 1
        return result

    # RelU activation function
    def relu(self, feature_map):
        # preparing the output
        relu_out = np.zeros(feature_map.shape)
        # iterate over each convolved image
        for map_num in range(feature_map.shape[2]):
            # iterate over each row of the current convolved image
            for r in range(feature_map.shape[0]):
                # iterate over each column of the current convolved image
                for c in range(feature_map.shape[1]):
                    # take the positive pixel values as they are and replace negative ones with 0
                    res = np.maximum(feature_map[r, c, map_num], 0)
                    relu_out[r, c, map_num] = res
        return relu_out

    # Maxpooling function
    def maxpooling(self, feature_map, size=2, stride=2):
        # Preparing the output of the pooling operation.
        nr = math.floor(((feature_map.shape[0] - size) / stride) + 1)
        nc = math.floor(((feature_map.shape[1] - size) / stride) + 1)
        ni = feature_map.shape[2]
        pool_out = np.zeros(shape=(nr, nc, ni))
        indexes = defaultdict()  # Saving indexes of max values for backpropagation
        # Iterating over the feature map to apply maxpooling
        for n in range(ni):
            # getting the current convolved image from the feature map
            current_chan = feature_map[:, :, n]
            r_idx = 0
            for r in range(0, feature_map.shape[0] - size + 1, stride):
                c_idx = 0
                for c in range(0, feature_map.shape[1] - size + 1, stride):
                    # Getting the current region
                    current_region = current_chan[r:r + size, c:c + size]
                    # finding the max value in current_region and assigning it to the corresponding pixel in
                    # the pooled image
                    pool_out[r_idx, c_idx, n] = np.max(current_region)
                    # saving the indexes of the maximum value
                    max_idx = np.unravel_index(np.argmax(current_region, axis=None), current_region.shape)
                    indexes[(r_idx, c_idx, n)] = (max_idx[0] + r, max_idx[1] + c, n)
                    c_idx += 1
                r_idx += 1
        return pool_out, indexes

    # Flattening
    def flatten(self, pooled_map):
        # finding the size of the flattened vector
        flat = np.zeros(shape=(pooled_map.size, 1))
        idx = 0
        # Iterating over each maxpooled image in the map
        for n in range(pooled_map.shape[2]):
            # iterating  over each row of the current image
            for r in range(pooled_map.shape[0]):
                # iterating  over each column of the current image
                for c in range(pooled_map.shape[1]):
                    # assigning the current pixel's value (r, c,n) value to the corresponding cell in the vector
                    flat[idx] = pooled_map[r, c, n]
                    idx += 1
        return flat

    def predict(self, img):
        # convolve the image
        conv_fm = self.conv(img, self.filters, self.biases)
        # applying the relU activation function on the feature map obtained
        conv_relu = self.relu(conv_fm)
        # maxpooling operation
        maxp_fm, indexes = self.maxpooling(conv_relu, self.mp_size, self.mp_stride)
        # flattening the result of maxpooling
        fc_input = self.flatten(maxp_fm)
        # applying the flattened features as input to a Fully Connected layer and obtaining the output
        output = self.fffcn.predict(fc_input)
        # take the prediction that has the highest probability (softmax value)
        predicted = np.argmax(output, axis=0)
        return predicted

    def backpropagate(self, img, yc_idx):
        # convolving the image
        conv_fm = self.conv(img, self.filters, self.biases)
        # relu activation function
        conv_relu = self.relu(conv_fm)
        # maxpooling
        maxp_fm, indexes = self.maxpooling(conv_relu, self.mp_size, self.mp_stride)
        # flattening
        fc_input = self.flatten(maxp_fm)

        # backpropagating the fully connected layer and getting the input delta values
        deltas_in = self.fffcn.backpropagation(fc_input, yc_idx)
        # removing the delta for the bias
        deltas_in = np.delete(deltas_in, 0)
        # finding delta maxpooling by assigning each delta input to the corresponding cell in the maxpool
        delta_mp = np.zeros(shape=maxp_fm.shape)
        idx = 0
        for n in range(delta_mp.shape[2]):
            for r in range(delta_mp.shape[0]):
                for c in range(delta_mp.shape[1]):
                    delta_mp[r, c, n] = deltas_in[idx]
                    idx += 1
        # finding delta convolution using the previously saved indexes
        delta_c = np.zeros(shape=conv_fm.shape)
        for n in range(delta_mp.shape[2]):
            for r in range(delta_mp.shape[0]):
                for c in range(delta_mp.shape[1]):
                    index_t = indexes[(r, c, n)]
                    # checking the result of relu function for conv(r, c, n)
                    relu_val = conv_relu[index_t[0], index_t[1], index_t[2]]
                    if relu_val == 0:  # derivative of relU = 0, set local gradient to 0
                        delta_c[index_t[0], index_t[1], index_t[2]] = 0
                    else:  # derivative of relU = 1, keep local gradient as it is
                        delta_c[index_t[0], index_t[1], index_t[2]] = delta_mp[r, c, n]
        # obtaining delta filter = conv(image, delta_c)
        delta_filter = self.conv(img, delta_c, np.zeros(shape=self.biases.shape))

        # delta bias is the sum of each element of delta_c
        delta_bias = np.sum((np.sum(delta_c, axis=0)), axis=0)
        # updating the filters and biases
        current_f = copy.deepcopy(self.filters)

        new_f = current_f - delta_filter * self.lr
        self.filters = new_f
        self.biases = self.biases - delta_bias * self.lr

    # function to train the network
    def train(self, trainX, trainY, epochs):
        for epoch in range(epochs):
            for i in range(len(trainY)):
                self.backpropagate(trainX[i], trainY[i])

    # function to test the network, performance metric: accuracy
    def test(self, testX, testY):
        predictions = np.zeros(shape=testY.shape)
        for i in range(len(testY)):
            x = self.predict(testX[i])
            predictions[i] = x
        compare = (testY == predictions)
        accuracy = np.sum(compare) / len(testY)
        return accuracy
