# Image compression by Neural Gas

import numpy as np
from scipy import misc, spatial
from math import log, exp, pow
from sklearn.metrics import mean_squared_error
import cv2
from math import sqrt
from scipy.cluster.vq import vq
import scipy.misc
import sys


def mse(image_a, image_b):
    # calculate mean square error between two images
    err = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    return err


def psnr(a):

    tmp = (255*255)/float(a)
    return 10*log(tmp, 10)


class NG(object):

    def __init__(self, cb_size, dimensions, iterations, number_of_input_vectors, alp_i, alp_f,
                 lam_i, lam_f):

        self.number_of_clusters = int(cb_size)
        self.dimensions = dimensions
        self.number_of_iterations = int(iterations)
        self.alpha_i = alp_i
        self.alpha_f = alp_f
        self.lambda_i = lam_i
        self.lambda_f = lam_f
        self.number_of_input_vectors = number_of_input_vectors

        self.weight_vectors = np.random.uniform(0, 255, (self.number_of_clusters, self.dimensions))

    def get_k(self, input_vector, weights):

        distance_from_input = [np.linalg.norm(input_vector - weight_vector) for weight_vector in weights]
        sorted_indices = sorted(range(len(distance_from_input)), key=lambda j: distance_from_input[j])
        k = [sorted_indices.index(ind) for ind in range(0, self.number_of_clusters)]
        return np.array(k)

    def update_weights(self, iter_no, k, input_data):

        alpha_op = self.alpha_i * pow(self.alpha_f/float(self.alpha_i), iter_no/float(self.number_of_iterations))
        lambda_op = self.lambda_i * pow(self.lambda_f/float(self.lambda_i), iter_no/float(self.number_of_iterations))

        neighbourhood_function = [exp(-val / float(lambda_op)) for val in k]
        final_learning_rate = [alpha_op * val for val in neighbourhood_function]
        for l in range(self.number_of_clusters):
            weight_delta = [val*final_learning_rate[l] for val in (input_data - self.weight_vectors[l])]
            updated_weight = self.weight_vectors[l] + np.array(weight_delta)
            self.weight_vectors[l] = updated_weight

    def get_distortion(self, input_data, reconstruction_values):

        global image_width, image_height, block_height, block_height, image, codebook_size

        image_vector_indices, distance = vq(input_data, reconstruction_values)

        image_after_compression = np.zeros([image_width, image_height], dtype="uint8")
        for index, image_vector in enumerate(input_data):
            start_row = int(index / (image_width / block_width)) * block_height
            end_row = start_row + block_height
            start_column = (index * block_width) % image_width
            end_column = start_column + block_width
            image_after_compression[start_row:end_row, start_column:end_column] = \
                np.reshape(reconstruction_values[image_vector_indices[index]],
                           (block_width, block_height))

        output_image_name = "CB_size=" + str(codebook_size) + ".png"
        scipy.misc.imsave(output_image_name, image_after_compression)

        return psnr(mse(image, image_after_compression))

    def train(self, input_data):

        epoch = 1
        for iteration_no in range(1, self.number_of_iterations+1):
            input_vector_index = (iteration_no-1) % self.number_of_input_vectors
            k = self.get_k(input_data[input_vector_index], self.weight_vectors)
            self.update_weights(iteration_no, k, input_data[input_vector_index])
            if (iteration_no % self.number_of_input_vectors) == 0:
                epoch += 1
        return self.weight_vectors

# source image
image_location = sys.argv[1]
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
image_height = len(image)
image_width = len(image[0])

# dimension of the vector
block_width = int(sys.argv[3])
block_height = int(sys.argv[4])
vector_dimension = block_width*block_height

# dividing the image into 4*4 blocks of pixels
image_vectors = []
for i in range(0, image_height, block_height):
    for j in range(0, image_width, block_width):
        image_vectors.append(np.reshape(image[i:i+block_width, j:j+block_height], vector_dimension))
image_vectors = np.asarray(image_vectors).astype(float)
number_of_image_vectors = image_vectors.shape[0]

bits_per_codevector = int(sys.argv[2])
codebook_size = pow(2, bits_per_codevector)

epochs = int(sys.argv[5])

number_of_iterations = epochs * number_of_image_vectors

epsilon_i = float(sys.argv[6])
epsilon_f = float(sys.argv[7])
tau_i = float(sys.argv[8])
tau_f = float(sys.argv[9])

ng = NG(codebook_size, vector_dimension, number_of_iterations, number_of_image_vectors, epsilon_i, epsilon_f,
        tau_i, tau_f)

reconstruction_values = ng.train(image_vectors)

image_vector_indices, distance = vq(image_vectors, reconstruction_values)

image_after_compression = np.zeros([image_width, image_height], dtype="uint8")
image_vectors_after_compression = np.zeros([number_of_image_vectors, vector_dimension], dtype="uint8")
for index, image_vector in enumerate(image_vectors):
    image_vectors_after_compression[index] = reconstruction_values[image_vector_indices[index]]
    start_row = int(index / (image_width / block_width)) * block_height
    end_row = start_row + block_height
    start_column = (index * block_width) % image_width
    end_column = start_column + block_width
    image_after_compression[start_row:end_row, start_column:end_column] = \
        np.reshape(reconstruction_values[image_vector_indices[index]],
                   (block_width, block_height))

output_image_name = "CB_size=" + str(codebook_size) + ".png"
scipy.misc.imsave(output_image_name, image_after_compression)

print "Mean squared error = ", mean_squared_error(image_vectors, image_vectors_after_compression)
