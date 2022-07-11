""" Visualization utilities """
from pathlib import Path

import numpy as np
import k3d
from matplotlib import cm, colors
import trimesh


import imageio
import cv2
import matplotlib.pyplot as plt 
import h5py

import keras
import tensorflow as tf
from tensorflow.keras.layers import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical

from tensorflow.compat.v1.logging import INFO, set_verbosity

import tensorflow as tf
from IPython.display import Image
from keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose)
# Deconvolution3D vs Conv3DTranspose ?  



def visualize_occupancy(occupancy_grid, flip_axes=False):
    point_list = np.concatenate([c[:, np.newaxis] for c in np.where(occupancy_grid)], axis=1)
    visualize_pointcloud(point_list, 1, flip_axes=flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_cloud, point_size, colors=None, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def visualize_mesh(vertices, faces, flip_axes=False):
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        vertices[:, 2] = vertices[:, 2] * -1
        vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()


def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max() if np.sum(sdf < 0) > 0 else 1.
    sdf[sdf > 0] /= sdf[sdf > 0].max() if np.sum(sdf < 0) > 0 else 1.
    sdf /= -2.

    corners = np.array([
        [-.25, -.25, -.25],
        [.25, -.25, -.25],
        [-.25, .25, -.25],
        [.25, .25, -.25],
        [-.25, -.25, .25],
        [.25, -.25, .25],
        [-.25, .25, .25],
        [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)

    scale_factors = sdf[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertices = voxels.repeat(8, axis=0) + corners * scale_factors[:, np.newaxis]
    cube_vertex_colors = cm.get_cmap('seismic')(colors.Normalize(vmin=-1, vmax=1)(scale_factors))[:, :3]

    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5], [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0], [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[np.newaxis, :].repeat(12, axis=0).T.flatten()[:, np.newaxis]

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces, vertex_colors=cube_vertex_colors, process=False)
    mesh.export(str(filename))
    print(f"Exported to {filename}")


def visualize_shape_alignment(R=None, t=None):
    mesh_input = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_input.obj")
    mesh_target = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_target.obj")
    plot = k3d.plot(name='aligment', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    input_vertices = np.array(mesh_input.vertices)
    if not (R is None or t is None):
        t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, mesh_input.vertices.shape[0]))
        input_vertices = (R @ input_vertices.T + t_broadcast).T
    plt_mesh_0 = k3d.mesh(input_vertices.astype(np.float32), np.array(mesh_input.faces).astype(np.uint32), color=0xd00d0d)
    plt_mesh_1 = k3d.mesh(np.array(mesh_target.vertices).astype(np.float32), np.array(mesh_target.faces).astype(np.uint32), color=0x0dd00d)
    plot += plt_mesh_0
    plot += plt_mesh_1
    plt_mesh_0.shader = '3d'
    plt_mesh_1.shader = '3d'
    plot.display()




def plot_image_grid(image):
    data_all = []

    data_all.append(image)

    fig, ax = plt.subplots(3, 6, figsize=[16, 9])

    # coronal plane
    coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
    coronal = np.rot90(coronal, 1)

    # transversal plane
    transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
    transversal = np.rot90(transversal, 2)

    # sagittal plane
    sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
    sagittal = np.rot90(sagittal, 1)

    for i in range(6):
        n = np.random.randint(coronal.shape[2])
        ax[0][i].imshow(np.squeeze(coronal[:, :, n, :]))
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('Coronal', fontsize=15)

    for i in range(6):
        n = np.random.randint(transversal.shape[2])
        ax[1][i].imshow(np.squeeze(transversal[:, :, n, :]))
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('Transversal', fontsize=15)

    for i in range(6):
        n = np.random.randint(sagittal.shape[2])
        ax[2][i].imshow(np.squeeze(sagittal[:, :, n, :]))
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('Sagittal', fontsize=15)

    fig.subplots_adjust(wspace=0, hspace=0)


def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    # imageio.mimsave("/tmp/gif.gif", images, duration=0.01)
    # return Image(filename="/tmp/gif.gif", format='png')

    imageio.mimsave(r"C:\Users\ibrah\AppData\Local\Temp\gif.gif", images, duration=0.01)
    return Image(filename=r"C:\Users\ibrah\AppData\Local\Temp\gif.gif", format="png")

# Some code was borrowed from:
# https://github.com/ellisdg/3DUnetCNN/blob/master/unet3d/


def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1),
                             instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(
        input_layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False, depth=4, n_base_filters=32,
                  include_label_wise_dice_coefficients=False, metrics=[],
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-
    for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=
                                            current_layer.shape[1])(
            current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1].shape[1],
            input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1].shape[1],
            input_layer=current_layer,
            batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function,
                  metrics=metrics)
    return model


def visualize_patch(X, y):
    fig, ax = plt.subplots(1, 2, figsize=[10, 5], squeeze=False)

    ax[0][0].imshow(X[:, :, 0], cmap='Greys_r')
    ax[0][0].set_yticks([])
    ax[0][0].set_xticks([])
    ax[0][1].imshow(y[:, :, 0], cmap='Greys_r')
    ax[0][1].set_xticks([])
    ax[0][1].set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0)


class VolumeDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 sample_list,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim=(160, 160, 16),
                 num_channels=4,
                 num_classes=3,
                 verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sample_list) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.zeros((self.batch_size, self.num_channels, *self.dim),
                     dtype=np.float64)
        y = np.zeros((self.batch_size, self.num_classes, *self.dim),
                     dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("x"))
                # remove the background class
                y[i] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(sample_list_temp)

        return X, y


def get_labeled_image(image, label, is_categorical=False):
    if not is_categorical:
        label = to_categorical(label, num_classes=4).astype(np.uint8)

    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
        np.uint8)

    labeled_image = np.zeros_like(label[:, :, :, 1:])

    # remove tumor part from image
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

    # color labels
    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image


def predict_and_viz(image, label, model, threshold, loc=(100, 100, 50)):
    image_labeled = get_labeled_image(image.copy(), label.copy())

    model_label = np.zeros([3, 320, 320, 160])

    for x in range(0, image.shape[0], 160):
        for y in range(0, image.shape[1], 160):
            for z in range(0, image.shape[2], 16):
                patch = np.zeros([4, 160, 160, 16])
                p = np.moveaxis(image[x: x + 160, y: y + 160, z:z + 16], 3, 0)
                patch[:, 0:p.shape[1], 0:p.shape[2], 0:p.shape[3]] = p
                pred = model.predict(np.expand_dims(patch, 0))
                model_label[:, x:x + p.shape[1],
                y:y + p.shape[2],
                z: z + p.shape[3]] += pred[0][:, :p.shape[1], :p.shape[2],
                                      :p.shape[3]]

    model_label = np.moveaxis(model_label[:, 0:240, 0:240, 0:155], 0, 3)
    model_label_reformatted = np.zeros((240, 240, 155, 4))

    model_label_reformatted = to_categorical(label, num_classes=4).astype(
        np.uint8)

    model_label_reformatted[:, :, :, 1:4] = model_label

    model_labeled_image = get_labeled_image(image, model_label_reformatted,
                                            is_categorical=True)

    fig, ax = plt.subplots(2, 3, figsize=[10, 7])

    # plane values
    x, y, z = loc

    ax[0][0].imshow(np.rot90(image_labeled[x, :, :, :]))
    ax[0][0].set_ylabel('Ground Truth', fontsize=15)
    ax[0][0].set_xlabel('Sagital', fontsize=15)

    ax[0][1].imshow(np.rot90(image_labeled[:, y, :, :]))
    ax[0][1].set_xlabel('Coronal', fontsize=15)

    ax[0][2].imshow(np.squeeze(image_labeled[:, :, z, :]))
    ax[0][2].set_xlabel('Transversal', fontsize=15)

    ax[1][0].imshow(np.rot90(model_labeled_image[x, :, :, :]))
    ax[1][0].set_ylabel('Prediction', fontsize=15)

    ax[1][1].imshow(np.rot90(model_labeled_image[:, y, :, :]))
    ax[1][2].imshow(model_labeled_image[:, :, z, :])

    fig.subplots_adjust(wspace=0, hspace=.12)

    for i in range(2):
        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    return model_label_reformatted
