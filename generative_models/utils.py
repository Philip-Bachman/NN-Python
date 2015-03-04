""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy as np
import PIL as PIL
# Stuff for visualizing diagnostics
from sklearn.neighbors import KernelDensity
import matplotlib as mpl
mpl.use('Agg')

class batch(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size

    def __call__(self,f):
        def wrapper(t,X):
            X = np.array(X)
            p = 0
            rem = 0
            results = []
            while p < len(X):
                Z = X[p:p+self.batch_size]
                if Z.shape[0] != self.batch_size:
                    zeros = np.zeros((self.batch_size-len(Z),X.shape[1]))
                    rem = len(Z)
                    Z = np.array(np.vstack((Z,zeros)),dtype=X.dtype)

                temp_results = f(t,Z)
                if rem != 0:
                    temp_results = temp_results[:rem]

                results.extend(temp_results)
                p += self.batch_size
            return np.array(results,dtype='float32')
        return wrapper

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape=None, tile_shape=None, tile_spacing=(0, 0),
                        scale=True,
                        output_pixel_vals=True,
                        colorImg=False):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).
    """
    X = X * 1.0 # converts ints to floats
    
    if colorImg:
        channelSize = X.shape[1]/3
        X = (X[:,0:channelSize], X[:,channelSize:2*channelSize], X[:,2*channelSize:3*channelSize], None)
    
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        
        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype='uint8' if output_pixel_vals else out_array.dtype
                        ) + channel_defaults[i]
                if i < 3:
                    print('WHY AM I HERE (utils.py line 101)?')
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                xi = X[i]
                if scale:
                    # shift and scale this channel to be in [0...1]
                    xi = (X[i] - X[i].min()) / (X[i].max() - X[i].min())
                out_array[:, :, i] = tile_raster_images(xi, img_shape, tile_shape, tile_spacing, False, output_pixel_vals) 
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)
        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    if scale:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        tmp = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                        this_img = scale_to_unit_interval(tmp)
                    else:
                        this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    out_array[
                        tile_row * (H+Hs): tile_row * (H + Hs) + H,
                        tile_col * (W+Ws): tile_col * (W + Ws) + W
                        ] \
                        = this_img * (255 if output_pixel_vals else 1)
        return out_array

def visualize(EN, proto_key, layer_num, file_name):
    W = EN.proto_nets[proto_key][layer_num].W.get_value(borrow=True).T
    size = int(np.sqrt(W.shape[1]))
    # hist(W.flatten(),bins=50)
    image = PIL.Image.fromarray(tile_raster_images(X=W, \
            img_shape=(size, size), tile_shape=(10,W.shape[0]/10),tile_spacing=(1, 1)))
    image.save(file_name)
    return

def visualize_net_layer(net_layer, file_name, colorImg=False, \
        use_transpose=False, transform=None):
    W = net_layer.W.get_value(borrow=False).T
    if use_transpose:
        W = net_layer.W.get_value(borrow=False)
    if not (transform is None):
        W = transform(W)
    if colorImg:
        size = int(np.sqrt(W.shape[1] / 3.0))
    else:
        size = int(np.sqrt(W.shape[1]))
    num_rows = 10
    num_cols = int((W.shape[0] / num_rows) + 0.999)
    img_shape = (size, size)
    tile_shape = (num_rows, num_cols)
    image = tile_raster_images(X=W, img_shape=img_shape, tile_shape=tile_shape, \
            tile_spacing=(1, 1), scale=True, colorImg=colorImg)
    image = PIL.Image.fromarray(image)
    image.save(file_name)
    return

def visualize_samples(X_samp, file_name, num_rows=10):
    d = int(np.sqrt(X_samp.shape[1]))
    # hist(W.flatten(),bins=50)
    image = PIL.Image.fromarray(tile_raster_images(X=X_samp, img_shape=(d, d), \
            tile_shape=(num_rows,X_samp.shape[0]/num_rows),tile_spacing=(1, 1)))
    image.save(file_name)
    return

# Matrix to image
def mat_to_img(X, file_name, img_shape, num_rows=10, \
        scale=True, colorImg=False, tile_spacing=(1,1)):
    num_rows = int(num_rows)
    num_cols = int((X.shape[0] / num_rows) + 0.999)
    tile_shape = (num_rows, num_cols)
    # make a tiled image from the given matrix's rows
    image = tile_raster_images(X=X, img_shape=img_shape, \
            tile_shape=tile_shape, tile_spacing=tile_spacing, \
            scale=scale, colorImg=colorImg)
    # convert to a standard image format and save to disk
    image = PIL.Image.fromarray(image)
    image.save(file_name)
    return

def plot_kde_histogram(X, f_name, bins=25):
    """
    Plot KDE-smoothed histogram of the data in X. Assume data is univariate.
    """
    import matplotlib.pyplot as plt
    X_samp = X.ravel()[:,np.newaxis]
    X_min = np.min(X_samp)
    X_max = np.max(X_samp)
    X_range = X_max - X_min
    sigma = X_range / float(bins)
    plot_min = X_min - (X_range/3.0)
    plot_max = X_max + (X_range/3.0)
    plot_X = np.linspace(plot_min, plot_max, 1000)[:,np.newaxis]
    # make a kernel density estimator for the data in X
    kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_samp)
    # make a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(plot_X, np.exp(kde.score_samples(plot_X)))
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def plot_kde_histogram2(X1, X2, f_name, bins=25):
    """
    Plot KDE-smoothed histogram of the data in X1/X2. Assume data is 1D.
    """
    import matplotlib.pyplot as plt
    # make a figure and configure an axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    for (X, style) in [(X1, '-'), (X2, '--')]:
        X_samp = X.ravel()[:,np.newaxis]
        X_min = np.min(X_samp)
        X_max = np.max(X_samp)
        X_range = X_max - X_min
        sigma = X_range / float(bins)
        plot_min = X_min - (X_range/3.0)
        plot_max = X_max + (X_range/3.0)
        plot_X = np.linspace(plot_min, plot_max, 1000)[:,np.newaxis]
        # make a kernel density estimator for the data in X
        kde = KernelDensity(kernel='gaussian', bandwidth=sigma).fit(X_samp)
        ax.plot(plot_X, np.exp(kde.score_samples(plot_X)), linestyle=style)
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def plot_stem(x, y, f_name):
    """
    Plot a stem plot.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def plot_line(x, y, f_name):
    """
    Plot a line plot.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return

def plot_scatter(x, y, f_name, x_label=None, y_label=None):
    """
    Plot a scatter plot.
    """
    import matplotlib.pyplot as plt
    if x_label is None:
        x_label = 'Posterior KLd'
    if y_label is None:
        y_label = 'Expected Log-likelihood'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0+(0.05*box.width), box.y0+(0.05*box.height), 0.96*box.width, 0.96*box.height])
    ax.set_xlabel(x_label, fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.hold(True)
    ax.scatter(x, y, s=24, alpha=0.5, c=u'b', marker=u'o')
    plt.sca(ax)
    x_locs, x_labels = plt.xticks()
    plt.xticks(x_locs, fontsize=18)
    y_locs, y_labels = plt.yticks()
    plt.yticks(y_locs, fontsize=18)
    fig.savefig(f_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format='png', \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    return



