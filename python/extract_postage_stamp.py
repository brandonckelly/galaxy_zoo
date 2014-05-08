__author__ = 'brandonkelly'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os
import glob
from skimage.measure import find_contours
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import median_filter
import pandas as pd
from scipy import optimize
import datetime
import multiprocessing

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

doshow = False
do_parallel = True
debug = False


def extract_galaxy(image, flux_sigma, gcontour=None, zero_outside=True):
    if gcontour is None:
        # find the contour defining the boundary of the galaxy
        cpoint = np.array((image.shape[1], image.shape[0])) / 2
        floor = np.median(image)
        threshold = np.max([2.0 * flux_sigma, floor + 0.075 * (image[cpoint[0], cpoint[1]] - floor)])
        contour = find_contours(image, threshold)
        gcount = 0
        for c in contour:
            # find the contour that contains the central pixel: points interior to this belong to the galaxy
            this_path = Path(c, closed=True)
            if this_path.contains_point(cpoint):
                gcontour = this_path
                gcount += 1

    if debug:
        plt.clf()
        plt.imshow(image, cmap='hot')
        plt.plot(gcontour.vertices[:, 1], gcontour.vertices[:, 0], 'b')
        plt.show()

    if gcontour is None:
        # could not find the galaxy
        return None, gcontour, None
    else:
        # return the flux along the border of the galaxy, since we will use this to add noise later
        border = image[gcontour.vertices[:, 0].astype(int), gcontour.vertices[:, 1].astype(int)]
        # crop the image
        # note image is indexed (row, column) = (y, x), so gcontour.vertices[:, 0] = set of row values for the contour
        rmin, rmax = np.floor(gcontour.vertices[:, 0].min()), np.ceil(gcontour.vertices[:, 0].max())
        cmin, cmax = np.floor(gcontour.vertices[:, 1].min()), np.ceil(gcontour.vertices[:, 1].max())
        cropped = image[int(rmin):int(rmax), int(cmin):int(cmax)]
        cropped_contour = gcontour.deepcopy()
        cropped_contour.vertices[:, 0] -= rmin
        cropped_contour.vertices[:, 1] -= cmin

        if debug:
            plt.clf()
            plt.imshow(cropped, cmap='hot')
            plt.plot(cropped_contour.vertices[:, 1], cropped_contour.vertices[:, 0], 'b')
            plt.show()

        if zero_outside:
            # now find the pixels outside of the contour and set them to zero
            y, x = np.mgrid[:cropped.shape[0], :cropped.shape[1]]
            pixels = np.column_stack((y.ravel(), x.ravel()))
            galaxy_pixels = cropped_contour.contains_points(pixels)
            outside = np.where(~galaxy_pixels)[0]
            outside = np.unravel_index(outside, cropped.shape)
            cropped[outside] = 0.0

        return cropped, gcontour, border


def bivariate_gaussian(x, y, x_sigma, y_sigma, rho, xcent, ycent):
    xsqr = (x - xcent) ** 2 / x_sigma ** 2
    ysqr = (y - ycent) ** 2 / y_sigma ** 2
    xysqr = (x - xcent) * (y - ycent) / (x_sigma * y_sigma)
    bgauss = np.exp(-0.5 / (1.0 - rho ** 2) * (xsqr + ysqr - 2.0 * rho * xysqr))
    return bgauss


def sum_of_gaussians(xgrid, ygrid, params):
    """
    Gaussian function model.

    :param xgrid, ygrid: The x and y values, npix ** 2 element arrays.
    :param params: The sequence of parameters for the K gaussian functions. Each Gaussian function will have parameters
        (log(amplitude), log(x_sigma), log(y_sigma), arctanh(rho)).
    :param xcent: The mean for the x-values.
    :param ycent: The mean for the y-values.
    """
    amp = np.exp(params[0])
    x_sigma = np.exp(params[1])
    y_sigma = np.exp(params[2])
    rho = np.tanh(params[3])
    xcent = params[4]
    ycent = params[5]
    model = amp * bivariate_gaussian(xgrid, ygrid, x_sigma, y_sigma, rho, xcent, ycent)

    return model


def sum_of_gaussians_error(params, image, xgrid, ygrid):
    error = sum_of_gaussians(xgrid, ygrid, params) - image
    return error


def get_rotation_angle(image):
    # fit a gaussian function model to the image to get the rotation angle
    iparams = np.zeros(6)
    xcentroid = image.shape[1] / 2.0
    ycentroid = image.shape[0] / 2.0
    # make initial guess of sigmas no larger than 1/4 length of image
    Mxx = (image.shape[1] / 4.0) ** 2
    Myy = (image.shape[0] / 4.0) ** 2

    amp = image[ycentroid, xcentroid]

    iparams[0] = np.log(amp)
    iparams[1] = 0.5 * np.log(Mxx)
    iparams[2] = 0.5 * np.log(Myy)
    iparams[4] = xcentroid
    iparams[5] = ycentroid

    # subtract off the base level of the image as the median of the values along the border
    base_flux = image[image > 0].min()
    image_fit = image - base_flux
    rowgrid, colgrid = np.mgrid[:image.shape[0], :image.shape[1]]

    if not np.all(np.isfinite(iparams)):
        # non-finite initial guess, skip this source
        return None

    params, success = optimize.leastsq(sum_of_gaussians_error, iparams,
                                       args=(image_fit.ravel(), colgrid.ravel(), rowgrid.ravel()),
                                       ftol=1e-2, xtol=1e-2)

    # now get rotation angle from eigendecomposition of best-fit covariance matrix
    rho = np.tanh(params[3])
    xsigma = np.exp(params[1])
    ysigma = np.exp(params[2])
    covar = np.zeros((2, 2))
    covar[0, 0] = xsigma ** 2
    covar[1, 1] = ysigma ** 2
    covar[0, 1] = rho * xsigma * ysigma
    covar[1, 0] = covar[0, 1]

    vals, vecs = np.linalg.eigh(covar)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    rotang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    if not np.isfinite(rotang):
        rotang = None

    return rotang


def center_galaxy(image, original_image, centroid=None):
    if centroid is None:
        # apply median filter to find the galaxy centroid
        centroid = median_filter(image, size=10).argmax()
        centroid = np.unravel_index(centroid, image.shape)
    # recenter image
    roffset = centroid[0] - image.shape[0] / 2
    if roffset < 0:
        # add more white space to top of image
        extra_rows = image.shape[0] - 2 * centroid[0]
        image = np.vstack((np.zeros((extra_rows, image.shape[1])), image))
    elif roffset > 0:
        # add more white space to bottom of image
        extra_rows = 2 * centroid[0] - image.shape[0]
        image = np.vstack((image, np.zeros((extra_rows, image.shape[1]))))
    coffset = centroid[1] - image.shape[1] / 2
    if coffset > 0:
        # add more white space to right of image
        extra_columns = 2 * centroid[1] - image.shape[1]
        image = np.column_stack((image, np.zeros((image.shape[0], extra_columns))))
    elif coffset < 0:
        # add more white space to left of image
        extra_columns = image.shape[1] - 2 * centroid[1]
        image = np.column_stack((np.zeros((image.shape[0], extra_columns)), image))

    return image, centroid


def crop_image(image, shape):
    nrows, ncols = image.shape
    rdiff = nrows - shape[0]

    if rdiff < 0:
        # need to add rows
        extra_rows = np.abs(rdiff) / 2
        # first add to top of image
        if extra_rows != 0:
            image = np.vstack((np.zeros((extra_rows, ncols)), image))
        # add remaining rows to bottom of image
        extra_rows = shape[0] - image.shape[0]
        if extra_rows != 0:
            image = np.vstack((image, np.zeros(extra_rows, ncols)))
    elif rdiff > 0:
        # need to remove rows
        nremove = rdiff / 2
        if nremove != 0:
            image = image[nremove:, :]
        nremove = image.shape[0] - shape[0]
        if nremove != 0:
            image = image[:-nremove, :]
    nrows = image.shape[0]

    cdiff = ncols - shape[1]
    if cdiff < 0:
        # need to add columns
        extra_cols = np.abs(cdiff) / 2
        # first add to left of image
        if extra_rows != 0:
            image = np.column_stack((np.zeros((extra_cols, nrows)), image))
        # add remaining columns to right of image
        extra_cols = shape[1] - image.shape[1]
        if extra_cols != 0:
            image = np.vstack((image, np.zeros(extra_cols, nrows)))
    elif cdiff > 0:
        # need to remove columns
        nremove = cdiff / 2
        if nremove != 0:
            image = image[:, nremove:]
        nremove = image.shape[1] - shape[1]
        if nremove != 0:
            image = image[:, -nremove]

    return image


def extract_gal_image(file):
    source_id = file.split('/')[-1].split('.')[0]
    image_dir = os.path.dirname(file) + '/'
    print source_id
    im = np.array(Image.open(file)).astype(float)

    # use image from the middle (r?) band as size and rotation reference, so that's why we do it first
    centroid = None
    rotang = None
    gcontour = None
    gcontour2 = None
    for c in [1, 0, 2]:
        this_im = im[:, :, c]
        flux_sigma = 1.5 * np.median(np.abs(this_im - np.median(this_im)))  # MAD: robust estimate of sigma
        # first extract galaxy pixels
        cropped, gcontour, border = extract_galaxy(this_im.copy(), flux_sigma, gcontour)

        # check output
        if cropped is None:
            break

        if debug:
            plt.clf()
            plt.imshow(cropped, cmap='hot')
            plt.show()

        if c == 1:
            # get the rotation angle, using the c == 1 band as the reference
            rotang = get_rotation_angle(cropped)
            # check output
            if rotang is None:
                break

        # rotate the image using the orientation for image[c == 1] so that major axis is along horizontal
        cropped = rotate(this_im, rotang, reshape=False)

        # need to crop image again after rotation
        cropped, gcontour2, border2 = extract_galaxy(cropped.copy(), flux_sigma, gcontour2, zero_outside=False)

        # check output
        if cropped is None:
            break

        if debug:
            plt.clf()
            plt.imshow(cropped, cmap='hot')
            plt.title('Cropped, After Rotation')
            plt.show()

        # center the galaxy image about the peak in r-band brightness
        cropped, centroid = center_galaxy(cropped, this_im, centroid)

        # set the pixels with zero flux to noise by randomly sampling from the border of the galaxy contour
        zero_flux = (cropped == 0)
        idx = np.random.random_integers(0, len(border2)-1, np.sum(zero_flux))
        cropped[zero_flux] = border2[idx]

        if debug:
            plt.clf()
            plt.imshow(cropped, cmap='hot')
            plt.title('Centered')
            plt.show()


        # scale the image to have zero flux on average along the border
        background = np.median(np.hstack((cropped[:, 0], cropped[:, -1], cropped[0, 1:-1], cropped[-1, 1:-1])))
        cropped -= np.median(background)

        # plt.hist(cropped.ravel(), bins=100)
        # plt.show()
        # plt.plot(cropped[:, cropped.shape[1]/2])
        # plt.plot(cropped[cropped.shape[0]/2, :])
        # plt.show()

        # flip so asymmetry in middle band is always on the right
        if c == 1:
            column_collapse = cropped.mean(axis=0)
            column_collapse = column_collapse / np.sum(np.abs(column_collapse))
            col_asymmetry = np.sum(np.abs(column_collapse) * (np.arange(cropped.shape[1]) -
                                                              cropped.shape[1] / 2) ** 3)
        if np.sign(col_asymmetry) < 0:
            # image is asymmetric toward the left, flip it
            cropped = cropped[:, ::-1]

        # flip so asymmetry is always on the top
        if c == 1:
            row_collapse = cropped.mean(axis=1)
            row_collapse = row_collapse / np.sum(np.abs(row_collapse))
            row_asymmetry = np.sum(np.abs(row_collapse) * (np.arange(cropped.shape[0]) -
                                                           cropped.shape[0] / 2) ** 3)
        if np.sign(row_asymmetry) < 0:
            cropped = cropped[::-1, :]

        # save a plot comparing the original image to the extracted image
        plt.clf()
        plt.subplot(121)
        plt.imshow(this_im, cmap='hot')
        plt.title('Original')
        plt.subplot(122)
        rcent = cropped.shape[0] / 2
        ccent = cropped.shape[1] / 2
        plt.imshow(cropped, cmap='hot')
        plt.plot(np.array([0, cropped.shape[1]]), np.array([rcent, rcent]), 'g-')
        plt.plot(np.array([ccent, ccent]), np.array([0, cropped.shape[0]]), 'g-')
        cent = np.unravel_index(cropped.argmax(), cropped.shape)
        plt.plot(cent[1], cent[0], 'bo')
        plt.xlim(0, cropped.shape[1])
        plt.ylim(0, cropped.shape[0])
        plt.title('Extracted Galaxy Image')
        plt.savefig(plot_dir + source_id + '_' + str(c) + '.png')
        if doshow:
            plt.show()
        plt.close()

        # finally, save the extracted postage stampimage as a numpy array
        print 'Saving file to', image_dir + source_id + '_' + str(c) + '.npy'
        np.save(image_dir + source_id + '_' + str(c), cropped)

    return source_id


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    if do_parallel:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
        # warm up the pool
        pool.map(int, range(multiprocessing.cpu_count() - 1))

    file_dir = training_dir
    # file_dir = test_dir

    files = glob.glob(file_dir + '*.jpg')
    files = files[:1000]
    # id_list = ['100380']
    # files = [file_dir + id + '.jpg' for id in id_list]

    # find which ones we've already done
    already_done1 = glob.glob(plot_dir + '*_0.png')
    already_done2 = glob.glob(plot_dir + '*_1.png')
    already_done3 = glob.glob(plot_dir + '*_2.png')

    already_done1 = set([s.split('/')[-1].split('_')[0] for s in already_done1])
    already_done2 = set([s.split('/')[-1].split('_')[0] for s in already_done2])
    already_done3 = set([s.split('/')[-1].split('_')[0] for s in already_done3])

    already_done = already_done1 & already_done2 & already_done3
    print 'Already done', len(already_done), 'galaxies.'
    all_sources = set([tfile.split('/')[-1].split('.')[0] for tfile in files])

    left_to_do = all_sources - already_done

    print 'Have', len(left_to_do), 'galaxies left.'
    # training_files = [training_dir + sID + '.jpg' for sID in left_to_do]
    files = [file_dir + sID + '.jpg' for sID in left_to_do]

    print len(files), 'galaxies...'
    assert len(files) == len(left_to_do)
    print 'Source ID...'

    if do_parallel:
        gal_ids = pool.map(extract_gal_image, files)
    else:
        gal_ids = map(extract_gal_image, files)

    end_time = datetime.datetime.now()
    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(files), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'