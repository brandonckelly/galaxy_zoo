__author__ = 'brandonkelly'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from skimage.feature import peak_local_max
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import median_filter
import pandas as pd
from scipy import optimize
import datetime
from scipy import linalg
import multiprocessing

base_dir = os.environ['HOME'] + '/Projects/Kaggle/galaxy_zoo/'
data_dir = base_dir + 'data/'
training_dir = data_dir + 'images_training_rev1/'
test_dir = data_dir + 'images_test_rev1/'
plot_dir = base_dir + 'plots/'

doshow = True
file_dir = test_dir
do_parallel = True


# Define a function to make the ellipses
def ellipse(ra, rb, ang, x0, y0, Nb=100):
    xpos, ypos = x0, y0
    radm, radn = ra, rb
    an = ang
    co, si = np.cos(an), np.sin(an)
    the = np.linspace(0, 2 * np.pi, Nb)
    X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
    Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos
    return X, Y


def arctanh(x):
    z = 0.5 * np.log((1.0 + x) / (1.0 - x))
    return z


def tanh(z):
    x = (np.exp(2.0 * z) - 1.0) / (np.exp(2.0 * z) + 1.0)
    return x


def bivariate_gaussian(x, y, x_sigma, y_sigma, rho, xcent, ycent):
    xsqr = (x - xcent) ** 2 / x_sigma ** 2
    ysqr = (y - ycent) ** 2 / y_sigma ** 2
    xysqr = (x - xcent) * (y - ycent) / (x_sigma * y_sigma)
    bgauss = np.exp(-0.5 / (1.0 - rho ** 2) * (xsqr + ysqr - 2.0 * rho * xysqr))
    return bgauss


def sum_of_gaussians(xgrid, ygrid, params, xcent, ycent):
    """
    Sum of Gaussian functions model, with the centroids held fixed.

    :param xgrid, ygrid: The x and y values, npix ** 2 element arrays.
    :param params: The sequence of parameters for the K gaussian functions. Each Gaussian function will have parameters
        (log(amplitude), log(x_sigma), log(y_sigma), arctanh(rho)).
    :param xcent: The mean for the x-values.
    :param ycent: The mean for the y-values.
    """
    ngauss = len(xcent)
    model = 0.0
    for k in range(ngauss):
        amp = np.exp(params[4 * k])
        x_sigma = np.exp(params[4 * k + 1])
        y_sigma = np.exp(params[4 * k + 2])
        rho = tanh(params[4 * k + 3])
        model += amp * bivariate_gaussian(xgrid, ygrid, x_sigma, y_sigma, rho, xcent[k], ycent[k])

    return model


def sum_of_gaussians_error(params, image, xgrid, ygrid, xcent, ycent):
    error = sum_of_gaussians(xgrid, ygrid, params, xcent, ycent) - image
    return error


def extract_gal_image(file):

    source_id = file.split('/')[-1].split('.')[0]
    print source_id
    im = np.array(Image.open(file)).astype(float)
    ndim = im.shape

    error_messages = {'SourceID': source_id, 'ErrorFlag': 0}

    # use image from the middle (r?) band
    c = 1
    # find local maximum that is closest to center of image
    this_im = im[:, :, c]
    flux_sigma = 1.5 * np.median(np.abs(this_im - np.median(this_im)))  # MAD: robust estimate of sigma
    peak_threshold = np.median(this_im) + 8.0 * flux_sigma  # only look for 8-sigma peaks
    peak_threshold = min(peak_threshold, this_im.max() / 3)
    # apply median filter before finding local maxima
    filtered_im = median_filter(this_im, size=10)
    coords = peak_local_max(filtered_im, min_distance=20, threshold_abs=peak_threshold,
                            exclude_border=False)

    cdistance = np.sqrt((coords[:, 0] - ndim[0] / 2) ** 2 + (coords[:, 1] - ndim[1] / 2) ** 2)
    central_idx = cdistance.argmin()
    uvals, uidx = np.unique(filtered_im[coords[:, 0], coords[:, 1]], return_index=True)
    # make sure local max or duplicate closest to center is in uidx
    min_distance = 1e300
    for u in uidx:
        distance = np.sqrt((coords[u, 0] - coords[central_idx, 0]) ** 2 + (coords[u, 1] - coords[central_idx, 1]) ** 2)
        min_distance = min(distance, min_distance)
    # if minimum distance between coords[uidx, :] and coords[central_idx, :] < 5, local max closest to center of image
    # is not in the uidx, so add it
    if min_distance > 5:
        # make sure peak closest to center of image is included
        uidx = np.append(uidx, central_idx)

    coords = coords[uidx, :]  # only keep peaks with unique flux values
    distance = np.sqrt((coords[:, 0] - ndim[0] / 2) ** 2 + (coords[:, 1] - ndim[1] / 2) ** 2)
    d_idx = distance.argmin()
    # order the peaks by intensity, only keep top 5
    sort_idx = np.argsort(filtered_im[coords[:, 0], coords[:, 1]])[::-1]
    if d_idx in sort_idx[:5]:
        distance = distance[sort_idx[:5]]
        coords = coords[sort_idx[:5]]
    else:
        # make sure we keep the central galaxy
        sort_idx = sort_idx[:4]
        sort_idx = np.append(sort_idx, d_idx)
        distance = distance[sort_idx]
        coords = coords[sort_idx]

    # order the peaks by distance from the center of the image
    sort_idx = np.argsort(distance)
    distance = distance[sort_idx]
    coords = coords[sort_idx, :]

    if doshow:
        plt.imshow(this_im, cmap='hot')
        plt.plot([p[1] for p in coords], [p[0] for p in coords], 'bo')
        plt.plot(np.array([0, ndim[1]]), np.array([ndim[0]/2, ndim[0]/2]), 'g-')
        plt.plot(np.array([ndim[1]/2, ndim[1]/2]), np.array([0, ndim[0]]), 'g-')
        plt.xlim(0, ndim[1])
        plt.ylim(0, ndim[0])
        plt.show()
        # plt.imshow(filtered_im, cmap='hot')
        # plt.plot([p[1] for p in coords], [p[0] for p in coords], 'bo')
        # plt.xlim(0, ndim[1])
        # plt.ylim(0, ndim[0])
        # plt.show()

    # fit a mixture of gaussian functions model to the image, one gaussian for each local maximum
    nsources = len(distance)
    iparams = np.zeros(nsources * 4)
    xcentroids = np.zeros(nsources)
    ycentroids = np.zeros(nsources)
    for i in range(nsources):
        centroid = np.array([coords[i, 1], coords[i, 0]])
        xcentroids[i] = centroid[0]
        ycentroids[i] = centroid[1]
        xcent = np.arange(ndim[0]) - centroid[0]
        ycent = np.arange(ndim[1]) - centroid[1]
        Myy = np.sum(ycent ** 2 * this_im[:, coords[i, 1]]) / np.sum(this_im[:, coords[i, 1]])
        Mxx = np.sum(xcent ** 2 * this_im[coords[i, 0], :]) / np.sum(this_im[coords[i, 0], :])
        if i == 0:
            # main galaxy, so make initial guess of sigmas no larger than 1/4 length of image
            Mxx_min = (ndim[0] / 4.0) ** 2
            Myy_min = (ndim[1] / 4.0) ** 2
        else:
            # contaminants, make initial guess of sigmas no larger than 20 pixels
            Mxx_min = 20.0 ** 2
            Myy_min = 20.0 ** 2
        Mxx = min(Mxx, Mxx_min)
        Myy = min(Myy, Myy_min)

        amp = this_im[centroid[1], centroid[0]]

        iparams[4 * i] = np.log(amp)
        # iparams[4 * i + 1] = 0.5 * np.log(Mxx)
        # iparams[4 * i + 2] = 0.5 * np.log(Myy)
        iparams[4 * i + 1] = 0.5 * np.log(9.0)
        iparams[4 * i + 1] = 0.5 * np.log(9.0)

    # subtract off the base level of the image as the median of the values along the border
    border = np.hstack((this_im[:, 0], this_im[:, -1], this_im[0, 1:-1], this_im[-1, 1:-1]))
    base_flux = np.median(border)

    image_fit = this_im - base_flux
    rowgrid, colgrid = np.mgrid[:ndim[0], :ndim[1]]

    if ~np.all(np.isfinite(iparams)):
        # non-finite initial guess, skip this source
        print 'Non-finite initial guess at parameters detected for source', source_id, ', band', c
        error_messages['ErrorFlag'] = -99
        return error_messages


    params, success = optimize.leastsq(sum_of_gaussians_error, iparams,
                                       args=(image_fit.ravel(), colgrid.ravel(), rowgrid.ravel(),
                                             xcentroids, ycentroids), ftol=5e-2, xtol=5e-2)

    # if error, then save the info for analysis later
    error_messages['ErrorFlag'] = success

    if ~np.all(np.isfinite(params)):
        # don't crop image and save if non-finite parameters
        print 'Non-finite parameters detected for source', source_id, ', band', c
        return error_messages

    # get ellipse parameters for each Gaussian function
    rotang = np.zeros(nsources)
    amajor = np.zeros(nsources)
    aminor = np.zeros(nsources)
    gidx = 4 * np.arange(nsources)
    rho = tanh(params[gidx + 3])
    xsigma = np.exp(params[gidx + 1])
    ysigma = np.exp(params[gidx + 2])
    for k in range(nsources):
        radius = np.sqrt(xsigma[k] ** 2 + ysigma[k] ** 2)
        covar = np.zeros((2, 2))
        covar[0, 0] = xsigma[k] ** 2
        covar[1, 1] = ysigma[k] ** 2
        covar[0, 1] = rho[k] * xsigma[k] * ysigma[k]
        covar[1, 0] = covar[0, 1]

        vals, vecs = np.linalg.eigh(covar)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]

        rotang[k] = np.arctan2(*vecs[:, 0][::-1])
        amajor[k], aminor[k] = np.sqrt(vals)  # major and minor axes of ellipse

        this_centroid = np.array([xcentroids[k], ycentroids[k]])
        gal_centroid = np.array([xcentroids[0], ycentroids[0]])
        centdiff = this_centroid - gal_centroid
        gal_covar = np.zeros_like(covar)
        gal_covar[0, 0] = xsigma[0] ** 2
        gal_covar[1, 1] = ysigma[0] ** 2
        gal_covar[0, 1] = rho[0] * xsigma[0] * ysigma[0]
        gal_covar[1, 0] = covar[0, 1]

        mah_distance = np.sqrt(np.sum(centdiff * np.dot(linalg.inv(gal_covar), centdiff)))
        if k > 0 and radius < 20.0 and np.exp(params[4 * k] - params[0]) > 0.2 and mah_distance < 5.0:
            # subtract any nearby sources that look like stars and are 20% as bright as the central galaxy
            gauss_image = bivariate_gaussian(colgrid, rowgrid, xsigma[k], ysigma[k], rho[k], xcentroids[k],
                                             ycentroids[k])
            image_fit -= np.exp(params[4 * k]) * gauss_image

    # convert parameter output to a dictionary
    gauss_params = {'amplitude': np.exp(params[gidx]), 'xcent': xcentroids, 'ycent': ycentroids,
                    'xsigma': xsigma, 'ysigma': ysigma, 'rho': rho, 'theta': np.degrees(rotang),
                    'amajor': amajor, 'aminor': aminor}
    gauss_params = pd.DataFrame(gauss_params)  # convert to Pandas DataFrame
    gauss_params.index.name = 'GaussianID'

    # crop the image to 2.5-sigma
    arange = int(2.5 * np.abs(gauss_params['aminor'][0]))
    brange = int(2.5 * np.abs(gauss_params['amajor'][0]))

    for band in [1, 0, 2]:  # do middle band first since we want to use its asymmetry info
        this_im = im[:, :, band]
        # subtract off the base level of the image as the median of the values along the border
        border = np.hstack((this_im[:, 0], this_im[:, -1], this_im[0, 1:-1], this_im[-1, 1:-1]))
        base_flux = np.median(border)

        image_fit = this_im - base_flux

        # center the image over the center of the galaxy
        rcent = coords[0, 0]
        ccent = coords[0, 1]
        rrange = min(ndim[1] - rcent, rcent)
        crange = min(ndim[0] - ccent, ccent)
        rmin = rcent - rrange
        rmax = rcent + rrange
        cmin = ccent - crange
        cmax = ccent + crange

        image_fit = image_fit[rmin:rmax, cmin:cmax]

        if doshow:
            plt.imshow(image_fit, cmap='hot')
            plt.plot(np.array([0, image_fit.shape[1]]), np.array([image_fit.shape[0]/2, image_fit.shape[0]/2]), 'g-')
            plt.plot(np.array([image_fit.shape[1]/2, image_fit.shape[1]/2]), np.array([0, image_fit.shape[0]]), 'g-')
            plt.xlim(0, ndim[1])
            plt.ylim(0, ndim[0])
            plt.title('Centered Image')
            plt.show()

        # rotate the image so that semi-major axis is along the horizontal
        image_fit = rotate(image_fit, gauss_params['theta'][0], reshape=False)

        if doshow:
            rcent = image_fit.shape[0] / 2
            ccent = image_fit.shape[1] / 2
            plt.imshow(image_fit, cmap='hot')
            plt.plot(np.array([0, image_fit.shape[1]]), np.array([rcent, rcent]), 'g-')
            plt.plot(np.array([ccent, ccent]), np.array([0, image_fit.shape[0]]), 'g-')
            plt.xlim(0, image_fit.shape[1])
            plt.ylim(0, image_fit.shape[0])
            plt.title('Rotated Image')
            plt.show()

        # now crop the image to 2.5 sigma
        rcent = image_fit.shape[0] / 2
        ccent = image_fit.shape[1] / 2
        rrange = min(ndim[1] - rcent, rcent)
        crange = min(ndim[0] - ccent, ccent)
        rmin = rcent - arange
        rmin = max(rmin, 0)
        rmax = rcent + arange
        rmax = min(rmax, ndim[0])
        cmin = ccent - brange
        cmin = max(cmin, 0)
        cmax = ccent + brange
        cmax = min(cmax, ndim[1])
        cropped_im = image_fit[rmin:rmax, cmin:cmax].copy()

        if doshow:
            rcent = cropped_im.shape[0] / 2
            ccent = cropped_im.shape[1] / 2
            plt.imshow(cropped_im, cmap='hot')
            plt.plot(np.array([0, cropped_im.shape[1]]), np.array([rcent, rcent]), 'g-')
            plt.plot(np.array([ccent, ccent]), np.array([0, cropped_im.shape[0]]), 'g-')
            plt.xlim(0, cropped_im.shape[1])
            plt.ylim(0, cropped_im.shape[0])
            plt.title('Cropped Image')
            plt.show()

        # flip so asymmetry in middle band is always on the right
        if band == 1:
            column_collapse = cropped_im.mean(axis=0)
            column_collapse = column_collapse / np.sum(np.abs(column_collapse))
            col_asymmetry = np.sum(np.abs(column_collapse) * (np.arange(cropped_im.shape[1]) -
                                                              cropped_im.shape[1] / 2) ** 3)
        if np.sign(col_asymmetry) < 0:
            # image is asymmetric toward the left, flip it
            cropped_im = cropped_im[:, ::-1]
        # flip so asymmetry is always on the top
        if band == 1:
            row_collapse = cropped_im.mean(axis=1)
            row_collapse = row_collapse / np.sum(np.abs(row_collapse))
            row_asymmetry = np.sum(np.abs(row_collapse) * (np.arange(cropped_im.shape[0]) -
                                                           cropped_im.shape[0] / 2) ** 3)
        if np.sign(row_asymmetry) < 0:
            cropped_im = cropped_im[::-1, :]

        # save a plot of the image with the Gaussian ellipses and the cropped image
        plt.clf()
        plt.subplot(121)
        plt.imshow(this_im, cmap='hot')
        plt.title('Original')
        for k in range(nsources):
            # plot centroid of each ellipse
            plt.plot(xcentroids[k], ycentroids[k], 'bo')
            # plot 2-sigma region of each gaussian
            x_ell, y_ell = ellipse(2 * amajor[k], 2 * aminor[k], rotang[k], xcentroids[k], ycentroids[k])
            plt.plot(x_ell, y_ell, 'b')
        plt.xlim(0, ndim[0])
        plt.ylim(0, ndim[1])
        plt.subplot(122)
        plt.imshow(cropped_im, cmap='hot')
        plt.plot(cropped_im.shape[1] / 2, cropped_im.shape[0] / 2, 'b+')
        plt.xlim(0, cropped_im.shape[1])
        plt.ylim(0, cropped_im.shape[0])
        plt.title('Cropped')
        plt.savefig(plot_dir + source_id + '_' + str(band) + '.png')
        if doshow:
            plt.show()
        plt.close()
        # finally, save the cropped image as a numpy array
        np.save(training_dir + source_id + '_' + str(band), cropped_im)

        # save the mixture of gaussians model parameters
        gauss_params.to_csv(data_dir + 'gauss_fit/transfer/' + source_id + '_gauss_params.csv')

    return error_messages


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # warm up the pool
    pool.map(int, range(multiprocessing.cpu_count() - 1))

    # training_files = glob.glob(training_dir + '*.jpg')
    # training_files = training_files[55000:]

    # run on test set images
    files = glob.glob(file_dir + '*.jpg')
    files = files[50000:]
    # id_list = ['160788', '175306', '114125', '109698', '175870', '216293', '194473', '238866', '216338']
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
        err_msgs = pool.map(extract_gal_image, files)
    else:
        err_msgs = map(extract_gal_image, files)

    err_df = pd.DataFrame(err_msgs).set_index('SourceID')
    # if CSV file already exists, append to end of it
    if os.path.isfile(data_dir + 'gauss_fit/error_messages.csv'):
        err_old = pd.read_csv(data_dir + 'gauss_fit/error_messages.csv').set_index('SourceID')
        err_old = err_old.drop('Unnamed: 0', 1)
        err_df = pd.concat([err_old, err_df])
    # dump error messages to CSV file
    err_df.to_csv(data_dir + 'gauss_fit/error_messages.csv')

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(files), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'