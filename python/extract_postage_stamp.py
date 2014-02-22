__author__ = 'brandonkelly'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
from skimage.feature import peak_local_max
from scipy.ndimage.interpolation import rotate
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

doshow = False


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

    df_list = []
    error_messages = {'SourceID': [], 'Band': [], 'ErrorFlag': []}

    # loop over images in each band
    for c in range(3):
        print '    ', c, '...'
        # find local maximum that is closest to center of image
        this_im = im[:, :, c]
        flux_sigma = 1.5 * np.median(np.abs(this_im - np.median(this_im)))  # MAD: robust estimate of sigma
        peak_threshold = np.median(this_im) + 8.0 * flux_sigma  # only look for 5-sigma peaks
        peak_threshold = min(peak_threshold, this_im.max() / 3)
        coords = peak_local_max(this_im, min_distance=20, threshold_abs=peak_threshold, exclude_border=False)

        uvals, uidx = np.unique(this_im[coords[:, 0], coords[:, 1]], return_index=True)
        coords = coords[uidx, :]  # only keep peaks with unique flux values
        distance = np.sqrt((coords[:, 1] - ndim[0] / 2) ** 2 + (coords[:, 0] - ndim[1] / 2) ** 2)
        d_idx = distance.argmin()
        # order the peaks by intensity, only keep top 5
        sort_idx = np.argsort(this_im[coords[:, 0], coords[:, 1]])[::-1]
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

        # plt.imshow(this_im, cmap='hot')
        # plt.plot([p[1] for p in coords], [p[0] for p in coords], 'bo')
        # plt.xlim(0, ndim[0])
        # plt.ylim(0, ndim[1])
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
            iparams[4 * i + 1] = 0.5 * np.log(Mxx)
            iparams[4 * i + 2] = 0.5 * np.log(Myy)

        # subtract off the base level of the image as the median of the values along the border
        border = np.hstack((this_im[:, 0], this_im[:, -1], this_im[0, 1:-1], this_im[-1, 1:-1]))
        base_flux = np.median(border)

        image_fit = this_im - base_flux
        rowgrid, colgrid = np.mgrid[:ndim[0], :ndim[1]]

        if ~np.all(np.isfinite(iparams)):
            # non-finite initial guess, skip this source
            print 'Non-finite initial guess at parameters detected for source', source_id, ', band', c
            error_messages['SourceID'].append(source_id)
            error_messages['Band'].append(c)
            error_messages['ErrorFlag'].append(-99)
            continue

        params, success = optimize.leastsq(sum_of_gaussians_error, iparams,
                                           args=(image_fit.ravel(), colgrid.ravel(), rowgrid.ravel(),
                                                 xcentroids, ycentroids), ftol=5e-2, xtol=5e-2)

        # if error, then save the info for analysis later
        error_messages['SourceID'].append(source_id)
        error_messages['Band'].append(c)
        error_messages['ErrorFlag'].append(success)

        if ~np.all(np.isfinite(params)):
            # don't crop image and save if non-finite parameters
            print 'Non-finite parameters detected for source', source_id, ', band', c
            continue

        # get ellipse parameters for each Gaussian function
        rotang = np.zeros(nsources)
        amajor = np.zeros(nsources)
        aminor = np.zeros(nsources)
        gidx = 4 * np.arange(nsources)
        rho = tanh(params[gidx + 3])
        xsigma = np.exp(params[gidx + 1])
        ysigma = np.exp(params[gidx + 2])
        for k in range(nsources):
            radius = np.sqrt(xsigma[k] **2 + ysigma[k] ** 2)
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
        df_list.append(gauss_params)

        # crop the image to 3-sigma and save it
        xrange = 4.0 * np.abs(gauss_params['amajor'][0])
        yrange = 4.0 * np.abs(gauss_params['aminor'][0])
        xmin = int(coords[0, 1] - xrange)
        if xmin < 0:
            xmin = 0
        xmax = int(coords[0, 1] + xrange)
        if xmax > ndim[0]:
            xmax = ndim[0]
        ymin = int(coords[0, 0] - yrange)
        if ymin < 0:
            ymin = 0
        ymax = int(coords[0, 0] + yrange)
        if ymax > ndim[1]:
            ymax = ndim[1]

        # rotate the image so that semi-major axis is along the horizontal
        image_fit = rotate(image_fit, gauss_params['theta'][0], reshape=False)
        cropped_im = image_fit[ymin:ymax, xmin:xmax]  # crop the image, remember arrays are row-major

        # make the floor value zero
        border = np.hstack((cropped_im[:, 0], cropped_im[:, -1], cropped_im[0, 1:-1], cropped_im[-1, 1:-1]))
        cropped_im -= np.median(border)

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
        plt.title('Cropped')
        plt.savefig(plot_dir + source_id + '_' + str(c) + '.png')
        if doshow:
            plt.show()
        plt.close()
        # finally, save the cropped image as a numpy array
        np.save(training_dir + source_id + '_' + str(c), cropped_im)

    # save the mixture of gaussians model parameters
    dataframe = pd.concat(df_list, keys=['1', '2', '3'])
    dataframe.index.names = ('Band', 'GaussianID')
    dataframe.to_csv(data_dir + 'gauss_fit/' + source_id + '_gauss_params.csv')

    err_df = pd.DataFrame(error_messages).set_index('Band')

    return err_df


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    # warm up the pool
    pool.map(int, range(multiprocessing.cpu_count() - 1))

    training_files = glob.glob(training_dir + '*.jpg')
    training_files = training_files[40000:45000]
    # training_files = [training_dir + '685386.jpg']

    print len(training_files), 'galaxies...'
    print 'Source ID...'

    do_parallel = True
    if do_parallel:
        err_dfs = pool.map(extract_gal_image, training_files)
    else:
        err_dfs = map(extract_gal_image, training_files)

    source_ids = []
    for file in training_files:
        source_ids.append(file.split('/')[-1].split('.')[0])

    err_df = pd.concat(err_dfs, keys=source_ids)
    err_df = err_df.drop('SourceID', 1)
    err_df.index.names = ('SourceID', 'Band')
    # dump error messages to CSV file
    err_df.to_csv(data_dir + 'gauss_fit/error_messages.csv')

    end_time = datetime.datetime.now()

    tdiff = end_time - start_time
    tdiff = tdiff.total_seconds()
    print 'Did', len(training_files), 'galaxies in', tdiff / 60.0 / 60.0, 'hours.'