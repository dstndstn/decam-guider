import sys
import os
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.util import Sip

#sys.path.insert(0, 'legacypipe/py')
from legacypipe.ps1cat import ps1cat

# Obsbot isn't a proper module
sys.path.insert(0, 'obsbot')
from measure_raw import RawMeasurer
from camera_decam import nominal_cal

import scipy.optimize
from legacypipe.ps1cat import ps1_to_decam

def assemble_full_frames(fn):
    F = fitsio.FITS(fn, 'r')
    phdr = F[0].read_header()
    chipnames = []
    imgs = []
    biases = []
    # 4 guide chips
    for i in range(4):
        #DATASEC = '[7:1030,1:2048]'    / Data section to display
        #DETPOS  = 'GS1     '
        # two amps per chip
        ampimgs = []
        thisbias = []
        for j in range(2):
            hdu = i*2 + j + 1
            img = F[hdu].read()
            #print('Full amp image shape:', img.shape)
            hdr = F[hdu].read_header()
            datasec = hdr['DATASEC'].strip('[]').split(',')
            assert(len(datasec) == 2)
            #print(datasec)
            #v = [w.split(':') for w in datasec]
            (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec]]
            #print(v)
            #(x0,x1),(y0,y1) = [int(x) for x in [w.split(':') for w in datasec]]
            ampimg = img[y0-1:y1, x0-1:x1]
            #print('DATASEC shape:', ampimg.shape)

            biassec = hdr['BIASSEC'].strip('[]').split(',')
            assert(len(biassec) == 2)
            (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in biassec]]
            biasimg = img[y0-1:y1, x0-1:x1]
            #print('BIASSEC shape:', biasimg.shape)
            #print('BIAS median:', np.median(biasimg))

            # First ~50 rows are bad (very bright)
            biasimg = biasimg[48:, :]

            # Sort the resulting 2000 x 50 array so that the 50 columns are sorted for each row
            s = np.sort(biasimg, axis=1)
            # Take the mean of the middle 10 pixels, for each row
            m = np.mean(s[:, 20:30], axis=1)
            m = np.median(m)
            ampimg = ampimg.astype(np.float32)
            ampimg -= m
            thisbias.append(m)
            #plt.imshow(biasimg)
            #m = np.median(biasimg)
            #plt.plot(np.median(biasimg, axis=1),'-')
            #plt.plot(m, '-')
            #plt.ylim(m-20, m+20)
            #plt.axvline(50)
            #plt.show()
            
            ampimgs.append(ampimg)
        biases.append(thisbias)            
        chipnames.append(hdr['DETPOS'])
        imgs.append(np.hstack(ampimgs))#.astype(np.float32))
        #print(imgs[-1].shape)
    return chipnames, imgs, phdr, biases

class DECamGuiderMeasurer(RawMeasurer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_trim = 50
        # HACK
        self.airmass = 1.0
        # HACK
        self.gain = 2.68

    def read_raw(self, F, ext):
        img,hdr = super().read_raw(F, ext)
        img *= self.gain
        return img,hdr

    def get_exptime(self, primhdr):
        return primhdr['GEXPTIME']

    def get_airmass(self, primhdr):
        return self.airmass

    def filter_detected_sources(self, detsn, x, y, ps):
        h,w = detsn.shape
        ix = np.clip(np.round(x), 0, w-1).astype(int)
        iy = np.clip(np.round(y), 0, h-1).astype(int)
        # Remove trails -- remove stars that have a brighter star within dx columns
        # and larger y.
        I = np.argsort(-detsn[iy, ix])
        keep = np.ones(len(x), bool)
        dx = 3
        for j,i in enumerate(I):
            Ibrighter = I[:j]
            j = np.flatnonzero((np.abs(x[i] - x[Ibrighter]) < dx) * (y[Ibrighter] > y[i]))
            if len(j):
                keep[i] = False
        return keep

    def get_sky_and_sigma(self, img):
        h,w = img.shape

        # Remove a V-shape pattern (MAGIC number)
        guider_horiz_slope = 0.00188866
        xx = np.arange(w)
        hbias = np.abs(xx - (w//2 - 0.5)) * guider_horiz_slope
        skymod = np.zeros(img.shape, np.float32)
        skymod += hbias[np.newaxis,:]

        # Measure and remove a sky background ramp
        ramp = np.median(img - skymod, axis=1)

        # Fit a line (L1 norm)
        def obj(p, x, y):
            b,m = p
            ymod = b + m*x
            return np.sum(np.abs(ymod - y))
        x = np.arange(len(ramp))
        r = scipy.optimize.minimize(obj, (0., 1.), (x, ramp))
        b,m = r.x

        # MAGIC number:
        # Guider readout times: 160 us to shift a row.  4 us per pixel.  1024 pix + overscan = 1080
        # = 4480 us per row
        row_exptime = 4480e-6
        fake_exptime = self.get_exptime(self.primhdr)
        sky1 = fake_exptime * m / row_exptime

        # Remove sky ramp
        skymod += (m * x)[:, np.newaxis]

        # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        #plt.clf()
        #plt.hist(np.abs((img - skymod)[slice1] - (img - skymod)[slice2]).ravel())
        #plt.title('pixel diffs')
        #plt.show()
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)

        if False:
            plt.clf()
            plt.subplot(1,2,1)
            mn,mx = np.percentile(img.ravel(), [5,95])
            plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.subplot(1,2,2)
            mn,mx = np.percentile((img-skymod).ravel(), [5,95])
            plt.imshow(img-skymod, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.show()

        #plt.clf()
        ##plt.plot(np.median(img, axis=0), label='ax 0')
        #plt.plot(np.median(img, axis=1), label='ax 1')
        #plt.plot(x, b + m*x, '-')        
        #plt.legend()
        #plt.show()
        return skymod, sky1, sig1

    def get_wcs(self, hdr):
        return self.wcs

    def get_ps1_band(self, band):
        print('Band', band)
        return ps1cat.ps1band[band]

    def colorterm_ps1_to_observed(self, ps1stars, band):
        return ps1_to_decam(ps1stars, band)


def process_acq_image(filename, astrometry_config_file, radec_boresight=None,
                      airmass=None,
                      debug=False):
    '''
    Arguments:
    * filename: string, filename of guider acquisition (first exposure) FITS file
    * astrometry_config_file: string, filename, Astrometry.net config file
    * radec_boresight = (ra, dec) where ra,dec are floats, in decimal degrees.
    '''
    print('Reading', filename)
    chipnames,imgs,phdr,biases = assemble_full_frames(filename)
    expnum = int(phdr['EXPNUM'])
    filt = phdr['FILTER']
    print('Expnum', expnum, 'Filter', filt)

    procdir = 'data-processed'
    if not os.path.exists(procdir):
        try:
            os.makedirs(procdir)
        except:
            pass

    if debug:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(os.path.join(procdir, 'plot'))
        plt.clf()

    wcsfns = {}
    imgfns = {}
    goodchips = []
    any_img = False
    for i,(chip,img,biaslr) in enumerate(zip(chipnames, imgs, biases)):
        imgfn = os.path.join(procdir, '%s-acq-%s.fits' % (expnum, chip))
        imgfns[chip] = imgfn
        # HACK - speed up re-runs
        wcsfn = os.path.join(procdir, '%s-acq-%s.wcs' % (expnum, chip))
        wcsfns[chip] = wcsfn
        if os.path.exists(wcsfn):
            goodchips.append(chip)
            continue
        axyfn = os.path.join(procdir, '%s-acq-%s.axy' % (expnum, chip))
        if os.path.exists(axyfn):
            print('Exists:', axyfn, '-- assuming it will not solve')
            continue

        # Save images for each guider chip -- build header
        hdr = fitsio.FITSHDR()
        for k in ['UTSHUT', 'GEXPTIME', 'FILTER', 'EXPNUM']:
            hdr[k] = phdr[k]
        hdr['CHIPNAME'] = chip
        l,r = biaslr
        hdr['BIAS_L'] = l
        hdr['BIAS_R'] = r        
        fitsio.write(imgfn, img, header=hdr, clobber=True)
        print('Wrote', imgfn)
        imgfns[chip] = imgfn

        cmd = ('solve-field ' +
               '--config %s ' % astrometry_config_file +
               '--scale-low 0.25 --scale-high 0.27 --scale-units app ' +
               '--solved none --match none --corr none --new-fits none ' +
               '--no-tweak ' +
               '--depth 30 ' +
               '--continue --nsigma 6 ')

        if debug:
            cmd = cmd + '--plot-scale 0.5 '
        else:
            cmd = cmd + '--no-plots '

        if radec_boresight is not None:
            cmd = cmd + '--ra %.4f --dec %.4f --radius 5 ' % radec_boresight

        cmd = cmd + imgfn
        #cmd = cmd + ' -v --no-delete-temp'
        print(cmd)
        rtn = os.system(cmd)
        print('rtn:', rtn)
        if rtn:
            continue

        if debug:
            any_img = True
            plt.subplot(2,2, i+1)
            plt.imshow(img, origin='lower', interpolation='nearest', vmin=-30, vmax=+50)
            plt.xticks([]); plt.yticks([])
            plt.title(chip)

        if os.path.exists(wcsfn):
            goodchips.append(chip)

    if debug and any_img:
        plt.suptitle(filename)
        ps.savefig()

    chipmeas = {}
    for chip in goodchips:
        print()
        print('Measuring', chip)
        wcsfn = wcsfns[chip]
        imgfn = imgfns[chip]
        wcs = Sip(wcsfn)
        p = ps1cat(ccdwcs=wcs)
        stars = p.get_stars()

        ext = 0
        meas = DECamGuiderMeasurer(imgfn, ext, nominal_cal)
        if airmass is not None:
            meas.airmass = airmass
        meas.wcs = wcs
        # max astrometric shift, in arcsec
        meas.maxshift = 5.

        # Star detection threshold
        meas.det_thresh = 6.

        kw = {}
        if debug:
            #kw.update(ps=ps)
            pass

        R = meas.run(**kw)
        chipmeas[chip] = (meas, R)
        # 'band': 'M411'
        # 'airmass': 1.0
        # 'exptime': 30.0
        # 'skybright': 23.42693784008395
        # 'rawsky': 5.5481564790558835
        # 'pixscale': 0.262,
        # 'primhdr': ...
        # 'wcs': SIP obj
        # 'ra_ccd': 149.90534729304895, 
        # 'dec_ccd': 1.8049787089344023,
        # 'extension': 0, 
        # 'camera': '', 
        # 'ndetected': 13,
        # 'affine': [974.0, 974.0, 0.1914130766322191, 1.0000301152024258, -1.8359413769375443e-05, 0.07854012216121929, -3.324615770705641e-05, 1.0001196245634363],
        # 'px': array([ 256.38712122, ...])   Pan-STARRS pixel x coords
        # 'py': array([1416.97103258, ...])   
        # 'dx': -1.0563087159905251,
        # 'dy': -0.08010233474379902,
        # 'nmatched': 8,
        # 'x': array([1346.80603865,  712.99193725, 1871.18362218,  807.88778635, 451.31905514,   56.22935702,  151.79373109, 1803.74869803]),
        # 'y': array([ 118.74601323,  356.778718  ,  559.01479522,  713.95439703, 854.79866113, 1191.84354098, 1418.8836714 , 1524.29286497]),
        # 'apflux': <Column name='aperture_sum' dtype='float64' length=8> 
        # 'apmag': <Column name='aperture_sum' dtype='float64' length=8>
        # 'colorterm': array([0.24142946, 0.30522041, 0.24119551, 0.42094694, 0.33545435, 0.41233086, 0.26307692, 0.52023876]),
        # (for the 8 matched stars)
        # 'refmag': array([15.924142, 16.431793, 14.727116, 14.141509, 16.91051 , 16.661407, 16.563139, 16.419907], dtype=float32),
        # 'refstars': <tabledata object with 8 rows and 19 columns: obj_id, ra, dec, nmag_ok, stdev, mean, median, mean_ap, median_ap, posstdev,
        #                       epochmean, ra_ok, dec_ok, epochmean_ok, posstdev_ok, x, y, mag, gicolor>
        # 'zp': 19.14134165823857,
        # 'transparency': 0.007166987768690248,
        # 'zp_mean': 19.165709331700608, 
        # 'zp_skysub': 19.15412810056008,
        # 'zp_med': 19.105666763760993,
        # 'zp_med_skysub': 19.14134165823857,
        # 'seeing': 1.5064726327517675
        
        #got = True
        #break
    #if got:

    zp0 = None
    kx = None

    dmags = []
    for chip in goodchips:
        meas,R = chipmeas[chip]
        ref = R['refstars']
        apflux = R['apflux']
        exptime = R['exptime']
        apmag = -2.5 * np.log10(apflux / exptime)
        dmags.append(apmag - ref.mag)
        if zp0 is None:
            zp0 = meas.zeropoint_for_exposure(filt, ext=meas.ext, exptime=exptime,
                                              primhdr=R['primhdr'])
            kx = nominal_cal.fiducial_exptime(filt).k_co
    dmags = np.hstack(dmags)
    zpt = -np.median(dmags)
    print()
    print('All chips:')
    print('Zeropoint:    %.3f   (with %i stars)' % (zpt, len(dmags)))
    del dmags

    transparency = 10.**(-0.4 * (zp0 - zpt - kx * (airmass - 1.)))
    print('Transparency:  %.3f' % transparency)

    if debug:
        plt.clf()
        diffs = []
        for chip in goodchips:
            meas,R = chipmeas[chip]
            ref = R['refstars']
            for i,b in enumerate('grizy'):
                ref.set('ps1_mag_%s' % b, ref.median[:,i])
            apflux = R['apflux']
            exptime = R['exptime']
            apmag = -2.5 * np.log10(apflux / exptime)
            plt.plot(ref.ps1_mag_g - ref.ps1_mag_i,  apmag - ref.ps1_mag_g, '.', label=chip)
            diffs.append(apmag - (ref.ps1_mag_g + R['colorterm']))

        xl,xh = plt.xlim()
        gi = np.linspace(xl, xh, 100)
        fakemag = np.zeros((len(gi),3))
        fakemag[:,0] = gi
        cc = meas.colorterm_ps1_to_observed(fakemag, filt)
        offset = np.median(np.hstack(diffs))
        plt.plot(gi, offset + cc, '-')
        plt.xlim(xl,xh)
        m = np.mean(cc) + offset
        yl,yh = plt.ylim()
        plt.ylim(max(yl, m-1), min(yh, m+1))
        plt.legend()
        plt.xlabel('PS1 g-i (mag)')
        plt.ylabel('%s - g (mag)' % filt)
        ps.savefig()

        plt.clf()
        for chip in goodchips:
            meas,R = chipmeas[chip]
            ref = R['refstars']
            apflux = R['apflux']
            exptime = R['exptime']
            apmag = -2.5 * np.log10(apflux / exptime)
            plt.plot(ref.ps1_mag_g - ref.ps1_mag_i,  apmag - ref.mag, '.', label=chip)
        yl,yh = plt.ylim()
        plt.ylim(max(yl, m-1), min(yh, m+1))
        plt.axhline(-zpt, color='k', linestyle='--')
        plt.legend()
        plt.xlabel('PS1 g-i (mag)')
        plt.ylabel('%s - ref (g+color term) (mag)' % filt)
        ps.savefig()

        plt.clf()
        rr = []
        for chip in goodchips:
            meas,R = chipmeas[chip]
            ref = R['refstars']
            apflux = R['apflux']
            exptime = R['exptime']
            apmag = -2.5 * np.log10(apflux / exptime)
            plt.plot(ref.mag, apmag + zpt, '.', label=chip)
            rr.append(ref.mag)
        rr = np.hstack(rr)
        mn,mx = np.min(rr), np.max(rr)
        lohi = [mn-0.5, mx+0.5]
        plt.plot(lohi, lohi, 'k-', alpha=0.5)
        plt.axis(lohi*2)
        plt.legend()
        plt.xlabel('PS1 ref (mag)')
        plt.ylabel('%s (mag)' % filt)
        ps.savefig()


def process_roi_image(state, roi_settings, roifn):
    pass



if __name__ == '__main__':
    expnum = 1336361

    #acqfn = 'data-ETC/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum)
    acqfn = '~/ibis-data-transfer/guider-acq/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum)

    # 1336360:
    #kwa = dict(
    #    radec_boresight=(351.5373, -1.539),
    #    airmass = 1.15)

    # 1336361: M438
    kwa = dict(
        radec_boresight=(34.8773, -6.181),
        airmass = 1.65)

    astrometry_index_file = '~/data/INDEXES/5200/cfg'
    acq = process_acq_image(acqfn, astrometry_index_file,
                            debug=True,
                            **kwa)

    #roi_settings = {"roi": {"GS1": [1581.9344, 495.1462], "GN2": [1536.8226, 805.9154], "GS2": [195.65, 441.155], "GN1": [1522.0468, 1737.2205]}, "expid": 1336360}
    #roi_selected(acq, roi_settings)
    import json
    roi_settings = json.load(open('/Users/dstn/ibis-data-transfer/guider-acq/roi_settings_%08i.dat' % expnum))

    roinum = 1
    roifn = '~/ibis-data-transfer/guider-sequences/%i/DECam_guider_%i_%08i.fits.gz' % (expnum, expnum, roinum)

    state = acq

    state = process_roi_image(state, roi_settings, roifn)
