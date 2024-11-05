import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.util import Sip

from tractor.sfd import SFDMap
#sys.path.insert(0, 'legacypipe/py')
from legacypipe.ps1cat import ps1cat

# Obsbot isn't a proper module
sys.path.insert(0, 'obsbot')
from measure_raw import RawMeasurer
from camera_decam import nominal_cal
from obsbot import exposure_factor, Neff

import scipy.optimize
from legacypipe.ps1cat import ps1_to_decam

import tractor
import photutils

from astrometry.util.starutil_numpy import hmsstring2ra, dmsstring2dec

sfd = SFDMap()

# Remove a V-shape pattern (MAGIC number)
guider_horiz_slope = 0.00188866

TRACTOR_PARAM_PSFSIGMA = 0
TRACTOR_PARAM_SKY = 1
TRACTOR_PARAM_X = 2
TRACTOR_PARAM_Y = 3
TRACTOR_PARAM_FLUX = 4

class IbisEtc(object):

    def __init__(self):
        self.ps = None
        self.debug = False

    def set_plot_base(self, base):
        if base is None:
            self.ps = None
            self.debug = False
        else:
            from astrometry.util.plotutils import PlotSequence
            self.ps = PlotSequence(os.path.join(procdir, base))
            self.debug = True

    def configure(self,
                  procdir='data-processed',
                  astrometry_config_file=None):
        '''
        Arguments:
        * procdir: string, directory, where we can save temp files
        * astrometry_config_file: string, filename, Astrometry.net config file
        '''
        self.procdir = procdir
        self.astrometry_config_file = astrometry_config_file

    def clear_after_exposure(self):
        # Clear all the data associated with the current science exposure
        self.sci_datetime = None
        self.acq_datetime = None
        self.acq_exptime = None
        self.roi_exptime = None
        self.expnum = None
        self.radec = None
        self.ebv = None
        self.filt = None
        self.airmass = None
        self.chipnames = None
        self.imgs = None
        self.chipmeas = None
        self.transparency = None
        self.transmission = None
        self.nom_zp = None
        self.wcschips = None
        self.goodchips = None
        self.flux0 = None
        self.acc_strips = None
        self.sci_acc_strips = None
        self.strip_skies = None
        self.strip_sig1s = None
        self.acc_strip_skies = None
        self.acc_strip_sig1s = None
        self.sci_acc_strip_skies = None
        self.roi_apfluxes = None
        self.roi_apskies = None
        self.tractors = None
        self.acc_rois = None
        self.tractor_fits = None
        self.roi_datetimes = None
        self.sci_times = None
        self.inst_seeing = None
        self.inst_sky = None
        self.inst_transparency = None
        self.cumul_seeing = None
        self.cumul_sky = None
        self.cumul_transparency = None
        self.efftimes = None

    def process_guider_acq_image(self, acqfn,
                                 fake_header=None):
        '''
        * acqfn: string, filename of guider acquisition (first exposure) FITS file

        * radec_boresight = (ra, dec) where ra,dec are floats, in decimal degrees.
        * airmass: float
        '''
        self.clear_after_exposure()
        print('Reading', acqfn)
        chipnames,imgs,phdr,biases = assemble_full_frames(acqfn)
        # ASSUME that science image starts at the same time as the guider acq image
        self.acq_datetime = datetime_from_header(phdr)
        self.sci_datetime = self.acq_datetime
        self.acq_exptime = float(phdr['GEXPTIME'])
        self.expnum = int(phdr['EXPNUM'])
        self.filt = phdr['FILTER']

        ra = hmsstring2ra(fake_header['RA'])
        dec = dmsstring2dec(fake_header['DEC'])
        self.airmass = float(fake_header['AIRMASS'])
        self.radec = (ra, dec)
        self.ebv = sfd.ebv(ra, dec)[0]

        print('Expnum', self.expnum, 'Filter', self.filt)
        self.chipnames = chipnames
        self.imgs = dict(zip(chipnames, imgs))

        if self.debug:
            plt.clf()

        wcsfns = {}
        imgfns = {}
        self.wcschips = []
        any_img = False
        for i,(chip,img,biaslr) in enumerate(zip(chipnames, imgs, biases)):
            imgfn = os.path.join(procdir, '%s-acq-%s.fits' % (expnum, chip))
            imgfns[chip] = imgfn
            # HACK - speed up re-runs
            wcsfn = os.path.join(procdir, '%s-acq-%s.wcs' % (expnum, chip))
            wcsfns[chip] = wcsfn
            if os.path.exists(wcsfn):
                self.wcschips.append(chip)
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
                   '--config %s ' % self.astrometry_config_file +
                   '--scale-low 0.25 --scale-high 0.27 --scale-units app ' +
                   '--solved none --match none --corr none --new-fits none ' +
                   '--no-tweak ' +
                   '--continue ' +
                   '--depth 30 ' +
                   '--nsigma 6 ')
    
            if self.debug:
                cmd = cmd + '--plot-scale 0.5 '
            else:
                cmd = cmd + '--no-plots '
    
            #if radec_boresight is not None:
            cmd = cmd + '--ra %.4f --dec %.4f --radius 5 ' % (ra, dec)
    
            cmd = cmd + imgfn
            #cmd = cmd + ' -v --no-delete-temp'
            print(cmd)
            rtn = os.system(cmd)
            print('rtn:', rtn)
            if rtn:
                continue
    
            if self.debug:
                any_img = True
                plt.subplot(2,2, i+1)
                plt.imshow(img, origin='lower', interpolation='nearest', vmin=-30, vmax=+50)
                plt.xticks([]); plt.yticks([])
                plt.title(chip)

            if os.path.exists(wcsfn):
                self.wcschips.append(chip)

        if self.debug and any_img:
            plt.suptitle(acqfn)
            self.ps.savefig()
        #state.update(procdir=procdir, goodchips=goodchips, wcsfns=wcsfns, imgfns=imgfns)

        self.chipmeas = {}
        for chip in chipnames:
            print()
            print('Measuring', chip)
            imgfn = imgfns[chip]
            if chip in self.wcschips:
                wcs = Sip(wcsfns[chip])
                p = ps1cat(ccdwcs=wcs)
                stars = p.get_stars()
            else:
                wcs = None
    
            ext = 0
            meas = DECamGuiderMeasurer(imgfn, ext, nominal_cal)
            meas.airmass = self.airmass
            meas.wcs = wcs
            # max astrometric shift, in arcsec (assume astrometry.net solution is pretty good)
            meas.maxshift = 5.
    
            kw = {}
            if self.debug:
                #kw.update(ps=ps)
                meas.debug = True
                meas.ps = self.ps
                pass

            R = meas.run(**kw)
            self.chipmeas[chip] = (meas, R)

        zp0 = None
        kx = None
        dmags = []
        seeings = []
        for chip in self.wcschips:
            meas,R = self.chipmeas[chip]
            ref = R['refstars']
            apflux = R['apflux']
            exptime = R['exptime']
            apmag = -2.5 * np.log10(apflux / exptime)
            dmags.append(apmag - ref.mag)
            seeings.append(R['seeings'])
            if zp0 is None:
                zp0 = meas.zeropoint_for_exposure(self.filt, ext=meas.ext, exptime=exptime,
                                                  primhdr=R['primhdr'])
                kx = nominal_cal.fiducial_exptime(self.filt).k_co
        dmags = np.hstack(dmags)
        seeings = np.hstack(seeings)
        seeing = np.median(seeings)
        zpt = -np.median(dmags)
        print()
        print('All chips:')
        print('Zeropoint:    %.3f   (with %i stars)' % (zpt, len(dmags)))
        print('Seeing:        %.2f arcsec' % seeing)
        del dmags
        self.transparency = 10.**(-0.4 * (zp0 - zpt - kx * (self.airmass - 1.)))
        print('Nom Zeropoint: %.3f' % zp0)
        print('Transparency:  %.3f' % self.transparency)
        self.transmission = 10.**(-0.4 * (zp0 - zpt))
        print('Transmission:  %.3f' % self.transmission)
        self.nom_zp = zp0

        if self.debug:
            plt.clf()
            diffs = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
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
            cc = meas.colorterm_ps1_to_observed(fakemag, self.filt)
            offset = np.median(np.hstack(diffs))
            plt.plot(gi, offset + cc, '-')
            plt.xlim(xl,xh)
            m = np.mean(cc) + offset
            yl,yh = plt.ylim()
            plt.ylim(max(yl, m-1), min(yh, m+1))
            plt.legend()
            plt.xlabel('PS1 g-i (mag)')
            plt.ylabel('%s - g (mag)' % self.filt)
            self.ps.savefig()

            plt.clf()
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
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
            plt.ylabel('%s - ref (g+color term) (mag)' % self.filt)
            self.ps.savefig()

            plt.clf()
            rr = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
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
            plt.ylabel('%s (mag)' % self.filt)
            self.ps.savefig()

    def process_roi_image(self, roi_settings, roi_num, roi_filename,
                          debug=False):
        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0.2)

        ### Question - should we just discard the first ROI frame?  It has different
        # exposure properties than the rest due to the readout timing patterns!
        # [0] is the full-frame image
        # [1] is the first ROI, with funny exposure properties
        # [2] is the firt normal ROI
        if roi_num == 1:
            print('Skipping first ROI image')
            return

        first_time = False
        if roi_num == 2:
            first_time = True
            roi = roi_settings['roi']
            goodchips = []
            flux0 = {}
            for i,chip in enumerate(self.chipnames):
                x,y = roi[chip]
                meas,R = self.chipmeas[chip]
                # Stars detected in the acq. image
                trim_x0, trim_y0 = R['trim_x0'], R['trim_y0']
                det_x = R['all_x'] + trim_x0
                det_y = R['all_y'] + trim_y0
                # Match to ROI
                d = np.hypot(x - det_x, y - det_y)
                j = np.argmin(d)
                if d[j] > 5:
                    print('Chip', chip, ': No close match found for ROI star at (x,y) = %.1f, %.1f' % (x,y))
                    continue
                goodchips.append(chip)
                print('Matched a star detected in acq image, %.1f pix away' % d[j])
                acqflux = R['all_apflux'][j]
                print('Flux in acq image:', acqflux)
                #print('Transmission:', self.transmission)
                flux0[chip] = acqflux / self.transmission
            self.goodchips = goodchips
            self.acq_flux = flux0

            if self.debug:
                plt.clf()
                for i,chip in enumerate(self.chipnames):
                    meas,R = self.chipmeas[chip]
                    x,y = roi[chip]
                    trim_x0, trim_y0 = R['trim_x0'], R['trim_y0']
                    det_x = R['all_x'] + trim_x0
                    det_y = R['all_y'] + trim_y0

                    plt.subplot(2, 4, i+1)
                    img = self.imgs[chip]
                    mn,mx = np.percentile(img.ravel(), [25,98])
                    ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
                    plt.imshow(img, **ima)
                    S = 50
                    ax = plt.axis()
                    if 'refstars' in R:
                        refstars = R['refstars']
                        plt.plot(refstars.x, refstars.y, 'o', mec='r', mfc='none')
                    plt.plot([x-S, x+S, x+S, x-S, x-S], [y-S, y-S, y+S, y+S, y-S], 'r-')
                    plt.axis(ax)
                    plt.xticks([]); plt.yticks([])
                    plt.title(chip)
    
                    ix,iy = int(x), int(y)
                    x0,y0 = ix-S,iy-S
                    slc = slice(iy-S, iy+S+1), slice(ix-S, ix+S+1)
                    plt.subplot(2, 4, 4 + i+1)
                    plt.imshow(img[slc], **ima)
                    ax = plt.axis()
                    if 'refstars' in R:
                        plt.plot(refstars.x-x0, refstars.y-y0, 'o',
                                mec='r', mfc='none', ms=20, mew=3)
                    plt.plot(det_x-x0, det_y-y0, 's',
                             mec='m', mfc='none', ms=10, mew=2)
                    plt.axis(ax)
                    plt.xticks([]); plt.yticks([])
                self.ps.savefig()

            # Init ROI data structures

            self.acc_strips = {}
            self.sci_acc_strips = {}
            self.strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.sci_acc_strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.roi_apfluxes = dict((chip,[]) for chip in self.chipnames)
            self.roi_apskies  = dict((chip,[]) for chip in self.chipnames)
            self.tractors = {}
            self.acc_rois = {}
            self.tractor_fits = dict((chip,[]) for chip in self.chipnames)
            self.roi_datetimes = []
            self.sci_times = []
            self.inst_seeing = dict((chip,[]) for chip in self.chipnames)
            self.inst_sky = dict((chip,[]) for chip in self.chipnames)
            self.inst_transparency = dict((chip,[]) for chip in self.chipnames)
            self.cumul_seeing = []
            self.cumul_sky = []
            self.cumul_transparency = []
            self.efftimes = []

        F = fitsio.FITS(roi_filename, 'r')
        if self.debug:
            self.roi_debug_plots(F)
        chips,imgs,phdr,biases = assemble_full_frames(roi_filename, drop_bias_rows=20)
        if first_time:
            self.roi_exptime = float(phdr['GEXPTIME'])
        else:
            assert(self.roi_exptime == float(phdr['GEXPTIME']))
        troi = datetime_from_header(phdr)
        self.roi_datetimes.append(troi)
        # How much total exposure time has the science exposure had at the end of
        # this guider frame?
        self.sci_times.append((troi - self.sci_datetime).total_seconds() + self.roi_exptime)

        # The bottom ~20-25 rows have a strong ramp, even after bias subtraction.
        # The top row can also go wonky!
        # Estimate the sky level in the remaining pixels
        for chip,img in zip(chips,imgs):
            subimg = img[25:-1, :]
            self.strip_skies[chip].append(np.median(subimg))
            self.strip_sig1s[chip].append(blanton_sky(subimg, step=3))
            if not chip in self.acc_strips:
                self.acc_strips[chip] = subimg.copy()
                dt_sci = self.sci_times[0]
                #print('Science frame time difference:', dt_sci)
                self.sci_acc_strips[chip] = dt_sci * subimg / self.roi_exptime
            else:
                self.acc_strips[chip] += subimg
                dt_sci = self.sci_times[-1] - self.sci_times[-2]
                #print('Science frame time difference:', dt_sci, '-> total', self.sci_times[-1])
                self.sci_acc_strips[chip] += dt_sci * subimg / self.roi_exptime

            self.acc_strip_sig1s[chip].append(blanton_sky(self.acc_strips[chip], step=3))
            self.acc_strip_skies[chip].append(np.median(self.acc_strips[chip]))
            self.sci_acc_strip_skies[chip].append(np.median(self.sci_acc_strips[chip]))

            #print(chip, 'acc  sky rate:', self.sci_acc_strip_skies[chip][-1] / self.sci_times[-1])
            #print(chip, 'inst sky rate:', self.strip_skies[chip][-1] / self.roi_exptime)

        # sky = np.median(np.hstack([img[25:-1,:] for img in imgs]))
        # print('Median sky level:', sky)
        #print('Median sky per chip:',
        #      ', '.join(['%.2f' % self.strip_skies[chip][-1] for chip in chips]))

        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0)
            for i,(chip,img) in enumerate(zip(chips, imgs)):
                plt.subplot(4, 1, i+1)
                plt.imshow(img, interpolation='nearest', origin='lower', vmin=0, vmax=10,
                           aspect='auto')
            plt.suptitle('bias-subtracted images')
            self.ps.savefig()

            plt.clf()
            plt.subplots_adjust(hspace=0)
            for i,(chip,img) in enumerate(zip(chips, imgs)):
                rowmed = np.median(img, axis=1)
                plt.subplot(5, 1, i+1)
                plt.imshow(img - rowmed[:, np.newaxis], interpolation='nearest',
                           origin='lower', vmin=-2, vmax=10, aspect='auto')
                plt.xlim(0, 2048)
            plt.subplot(5, 1, 5)
            for i,(chip,img) in enumerate(zip(chips, imgs)):
                rowmed = np.median(img, axis=1)
                colmed = np.median(img - rowmed[:, np.newaxis], axis=0)
                plt.plot(colmed)
                plt.xlim(0, 2048)
            plt.ylim(-3, +3)
            plt.suptitle('bias- and median-subtracted images')
            self.ps.savefig()

        # Remove a V-shape pattern (MAGIC number)
        h,w = imgs[0].shape
        xx = np.arange(w)
        hbias = np.abs(xx - (w/2 - 0.5)) * guider_horiz_slope
        for img in imgs:
            img -= hbias[np.newaxis,:]

        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0)
            for i,(chip,img) in enumerate(zip(chips, imgs)):
                rowmed = np.median(img, axis=1)
                plt.subplot(5, 1, i+1)
                plt.imshow(img - rowmed[:, np.newaxis], interpolation='nearest',
                           origin='lower', vmin=-2, vmax=10, aspect='auto')
                plt.xlim(0, 2048)
            plt.subplot(5, 1, 5)
            for i,(chip,img) in enumerate(zip(chips, imgs)):
                rowmed = np.median(img, axis=1)
                colmed = np.median(img - rowmed[:, np.newaxis], axis=0)
                plt.plot(colmed)
            plt.xlim(0, 2048)
            plt.ylim(-3, +3)
            plt.suptitle('bias- and median- and V-subtracted images')
            self.ps.savefig()

        # Trim the bottom (bright) pixels off; trim one pixel off the top too!
        Ntrim = 18
        orig_h = imgs[0].shape[0]
        imgs = [img[Ntrim:-1, :] for img in imgs]

        roi_xsize = 25
        roi_imgs = {}

        for i,(img,chip) in enumerate(zip(imgs, chips)):
            roi = roi_settings['roi']
            x,y = roi[chip]
            ix = int(np.round(x))
            roi_imgs[chip] = roi_img = img[:, ix-roi_xsize : ix+roi_xsize+1]
            # Position within the ROI of the star
            roi_starx = roi_xsize
            roi_stary = orig_h // 2 - Ntrim
            sky = self.strip_skies[chip][-1]
            # Aperture photometry
            apxy = np.array([[roi_starx, roi_stary]])
            ap = []
            aprad_pix = 15.
            aper = photutils.aperture.CircularAperture(apxy, aprad_pix)
            p = photutils.aperture.aperture_photometry(roi_img - sky, aper)
            apflux = p.field('aperture_sum')
            apflux = float(apflux.data[0])
            # Mask out star pixels before computing median (sky)
            h,w = roi_img.shape
            x = np.arange(w)
            y = np.arange(h)
            starmask = np.hypot(x[np.newaxis,:] - w/2, y[:,np.newaxis] - h/2) > aprad_pix
            apsky = np.median(roi_img[starmask])
            self.roi_apfluxes[chip].append(apflux)
            self.roi_apskies [chip].append(apsky)

        if self.debug:
            plt.clf()

        orig_roi_imgs = dict((k,v.copy()) for k,v in roi_imgs.items())

        for i,chip in enumerate(chips):
            if not first_time:
                tr = self.tractors[chip]
                tim = tr.images[0]
                # Accumulate ROI image
                tim.data += roi_imgs[chip]
                roi_imgs[chip] = tim.data
            roi_img = roi_imgs[chip]
            # Estimate per-pixel noise via Blanton's MAD
            sig1 = blanton_sky(roi_img, step=3)
            if first_time:
                tim = tractor.Image(roi_img, inverr=np.ones_like(roi_img)/sig1,
                                    psf=tractor.NCircularGaussianPSF([2.], [1.]),
                                    sky=tractor.ConstantSky(sky))
                tim.psf.freezeParam('weights')
                tim.sig1 = sig1
                h,w = roi_img.shape
                sky = self.strip_skies[chip][-1]
                flux = np.sum(roi_img) - sky * h*w
                flux = max(flux, 100)
                src = tractor.PointSource(tractor.PixPos(roi_starx, roi_stary),
                                          tractor.Flux(flux))
                tr = tractor.Tractor([tim], [src])
                self.tractors[chip] = tr
            else:
                # we already accumulated the image into tim.data above.
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1
                tr.optimize_loop(shared_params=False)

            if self.debug:
                plt.subplot(3, 4, 1+i)
                mx = np.percentile(orig_roi_imgs[chip].ravel(), 95)
                plt.imshow(orig_roi_imgs[chip], interpolation='nearest', origin='lower',
                           vmin=-5, vmax=mx)
                plt.title(chip + ' new ROI')
                plt.subplot(3, 4, 5+i)
                mx = np.percentile(roi_img.ravel(), 95)
                plt.imshow(roi_img, interpolation='nearest', origin='lower',
                           vmin=sky-3.*sig1, vmax=mx)
                plt.title(chip + ' acc ROI')
                # plt.subplot(4, 4, 9+i)
                # mod = tr.getModelImage(0)
                # plt.imshow(mod, interpolation='nearest', origin='lower',
                #            vmin=sky-3.*sig1, vmax=mx)
                # plt.title(chip + ' init mod')

            opt_args = dict(shared_params=False)
            X = tr.optimize_loop(**opt_args)
            s = tim.psf.getParams()[0]
            if s < 0:
                ### Wtf
                tim.psf.setParams([np.abs(s)])
                tr.optimize_loop(**opt_args)

            if self.debug:
                plt.subplot(3, 4, 9+i)
                mod = tr.getModelImage(0)
                plt.imshow(mod, interpolation='nearest', origin='lower',
                           vmin=sky-3.*sig1, vmax=mx)
                plt.title(chip + ' fit mod')

            self.acc_rois[chip] = (roi_img, tim.inverr, tr.getModelImage(0))

            self.tractor_fits[chip].append(tr.getParams())
            #  images.image0.psf.sigmas.param0 = 2.2991302175858706
            #  images.image0.sky.sky = 6.603832232554123
            #  catalog.source0.pos.x = 24.43478222069596
            #  catalog.source0.pos.y = 10.50428514888774
            #  catalog.source0.brightness.Flux = 55101.9692526979

        if first_time:
            self.roiflux = {}
            for chip in self.goodchips:
                self.roiflux[chip] = self.tractor_fits[chip][-1][TRACTOR_PARAM_FLUX]
                #print(chip, 'Flux in first ROI image:', self.roiflux[chip])

        if self.debug:
            self.ps.savefig()

        pixsc = nominal_cal.pixscale
        seeing = (np.mean([self.tractor_fits[chip][-1][TRACTOR_PARAM_PSFSIGMA]])
            * 2.35 * pixsc)
        skyrate = np.mean([self.sci_acc_strip_skies[chip][-1]
                           for chip in self.chipnames]) / self.sci_times[-1]
        skybr = -2.5 * np.log10(skyrate /pixsc/pixsc) + self.nom_zp
        # HACK -- arbitrary sky correction to match copilot
        skybr += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
        skybr = np.mean(skybr)

        # cumulative, sci-averaged transparency
        itrs = []
        trs = []
        for chip in self.chipnames:
            dflux = np.diff([params[TRACTOR_PARAM_FLUX]
                             for params in self.tractor_fits[chip]])
            flux0 = self.tractor_fits[chip][0][TRACTOR_PARAM_FLUX]
            tr = np.append(1., (dflux / flux0)) * self.transparency
            itrs.append(tr[-1])
            dsci = np.append(self.sci_times[0], np.diff(self.sci_times))
            tr = np.cumsum(tr * dsci) / etc.sci_times
            trs.append(tr[-1])
        trans = np.mean(trs)

        # Instantaneous
        for i,chip in enumerate(self.chipnames):
            if len(self.sci_times) > 1:
                iskyrate = ((self.sci_acc_strip_skies[chip][-1] - self.sci_acc_strip_skies[chip][-2]) /
                            (self.sci_times[-1] - self.sci_times[-2]))
            else:
                iskyrate = self.sci_acc_strip_skies[chip][-1] / self.sci_times[-1]
            iskybr = -2.5 * np.log10(iskyrate /pixsc/pixsc) + self.nom_zp
            # HACK -- arbitrary sky correction to match copilot
            iskybr += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
            self.inst_sky[chip].append(iskybr)

            isee = self.tractor_fits[chip][-1][TRACTOR_PARAM_PSFSIGMA] * 2.35 * pixsc
            self.inst_seeing[chip].append(isee)

            self.inst_transparency[chip].append(itrs[i])

        fid = nominal_cal.fiducial_exptime(self.filt)
        expfactor = exposure_factor(fid, nominal_cal, self.airmass, self.ebv,
                                    seeing, skybr, trans)
        efftime = self.sci_times[-1] / expfactor
        print('ROI', roi_num, 'sci exp time %.1f sec' % self.sci_times[-1],
              'efftime %.1f sec' % efftime,
              'seeing %.2f arcsec,' % seeing,
              'sky %.2f mag/arcsec^2,' % skybr,
              'transparency %.1f %%' % (100.*trans))

        self.cumul_sky.append(skybr)
        self.cumul_transparency.append(trans)
        self.cumul_seeing.append(seeing)
        self.efftimes.append(efftime)

    def roi_debug_plots(self, F):
        nguide = 4
        biasimgs = []
        dataimgs = []
        plt.clf()
        for j in range(nguide):
            # 2 amps
            for k in range(2):
                hdu = j*2 + k + 1
                im1  = F[hdu].read()
                hdr1 = F[hdu].read_header()
                biassec1 = hdr1['BIASSEC'].strip('[]').split(',')
                assert(len(biassec1) == 2)
                (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in biassec1]]
                bias = im1[y0-1:y1, x0-1:x1]
                chipname = hdr1['DETPOS']
                biasimgs.append(bias)
                #print('Chip', chipname, 'amp', k+1, ': bias shape', bias.shape)
                plt.subplot(2,4, hdu)
                mn = np.percentile(np.median(bias, axis=1), 5)
                plt.imshow(bias, interpolation='nearest', origin='lower', vmin=mn, vmax=mn+250)
                # The bottom ~10 rows are much brighter; but use 25 to match data
                #bias = bias[25:, :]
                #bias = np.median(bias)
                datasec1 = hdr1['DATASEC'].strip('[]').split(',')
                assert(len(datasec1) == 2)
                (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec1]]
                data = im1[y0-1:y1, x0-1:x1]
                dataimgs.append(data)
                #print('Data shape', data.shape)
                # The bottom ~25 (!) rows are significantly larger.
                #data = data[25:, :]
                #sky += np.median(data) - bias
        plt.suptitle('Bias')
        self.ps.savefig()

        plt.clf()
        for i,img in enumerate(dataimgs):
            plt.subplot(2, 4, i+1)
            mn = np.percentile(np.median(img, axis=1), 5)
            plt.imshow(img, interpolation='nearest', origin='lower', aspect='auto', vmin=mn, vmax=mn+250)
        plt.suptitle('Data')
        self.ps.savefig()

        plt.clf()
        for i,(bias,img) in enumerate(zip(biasimgs, dataimgs)):
            p = plt.plot(np.median(bias, axis=1), '--')
            plt.plot(np.median(img, axis=1), '-', color=p[0].get_color())
        plt.suptitle('medians: bias (dashed), data (solid)')
        plt.xlabel('Row')
        plt.ylabel('Counts')
        self.ps.savefig()
        
        plt.clf()
        for i,(bias,img) in enumerate(zip(biasimgs, dataimgs)):
            medbias = np.median(bias, axis=1)
            plt.plot(np.median(img - medbias[:,np.newaxis], axis=1), '-')
        plt.suptitle('bias-subtracted medians')
        plt.xlabel('Row')
        plt.ylabel('Counts')
        self.ps.savefig()

def datetime_from_header(hdr):
    t = hdr['UTSHUT'] #= '2024-02-29T01:45:30.524' / exp start
    dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
    return dt

def blanton_sky(img, dist=5, step=10):
    # Estimate per-pixel noise via Blanton's 5-pixel MAD
    slice1 = (slice(0,-dist,step),slice(0,-dist,step))
    slice2 = (slice(dist,None,step),slice(dist,None,step))
    mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    return sig1

def assemble_full_frames(fn, drop_bias_rows=48):
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

    ZEROPOINT_OFFSET = -2.5 * np.log10(2.24)
    SKY_BRIGHTNESS_CORRECTION = 1.21

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_trim = 50
        # HACK
        self.airmass = 1.0
        self.gain = 1.0
        # Star detection threshold
        self.det_thresh = 6.
        self.debug = False
        self.ps = None

    def zeropoint_for_exposure(self, band, **kwa):
        zp0 = super().zeropoint_for_exposure(band, **kwa)
        # the superclass applies a GAIN
        if zp0 is None:
            return zp0
        # HACK -- correct for GAIN difference.... shouldn't this be ~3.5 ???
        return zp0 + DECamGuiderMeasurer.ZEROPOINT_OFFSET

    def read_raw(self, F, ext):
        img,hdr = super().read_raw(F, ext)
        img = img * self.gain
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
        xx = np.arange(w)
        hbias = np.abs(xx - (w/2 - 0.5)) * guider_horiz_slope * self.gain
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
        # Guider readout times: 160 us to shift a row.  4 us per pixel.
        # 1024 pix + overscan = 1080 pix/row
        # = 4480 us per row
        row_exptime = 4480e-6
        fake_exptime = self.get_exptime(self.primhdr)
        sky1 = fake_exptime * m / row_exptime

        print('Ramp rate:', m / row_exptime, 'counts/sec')

        # Remove sky ramp
        skymod += (m * x)[:, np.newaxis]
        sig1 = blanton_sky(img - skymod)

        if self.debug:
            plt.clf()
            #plt.subplot(1,2,1)
            mn,mx = np.percentile(img.ravel(), [5,95])
            plt.imshow(img, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.colorbar()
            plt.title('Acq image')
            self.ps.savefig()
            
            plt.clf()
            #plt.subplot(1,2,2)
            mn,mx = np.percentile((img-skymod).ravel(), [5,95])
            plt.imshow(img-skymod, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.colorbar()
            plt.title('Sky-sub image')
            self.ps.savefig()

            plt.clf()
            plt.plot(xx, ramp, '-', label='Row-wise median')
            plt.plot(xx, b + m*xx, '-', label='Ramp model')
            plt.legend()
            self.ps.savefig()

        return skymod, sky1, sig1

    def get_wcs(self, hdr):
        return self.wcs

    def get_ps1_band(self, band):
        print('Band', band)
        return ps1cat.ps1band[band]

    def colorterm_ps1_to_observed(self, ps1stars, band):
        return ps1_to_decam(ps1stars, band)

# Measure_raw() returns:
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

if __name__ == '__main__':
    import json
    import pickle

    expnum = 1336361
    # 1336360:
    #kwa = dict(
    #    radec_boresight=(351.5373, -1.539),
    #    airmass = 1.15)
    # 1336361: M438
    kwa = dict(
        radec_boresight=(34.8773, -6.181),
        airmass = 1.65)

    procdir = 'data-processed2'
    astrometry_config_file = '~/data/INDEXES/5200/cfg'

    #acqfn = 'data-ETC/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum)
    acqfn = '~/ibis-data-transfer/guider-acq/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum)

    if not os.path.exists(procdir):
        try:
            os.makedirs(procdir)
        except:
            pass

    statefn = 'state.pickle'
    if not os.path.exists(statefn):

        from astrometry.util.starutil import ra2hmsstring, dec2dmsstring
        phdr = fitsio.read_header(acqfn)
        fake_header = dict(RA=ra2hmsstring(kwa['radec_boresight'][0], separator=':'),
                        DEC=dec2dmsstring(kwa['radec_boresight'][1], separator=':'),
                        AIRMASS=kwa['airmass'],
                        SCI_UT=phdr['UTSHUT'])

        etc = IbisEtc()
        etc.configure(procdir, astrometry_config_file)
        etc.set_plot_base('acq')
        etc.process_guider_acq_image(acqfn, fake_header)

        f = open(statefn,'wb')
        pickle.dump(etc, f)
        f.close()
    else:
        etc = pickle.load(open(statefn, 'rb'))


    state2fn = 'state2.pickle'
    if not os.path.exists(state2fn):

        roi_settings = json.load(open('/Users/dstn/ibis-data-transfer/guider-acq/roi_settings_%08i.dat' % expnum))

        for roi_num in range(1, 100):
            roi_filename = '~/ibis-data-transfer/guider-sequences/%i/DECam_guider_%i_%08i.fits.gz' % (expnum, expnum, roi_num)
            #etc.set_plot_base('roi-%03i' % roi_num)
            etc.set_plot_base(None)
            etc.process_roi_image(roi_settings, roi_num, roi_filename)

        f = open(state2fn,'wb')
        pickle.dump(etc, f)
        f.close()
    else:
        etc = pickle.load(open(state2fn, 'rb'))

    from astrometry.util.plotutils import PlotSequence
    ps = PlotSequence(os.path.join(procdir, 'roi-summary'))

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(etc.roi_apfluxes[chip], '.-', label=chip)
    # plt.ylabel('Aperture fluxes')
    # plt.legend()
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(etc.roi_apskies[chip], '.-', label=chip)
    # plt.ylabel('Aperture skies')
    # plt.legend()
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(etc.strip_skies[chip], '.-', label=chip)
    # plt.ylabel('Strip skies')
    # plt.legend()
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(etc.acc_strip_skies[chip], '.-', label=chip)
    # plt.ylabel('Accumulated strip skies')
    # plt.legend()
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(etc.acc_strip_sig1s[chip], '.-', label=chip)
    # plt.ylabel('Accumulated strip sig1s')
    # plt.legend()
    # ps.savefig()

    plt.clf()
    plt.subplots_adjust(hspace=0)
    mx = np.percentile(np.hstack([x.ravel() for x in etc.acc_strips.values()]), 98)
    for i,chip in enumerate(etc.chipnames):
        plt.subplot(4,1,i+1)
        plt.imshow(etc.acc_strips[chip], interpolation='nearest', origin='lower',
                   aspect='auto', vmin=0, vmax=mx)
        plt.xticks([]); plt.yticks([])
        plt.ylabel(chip)
    plt.suptitle('Accumulated strips')
    ps.savefig()

    plt.clf()
    for i,chip in enumerate(etc.chipnames):
        img, ie, mod = etc.acc_rois[chip]
        mn,mx = np.percentile(img, [25,98])
        ima = dict(interpolation='nearest', origin='lower', vmin=0, vmax=mx)
        plt.subplot(3,4, i+1)
        plt.imshow(img, **ima)
        plt.title(chip)
        plt.xticks([]); plt.yticks([])
        plt.subplot(3,4, i+5)
        plt.imshow(mod, **ima)
        plt.xticks([]); plt.yticks([])
        plt.subplot(3,4, i+9)
        plt.imshow((img - mod) * ie, interpolation='nearest', origin='lower',
                   vmin=-10, vmax=+10)
        plt.xticks([]); plt.yticks([])
    plt.suptitle('Accumulated ROIs')
    ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(np.cumsum(etc.strip_skies[chip]), etc.acc_strip_sig1s[chip], '.-', label=chip)
    # plt.xlabel('Accumulated Strip skies')
    # plt.ylabel('Accumulated Strip sig1s')
    # plt.legend()
    # ps.savefig()

    # all_params = etc.tractor_fits
    # paramnames = ['PSF sigma', 'Sky', 'X', 'Y', 'Flux']
    # plotparams = [
    #     #(TRACTOR_PARAM_SKY,True),
    #     (TRACTOR_PARAM_PSFSIGMA,False),
    #     #(TRACTOR_PARAM_FLUX,True)
    #     ]
    # for p,delta in plotparams:
    #     plt.clf()
    #     for chip in etc.chipnames:
    #         plt.plot([params[p] for params in all_params[chip]], '.-', label=chip)
    #     plt.ylabel('tractor ' + paramnames[p])
    #     plt.legend()
    #     ps.savefig()
    # 
    #     if not delta:
    #         continue
    #     plt.clf()
    #     for chip in etc.chipnames:
    #         pl = plt.plot(np.diff([params[p] for params in all_params[chip]]), '.-', label=chip)
    #         if p == TRACTOR_PARAM_FLUX:
    #             plt.axhline(all_params[chip][0][p], linestyle='--', color=pl[0].get_color())
    #     plt.ylabel('tractor delta ' + paramnames[p])
    #     
    #     plt.legend()
    #     ps.savefig()

    plt.clf()
    for chip in etc.chipnames:
        plt.plot(etc.sci_times, etc.inst_seeing[chip], '.-', label=chip)
    plt.legend()
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Accumulated Seeing (arcsec)')
    ps.savefig()

    plt.clf()
    for chip in etc.chipnames:
        plt.plot(etc.sci_times, etc.inst_transparency[chip], '.-', label=chip)
    plt.legend()
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Transparency (instantaneous)')
    ps.savefig()

    plt.clf()
    plt.plot(etc.sci_times, etc.cumul_transparency, 'k.-')
    plt.legend()
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Transparency (cumulative)')
    ps.savefig()


    # # Cumulative science-exposure-time-weighted transparency estimate
    # transparencies = {}
    # dsci = np.append(etc.sci_times[0], np.diff(etc.sci_times))
    # plt.clf()
    # for chip in etc.chipnames:
    #     tr = exp_transparencies[chip]
    #     tr = np.cumsum(tr * dsci) / etc.sci_times
    #     transparencies[chip] = tr
    #     plt.plot(tr, '.-', label=chip)
    # plt.ylabel('Transparency (cumulative)')
    # ps.savefig()

    # # Estimate total sky accumulated in science image -- from tractor fits
    # tt = dict((chip,[]) for chip in etc.chipnames)
    # scisky = dict((chip,[]) for chip in etc.chipnames)
    # print('Acq image start:', etc.acq_datetime)
    # plt.clf()
    # for chip in etc.chipnames:
    #     tt[chip].append(0)
    #     scisky[chip].append(0)
    #     meas,R = etc.chipmeas[chip]
    #     exptime = R['exptime']
    #     # First (actually second) ROI frame
    #     # Time from science image start end of first ROI frame
    #     first_roi = 0
    #     dt = etc.roi_datetimes[first_roi] + timedelta(seconds=exptime) - etc.sci_datetime
    #     dt = dt.total_seconds()
    #     tt[chip].append(dt)
    #     skyrate = etc.tractor_fits[chip][first_roi][TRACTOR_PARAM_SKY] / exptime
    #     scisky[chip].append(dt * skyrate)
    #     # Subsequent ROI frames - sky deltas
    #     skydeltas = np.diff([params[TRACTOR_PARAM_SKY] for params in etc.tractor_fits[chip][first_roi:]])
    #     skyrates = []
    #     dts = []
    #     for dsky,t,t_prev in zip(skydeltas, etc.roi_datetimes[first_roi+1:], etc.roi_datetimes[first_roi:]):
    #         dt = (t - t_prev).total_seconds()
    #         skyrate = dsky / exptime
    #         skyrates.append(skyrate)
    #         dts.append(dt)
    #     tt[chip].extend(tt[chip][-1] + np.cumsum(dts))
    #     scisky[chip].extend(scisky[chip][-1] + np.cumsum(np.array(dts) * np.array(skyrates)))
    #     plt.plot(tt[chip], scisky[chip], '.-', label=chip)
    # plt.xlabel('Elapsed time (sec)')
    # plt.ylabel('Predicted science image sky (counts)')
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     meas,R = etc.chipmeas[chip]
    #     skyrate = np.array(scisky[chip][1:]) / np.array(tt[chip][1:])
    #     pixsc = R['pixscale']
    #     skybr = -2.5 * np.log10(skyrate /pixsc/pixsc) + etc.zp0
    #     plt.plot(tt[chip][1:], skybr, '.-', label=chip)
    # plt.ylabel('Sky brightness (mag/arcsec^2)')
    # plt.legend()
    # ps.savefig()

    # plt.clf()
    # for chip in etc.chipnames:
    #     plt.plot(np.append(0,etc.sci_times), np.append(0, etc.sci_acc_strip_skies[chip]), '.-', label=chip)
    # plt.legend()
    # plt.xlabel('Science exposure time (sec)')
    # plt.ylabel('Predicted science image sky (counts)')
    # #plt.title('from sci-weighted acc strip')
    # ps.savefig()

    # skybrights = {}
    # plt.clf()
    # for chip in etc.chipnames:
    #     skyrate = np.array(etc.sci_acc_strip_skies[chip]) / np.array(etc.sci_times)
    #     meas,R = etc.chipmeas[chip]
    #     pixsc = R['pixscale']
    #     skybr = -2.5 * np.log10(skyrate /pixsc/pixsc) + etc.nom_zp
    # 
    #     # HACK -- arbitrary sky correction to match copilot
    #     skybr += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
    # 
    #     skybrights[chip] = skybr
    #     print(chip, 'Final sky rate:', skyrate[-1])
    #     plt.plot(etc.sci_times, skybr, '.-', label=chip)
    # plt.xlabel('Science exposure time (sec)')
    # plt.ylabel('Average sky brightness (mag/arcsec^2)')
    # #plt.title('from sci-weighted acc strip')
    # plt.legend()
    # ps.savefig()

    plt.clf()
    for chip in etc.chipnames:
        plt.plot(etc.sci_times, etc.inst_sky[chip], '.-', label=chip)
    plt.legend()
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Sky brightness (instantaneous) (mag/arcsec^2)')
    ps.savefig()

    plt.clf()
    plt.plot(etc.sci_times, etc.cumul_sky, 'k.-')
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Sky brightness (cumulative) (mag/arcsec^2)')
    ps.savefig()
    

    # Copilot terminology:
    # efftime = exptime / expfactor
    fid = nominal_cal.fiducial_exptime(etc.filt)
    ebv = etc.ebv
    exptimes = np.array(etc.sci_times)
    expfactors = np.zeros(len(exptimes))
    for i in range(len(exptimes)):
        expfactor = exposure_factor(fid, nominal_cal, etc.airmass, ebv,
                                    etc.cumul_seeing[i], etc.cumul_sky[i],
                                    etc.cumul_transparency[i])
        expfactors[i] = expfactor

    transp = np.vstack([etc.inst_transparency[chip] for chip in etc.chipnames]).mean(axis=0)
    skybr = np.vstack([etc.inst_sky[chip] for chip in etc.chipnames]).mean(axis=0)

    pixsc = nominal_cal.pixscale
    plt.clf()
    neff_fid = Neff(fid.seeing, pixsc)
    neff     = Neff(np.array(etc.cumul_seeing), pixsc)
    efftime_seeing = neff_fid / neff
    efftime_trans = transp**2
    efftime_airmass = 10.**-(0.8 * fid.k_co * (etc.airmass - 1.))
    efftime_sky = 10.**(0.4 * (skybr - fid.skybright))
    efftime_ebv = 10.**(-0.8 * fid.A_co * ebv)
    plt.semilogy(exptimes, efftime_seeing, '-', label='Seeing')
    plt.semilogy(exptimes, efftime_trans, '-', label='Transparency')
    plt.semilogy(exptimes, efftime_sky, '-', label='Sky brightness')
    plt.axhline(efftime_airmass, color='r', label='Airmass')
    plt.axhline(efftime_ebv, color='k', label='Dust extinction')
    plt.xlim(exptimes.min(), exptimes.max())
    plt.legend()
    plt.ylabel('Efftime factor')
    plt.xlabel('Science exposure time (sec)')
    ps.savefig()

    plt.clf()
    plt.plot(exptimes, exptimes / expfactors, '.-')
    plt.xlabel('Science exposure time (sec)')
    plt.ylabel('Effective time (sec)')
    ps.savefig()

