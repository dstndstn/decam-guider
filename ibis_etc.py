import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pylab as plt
import matplotlib
import time

import pyfftw

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
        self.acc_whole_strips = None
        self.acc_biases = None
        self.sci_acc_strips = None
        self.strip_skies = None
        self.strip_sig1s = None
        self.acc_strip_skies = None
        self.acc_strip_sig1s = None
        self.sci_acc_strip_skies = None
        self.roi_apfluxes = None
        self.roi_apskies = None
        self.tractors = None
        self.inst_tractors = None
        self.acc_rois = None
        self.tractor_fits = None
        self.roi_datetimes = None
        self.sci_times = None
        self.inst_seeing = None
        self.inst_seeing_2 = None
        self.inst_sky = None
        self.inst_transparency = None
        self.cumul_seeing = None
        self.cumul_sky = None
        self.cumul_transparency = None
        self.efftimes = None
        self.first_roi_datetime = None
        self.rois = None

    def process_guider_acq_image(self, acqfn,
                                 fake_header=None):
        '''
        * acqfn: string, filename of guider acquisition (first exposure) FITS file

        * radec_boresight = (ra, dec) where ra,dec are floats, in decimal degrees.
        * airmass: float
        '''
        self.clear_after_exposure()
        print('Reading', acqfn)
        chipnames,imgs,phdr,biases,_ = assemble_full_frames(acqfn, fit_exp=False)
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
            kw.update(get_image=True)

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

        for chip in self.wcschips:
            meas,R = self.chipmeas[chip]
            ref = R['refstars']
            for i,b in enumerate('grizy'):
                ref.set('ps1_mag_%s' % b, ref.median[:,i])

        if self.debug:
            plt.clf()
            diffs = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
                ref = R['refstars']
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
            phdr = fitsio.read_header(roi_filename)
            troi = datetime_from_header(phdr)
            self.first_roi_datetime = troi

            self.rois = roi_settings['roi']
            
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
            self.acc_whole_strips = {}
            self.acc_biases = {}
            self.sci_acc_strips = {}
            self.all_sci_acc_strips = {}
            self.all_acc_biases = {}
            self.strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.sci_acc_strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.roi_apfluxes = dict((chip,[]) for chip in self.chipnames)
            self.roi_apskies  = dict((chip,[]) for chip in self.chipnames)
            self.tractors = {}
            self.inst_tractors = {}
            self.acc_rois = {}
            self.tractor_fits = dict((chip,[]) for chip in self.chipnames)
            self.roi_datetimes = []
            self.sci_times = []
            self.inst_seeing = dict((chip,[]) for chip in self.chipnames)
            self.inst_seeing_2 = dict((chip,[]) for chip in self.chipnames)
            self.inst_sky = dict((chip,[]) for chip in self.chipnames)
            self.inst_transparency = dict((chip,[]) for chip in self.chipnames)
            self.cumul_seeing = []
            self.cumul_sky = []
            self.cumul_transparency = []
            self.efftimes = []
            self.dt_walls = []

        F = fitsio.FITS(roi_filename, 'r')
        if self.debug and False:
            self.roi_debug_plots(F)
        print('Reading', roi_filename)
        kw = {}
        if self.debug and first_time:
            kw.update(ps=self.ps)
        chips,imgs,phdr,biasvals,biasimgs = assemble_full_frames(roi_filename, **kw)
        #drop_bias_rows=20, 
        # # Remove a V-shape pattern (MAGIC number)
        # for chip,img in zip(chips, imgs):
        #     h,w = img.shape
        #     xx = np.arange(w)
        #     hbias = np.abs(xx - (w/2 - 0.5)) * guider_horiz_slope
        #     img -= hbias[np.newaxis,:]

        if first_time:
            self.roi_exptime = float(phdr['GEXPTIME'])
        else:
            assert(self.roi_exptime == float(phdr['GEXPTIME']))
        troi = datetime_from_header(phdr)
        self.roi_datetimes.append(troi)
        # How much total exposure time has the science exposure had at the end of
        # this guider frame?
        self.sci_times.append((troi - self.sci_datetime).total_seconds() + self.roi_exptime)

        if first_time:
            dt_wall = (self.roi_datetimes[-1] - self.first_roi_datetime).total_seconds()
            dt_sci = self.sci_times[0]
        else:
            dt_wall = (self.roi_datetimes[-1] - self.roi_datetimes[-2]).total_seconds()
            dt_sci = self.sci_times[-1] - self.sci_times[-2]
        print('Wall time from start of last ROI to start of this one:', dt_wall)
        print('dt sci: %.3f, dt wall: %.3f' % (dt_sci, dt_wall))

        self.dt_walls.append(dt_wall)

        # The bottom ~20-25 rows have a strong ramp, even after bias subtraction.
        # The top row can also go wonky!
        # Estimate the sky level in the remaining pixels
        for ichip,(chip,img,biases) in enumerate(zip(chips,imgs,biasvals)):
            subimg = img[25:-1, :]
            self.strip_skies[chip].append(np.median(subimg))
            self.strip_sig1s[chip].append(blanton_sky(subimg, step=3))
            bl,br = biases
            if not chip in self.acc_strips:
                self.acc_strips[chip] = subimg.copy()
                self.acc_whole_strips[chip] = img.copy()
                self.acc_biases[chip+'_L'] = (biasimgs[ichip*2].copy() - bl)
                self.acc_biases[chip+'_R'] = (biasimgs[ichip*2+1].copy() - br)
                self.sci_acc_strips[chip] = dt_sci * subimg.copy() / dt_wall
                self.all_sci_acc_strips[chip] = [self.sci_acc_strips[chip].copy()]
                self.all_acc_biases[chip+'_L'] = [self.acc_biases[chip+'_L'].copy()]
                self.all_acc_biases[chip+'_R'] = [self.acc_biases[chip+'_R'].copy()]
            else:
                # HACK extraneous .copy(), don't think we need them
                self.acc_strips[chip] += subimg.copy()
                self.acc_whole_strips[chip] += img.copy()
                self.acc_biases[chip+'_L'] += (biasimgs[ichip*2].copy() - bl)
                self.acc_biases[chip+'_R'] += (biasimgs[ichip*2+1].copy() - br)
                self.sci_acc_strips[chip] += (dt_sci * subimg.copy() / dt_wall)
                self.all_sci_acc_strips[chip].append(self.sci_acc_strips[chip].copy())
                self.all_acc_biases[chip+'_L'].append(self.acc_biases[chip+'_L'].copy())
                self.all_acc_biases[chip+'_R'].append(self.acc_biases[chip+'_R'].copy())

            self.acc_strip_sig1s[chip].append(blanton_sky(self.acc_strips[chip], step=3))
            self.acc_strip_skies[chip].append(np.median(self.acc_strips[chip]))
            self.sci_acc_strip_skies[chip].append(np.median(self.sci_acc_strips[chip]))

            print(chip, 'subimage median:', np.median(subimg))
            #print(chip, 'acc  sky rate:', self.sci_acc_strip_skies[chip][-1] / self.sci_times[-1])
            print(chip, 'sci acc sky: %.2f' % self.sci_acc_strip_skies[chip][-1])

            print(chip, 'inst sky rate: %.2f counts/sec/pixel' % (self.strip_skies[chip][-1] / dt_wall))

        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0)
            for ichip,(chip,img) in enumerate(zip(chips,imgs)):
                subimg = img[25:-1, :]
                plt.subplot(8,1,ichip+1)
                m = np.median(subimg.ravel())
                r = 10
                lo,md,hi = np.percentile(subimg.ravel(), [10,50,90])
                print(chip, 'subimg percentiles: %.2f %.2f %.2f' % (lo,md,hi))
                plt.imshow(subimg, interpolation='nearest', origin='lower',
                           vmin=m-r, vmax=m+r, aspect='auto')
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            for ichip,(chip,img) in enumerate(zip(chips,imgs)):
                plt.subplot(8,1,ichip+5)
                #m = np.median(self.sci_acc_strips[chip].ravel())
                # ss = self.sci_acc_strip_skies[chip]
                # if len(ss) > 1:
                #     m2 = ss[-2] + m
                # else:
                #     m2 = ss[-1] + m
                # lo,md,hi = np.percentile(self.sci_acc_strips[chip].ravel(), [10,50,90])
                # print(chip, 'sci percentiles: %.2f %.2f %.2f' % (lo, md, hi))
                # plt.imshow(self.sci_acc_strips[chip], interpolation='nearest', origin='lower',
                #            vmin=m2-r*10, vmax=m2+r*10, aspect='auto')
                ss = self.acc_strip_skies[chip]
                if len(ss) > 1:
                    m2 = ss[-2] + m
                else:
                    m2 = ss[-1] + m
                plt.imshow(self.acc_strips[chip], interpolation='nearest', origin='lower',
                            vmin=m2-r*10, vmax=m2+r*10, aspect='auto')
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            #plt.suptitle('ROI images and Science-weighted accumulated')
                plt.suptitle('ROI images and Accumulated')
            self.ps.savefig()

            # plt.clf()
            # for ichip,(chip,biasval) in enumerate(zip(chips,biasvals)):
            #     bl,br = biasval
            #     biasl = biasimgs[ichip*2  ] - bl
            #     biasr = biasimgs[ichip*2+1] - br
            # 
            #     # Fit an exponential drop-off plus a constant
            #     def objective(params, x, b):
            #         offset, eamp, escale = params
            #         model = offset + eamp * np.exp(-x / escale)
            #         return np.sum(np.abs(b - model[:,np.newaxis]))
            # 
            #     models = []
            #     #meds = []
            #     for bias in [biasl, biasr]:
            #         # Trim off top row -- it sometimes glitches!
            #         bias = bias[:-1, :]
            # 
            #         plt.subplot(2,1,1)
            #         plt.plot(np.median(bias, axis=0), '-')
            #         plt.plot(np.median(bias, axis=1), '-')
            # 
            #         med = np.median(bias, axis=1)
            #         #meds.append(med)
            #         N,_ = bias.shape
            #         x = np.arange(N)
            #         #print('Starting fit...')
            #         r = scipy.optimize.minimize(objective, (1., med[0], 7.), args=(x, bias),
            #                                     method='Nelder-Mead')#tol=1e-3)
            #         #print('bias fit opt result:', r)
            #         offset, eamp, escale = r.x
            #         model = offset + eamp * np.exp(-x / escale)
            #         plt.plot(x, model, '--', color='k', alpha=0.3)
            #         models.append(model)
            # 
            #         plt.subplot(2,1,2)
            #         plt.plot(med - model, '-')
            #         plt.xlabel('Overscan row (pixels)')
            #         plt.ylabel('Row-wise median - model')
            # plt.suptitle('Bias image vs exponential decay')
            # self.ps.savefig()
            # 
            # def objective2(params, x, b):
            #     offset, slope, eamp, escale = params
            #     model = offset + slope * x + eamp * np.exp(-x / escale)
            #     r = np.sum(np.abs(b - model[:,np.newaxis]))
            #     return r
            # 
            # plt.clf()
            # for ichip,(chip,img) in enumerate(zip(chips,imgs)):
            #     # First row doesn't seem to fit the exponential model
            #     #img = img[1:-1, :]
            #     # Last row sometimes glitches
            #     img = img[:-1, :]
            #     med = np.median(img, axis=1)
            #     plt.subplot(2,1,1)
            #     plt.plot(med, '-')
            #     plt.yscale('log')
            #     plt.ylim(0.01, 1e3)
            #     N,_ = img.shape
            #     x = np.arange(N)
            #     r = scipy.optimize.minimize(objective2, (1., 0.01, med[0], 7.), args=(x, img),
            #                                 method='Nelder-Mead')
            #     offset, slope, eamp, escale = r.x
            #     model = offset + slope * x + eamp * np.exp(-x / escale)
            #     plt.plot(x, model, '--', color='k', alpha=0.3)
            #     plt.subplot(2,1,2)
            #     plt.plot(med - model, '-')
            #     plt.ylim(-1, +1)
            #     plt.xlabel('ROI image row (pixels)')
            #     plt.ylabel('Row-wise median - model')
            # plt.suptitle('ROI image vs exponential decay + slope')
            # self.ps.savefig()

        for chip in chips:
            print(chip, 'cumulative sky rate: %.2f counts/sec/pixel' % (self.acc_strip_skies[chip][-1] / sum(self.dt_walls)))

        if self.debug and False:
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
        pixsc = nominal_cal.pixscale

        for i,chip in enumerate(chips):
            if not first_time:
                itr = self.inst_tractors[chip]
                tim = itr.images[0]
                tim.data = roi_imgs[chip].copy()
                sig1 = blanton_sky(tim.data, step=3)
                tim.inverr[:,:] = 1./sig1

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

                itr = tr.copy()
                self.inst_tractors[chip] = itr

            else:
                # we already accumulated the image into tim.data above.
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1
                tr.optimize_loop(shared_params=False)

            itr.optimize_loop(shared_params=False)
            #print('Instantaneous tractor fit:')
            #itr.printThawedParams()
            isee = itr.getParams()[TRACTOR_PARAM_PSFSIGMA] * 2.35 * pixsc
            self.inst_seeing_2[chip].append(isee)

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

        # Cumulative measurements
        seeing = (np.mean([self.tractor_fits[chip][-1][TRACTOR_PARAM_PSFSIGMA]
                           for chip in self.chipnames]) * 2.35 * pixsc)
        #skyrate = np.mean([self.sci_acc_strip_skies[chip][-1]
        #                   for chip in self.chipnames]) / self.sci_times[-1]
        skyrate = np.mean([self.acc_strip_skies[chip][-1]
                           for chip in self.chipnames]) / sum(self.dt_walls)
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
            tr = np.cumsum(tr * dsci) / self.sci_times
            trs.append(tr[-1])
        trans = np.mean(trs)

        # Instantaneous measurements
        for i,chip in enumerate(self.chipnames):
            if len(self.sci_times) > 1:
                iskyrate = ((self.sci_acc_strip_skies[chip][-1] - self.sci_acc_strip_skies[chip][-2]) /
                            (self.sci_times[-1] - self.sci_times[-2]))
                print('Count difference:', (self.sci_acc_strip_skies[chip][-1] - self.sci_acc_strip_skies[chip][-2]))
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

def assemble_full_frames(fn, drop_bias_rows=48, fit_exp=True, ps=None):
    F = fitsio.FITS(fn, 'r')
    phdr = F[0].read_header()
    chipnames = []
    imgs = []
    biases = []
    biasimgs = []
    # 4 guide chips
    for i in range(4):
        # two amps per chip (assumed in Left-Right order)
        ampimgs = []
        biasvals = []
        for j in range(2):
            hdu = i*2 + j + 1
            img = F[hdu].read()
            hdr = F[hdu].read_header()

            # Grab the data section
            datasec = hdr['DATASEC'].strip('[]').split(',')
            assert(len(datasec) == 2)
            (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec]]
            # Trim off last row -- it sometimes glitches!
            y1 -= 1
            maxrow = y1
            dataslice = slice(y0-1, y1), slice(x0-1, x1)
            data_x0, data_y0 = x0-1, y0-1

            # Grab the overscan/"bias" section
            biassec = hdr['BIASSEC'].strip('[]').split(',')
            assert(len(biassec) == 2)
            (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in biassec]]
            # Trim off last row -- it sometimes glitches!
            y1 -= 1
            maxrow = max(maxrow, y1)
            biasslice = slice(y0-1, y1), slice(x0-1, x1)
            bias_x0, bias_y0 = x0-1, y0-1

            ampimg  = img[dataslice]
            biasimg = img[biasslice]

            # the ROI images are 2048  x 1080 - with only the bottom 55 rows non-zero,
            # so trim down to 55 x 1080 here.
            # There is some prescan before the image, and some overscan (bias) after.
            # Note, the logic below depends on this having the same origin as the full image
            fitimg = img[:maxrow, :].astype(np.float32)

            bias_level = 0

            if fit_exp:

                def exp_model(eamp, escale, pixel0, shape, ystride, xstride):
                    h,w = shape
                    # split the 2-d exponential into horizontal and vertical factors
                    model = (eamp *
                             np.exp(-(pixel0 + ystride * np.arange(h)) / escale)[:,np.newaxis] *
                             np.exp(-xstride * np.arange(w) / escale)[np.newaxis,:])
                    return model

                # Fit an exponential drop-off to the whole pixel stream
                # (in the data and bias slices), plus constants for the bias and data sections.
                def objective(params, data_img, bias_img, data_pixel0, bias_pixel0,
                              ystride, xstride):
                    data_offset, bias_offset, eamp, escale = params
                    model = exp_model(eamp, escale, data_pixel0, data_img.shape,
                                      ystride, xstride)
                    r = np.sum(np.abs(model + data_offset - data_img))
                    h,w = bias_img.shape
                    model = exp_model(eamp, escale, bias_pixel0, bias_img.shape,
                                      ystride, xstride)
                    r += np.sum(np.abs(model + bias_offset - bias_img))
                    return r

                # Flip the image left-right if it's the right-hand amp, because the readout
                # is in the opposite direction
                rev = (j == 1)

                # Remove the V-shaped pattern before fitting
                h,w = fitimg.shape
                xx = np.arange(w)
                if not rev:
                    xx = xx[::-1]
                v_model = xx * guider_horiz_slope
                #fitimg -= v_model

                # We used to enumerate the pixel number / order of pixel readout, but that array isn't
                # actually required
                # pixelnum = np.arange(h*w).reshape(h,w)
                # if rev:
                #     pixelnum = pixelnum[:, ::-1]
                # dpix = pixelnum[dataslice]
                # bpix = pixelnum[biasslice]
                ystride = w

                if rev:
                    data_pixel0 = data_y0 * w + (w-1 - data_x0)
                    bias_pixel0 = bias_y0 * w + (w-1 - bias_x0)
                    xstride = -1
                else:
                    data_pixel0 = data_y0 * w + data_x0
                    bias_pixel0 = bias_y0 * w + bias_x0
                    xstride = +1

                # print('dpix:', dpix[:3, :3])
                # print('data x0,y0:', data_x0, data_y0)
                # print('data x1:', data_x1)
                # assert(dpix[0,0] == data_pixel0)
                # assert(bpix[0,0] == bias_pixel0)
                # xdir = -1 if rev else +1
                # dh,dw = dpix.shape
                # assert(np.all(dpix[0,0] + pstride * np.arange(dh)[:,np.newaxis] + xdir * np.arange(dw)[np.newaxis,:] == dpix))
                # bh,bw = bpix.shape
                # assert(np.all(bpix[0,0] + pstride * np.arange(bh)[:,np.newaxis] + xdir * np.arange(bw)[np.newaxis,:] == bpix))

                rowmed = np.median(fitimg[biasslice], axis=1)
                med = np.median(rowmed[len(rowmed)//2:])
                #t0 = time.time()
                #x0 = (med, med, rowmed[0]-med, 4800.)
                # Shift the levels of the images so that the offset parameters have values
                # around 1.0 - to make life easier for the optimizer.
                shift = (med - 1)
                fit_data = fitimg[dataslice] - v_model[dataslice[1]][np.newaxis,:] - shift
                fit_bias = fitimg[biasslice] - shift
                x0 = (1., 1., float(rowmed[0] - med), 4800.)
                #print('Initial parameters:', x0)
                r = scipy.optimize.minimize(objective, x0,
                                            args=(fit_data, fit_bias,
                                                  data_pixel0, bias_pixel0, ystride, xstride),
                                            method='Nelder-Mead')
                #t1 = time.time()
                #print('Fitting took %.3f sec' % (t1-t0))
                assert(r.success)
                #print('Fitted parameters:', r.x)
                data_offset, bias_offset, eamp, escale = r.x
                data_offset += shift
                bias_offset += shift
                # Evaluate the models (without the offset terms)
                data_model = exp_model(eamp, escale, data_pixel0,
                                       ampimg.shape, ystride, xstride)
                data_model += v_model[dataslice[1]][np.newaxis,:]
                bias_model = exp_model(eamp, escale, bias_pixel0,
                                       biasimg.shape, ystride, xstride)
                # HACK?  not adding V model to bias... it's small
                bias_level = bias_offset

                # Subtract the exponential decay part of the model from the returned images
                ampimg = ampimg - data_model
                biasimg = biasimg - bias_model
                # For the data image, subtract off the bias level.
                ampimg -= bias_level

                if ps is not None:
                    # Images
                    plt.clf()
                    plt.subplot(2,3,1)
                    mn,mx = np.percentile(fitimg[biasslice].ravel(), [1,99])
                    ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
                    plt.imshow(fitimg[biasslice], **ima)
                    plt.title('Bias image')
                    plt.subplot(2,3,2)
                    plt.imshow(bias_model + bias_offset, **ima)
                    plt.title('Bias model')
                    plt.subplot(2,3,3)
                    bias_resid = fitimg[biasslice] - (bias_model+bias_offset)
                    dmx = np.percentile(np.abs(bias_resid).ravel(), 99)
                    plt.imshow(bias_resid, interpolation='nearest', origin='lower',
                               vmin=-dmx, vmax=+dmx)
                    plt.title('Bias resid')
                    plt.subplot(2,3,4)
                    ima.update(aspect='auto')
                    plt.imshow(fitimg[dataslice], **ima)
                    plt.title('Data image')
                    plt.subplot(2,3,5)
                    plt.imshow(data_model+data_offset, **ima)
                    plt.title('Data model')
                    plt.subplot(2,3,6)
                    data_resid = fitimg[dataslice] - (data_model+data_offset)
                    dmx = np.percentile(np.abs(data_resid).ravel(), 99)
                    plt.imshow(data_resid, interpolation='nearest', origin='lower',
                               aspect='auto', vmin=-dmx, vmax=+dmx)
                    plt.title('Data resid')
                    ps.savefig()
                    # Median plots
                    plt.clf()
                    plt.subplot(2,2,1)
                    off = 2
                    plt.axhline(off, color='k', alpha=0.5)
                    plt.plot(off + np.median(fitimg[biasslice], axis=1) - bias_offset, '-')
                    plt.plot(off + np.median(bias_model, axis=1), '-')
                    plt.yscale('log')
                    plt.title('Bias')
                    plt.subplot(2,2,2)
                    plt.plot(np.median(bias_resid, axis=1), '-')
                    plt.title('Bias resid')
                    plt.subplot(2,2,3)
                    plt.axhline(off, color='k', alpha=0.5)
                    plt.plot(off + np.median(fitimg[dataslice], axis=1) - data_offset, '-')
                    plt.plot(off + np.median(data_model, axis=1), '-')
                    plt.yscale('log')
                    plt.title('Data')
                    plt.subplot(2,2,4)
                    diff = np.median(data_resid, axis=1)
                    plt.plot(diff, '-')
                    mx = max(np.abs(diff))
                    mx = max(1, mx)
                    plt.ylim(-mx,+mx)
                    plt.title('Data resid')
                    ps.savefig()

            else:
                # Not subtracting exponential model.
                # Drop rows at the beginning
                biasimg = biasimg[drop_bias_rows:, :]
                # Sort the resulting 2000 x 50 array so that the 50 columns
                # are sorted for each row
                s = np.sort(biasimg, axis=1)
                # Take the mean of the middle 10 pixels, for each row
                m = np.mean(s[:, 20:30], axis=1)
                # For a scalar level, take the median
                m = np.median(m)
                bias_level = m
                ampimg = ampimg - bias_level

            biasimgs.append(biasimg)
            biasvals.append(bias_level)
            ampimgs.append(ampimg)
        biases.append(biasvals)            
        chipnames.append(hdr['DETPOS'])
        imgs.append(np.hstack(ampimgs))

    return chipnames, imgs, phdr, biases, biasimgs

class DECamGuiderMeasurer(RawMeasurer):

    ZEROPOINT_OFFSET = -2.5 * np.log10(2.24)
    SKY_BRIGHTNESS_CORRECTION = 0.

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
            mn,mx = np.percentile((img-skymod).ravel(), [1,98])
            plt.imshow(img-skymod, interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.colorbar()
            plt.title('Sky-sub image')
            self.ps.savefig()

            plt.clf()
            plt.plot(xx, ramp, '-', label='Row-wise median')
            plt.plot(xx, b + m*xx, '-', label='Ramp model')
            plt.legend()
            self.ps.savefig()

        # Estimate noise on sky-subtracted image.
        sig1a = blanton_sky(img - skymod)

        # Estimate noise on 2x2-binned sky-sub image
        sig2a = blanton_sky(self.bin_image(img - skymod, 2))

        pattern = self.estimate_pattern_noise(img - skymod)

        if self.debug:
            plt.clf()
            plt.imshow(pattern, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('Pattern noise estimate')
            self.ps.savefig()

            plt.clf()
            plt.imshow(img - skymod - pattern, interpolation='nearest', origin='lower',
                       vmin=mn, vmax=mx)
            plt.colorbar()
            plt.title('(Sky+Pattern)-sub image')
            self.ps.savefig()

            print('Trimmed image shape:', img.shape)
            h,w = img.shape
            plt.clf()
            S = 150
            x0,x1, y0,y1 = w//2-S//2, w//2+S//2+1, 0, S
            slc = slice(y0, y1), slice(x0, x1)
            ima = dict(interpolation='nearest', origin='lower', extent=[x0,x1,y0,y1],
                       vmin=-3.*sig1a, vmax=+3.*sig1a)
            plt.subplot(2,2,1)
            plt.imshow(img[slc] - np.median(img[slc]), **ima)
            plt.title('Image')
            plt.subplot(2,2,2)
            im = img[slc] - skymod[slc]
            plt.imshow(im - np.median(im), **ima)
            plt.title('Sky-sub')
            plt.subplot(2,2,3)
            im = pattern[slc]
            plt.imshow(im - np.median(im), **ima)
            plt.title('Pattern')
            plt.subplot(2,2,4)
            im = img[slc] - skymod[slc] - pattern[slc]
            plt.imshow(im - np.median(im), **ima)
            plt.title('Sky- and Pattern-sub')
            self.ps.savefig()

            def fftpow(img, rolloff_fraction=0.04, clipping=20.):
                Ny, Nx = img.shape
                Nh = Ny // 2  # half height of image
                # clip bright pixels
                imgmed = np.median(img.ravel())
                img = np.minimum(img, imgmed + clipping)
                tot = 0.5 * (img + img[:, ::-1])
                tot -= np.mean(tot, axis=1)[:, np.newaxis]
                apod = np.ones(Nh, dtype=np.float32)
                ax = int(round(rolloff_fraction * Nh))
                apod[:ax + 1] = 0.5 + 0.5 * np.cos(np.pi * (1 - np.arange(ax + 1) / ax))
                apod[-(ax + 1):] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(ax + 1) / ax))
                tot[:, :Nh] *= apod[np.newaxis, :]
                tot[:, Nh:] = 0.0
                fimg = pyfftw.interfaces.numpy_fft.rfft(tot, axis=1)
                fpow = fimg.real**2 + fimg.imag**2
                return fpow
            
            plt.clf()
            plt.subplot(2,2,1)
            p = fftpow(img - skymod).T
            mn,mx = np.percentile(p.ravel(), [50,99])
            ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
            plt.title('img-sky (T)')
            plt.imshow(p, **ima)
            plt.subplot(2,2,2)
            plt.title('pattern (T)')
            plt.imshow(fftpow(pattern).T, **ima)
            plt.subplot(2,2,3)
            plt.title('img-sky-pattern (T)')
            plt.imshow(fftpow(img-skymod-pattern).T, **ima)
            plt.suptitle('Fourier power')
            self.ps.savefig()

            plt.clf()
            med = np.median(img - skymod)
            ha = dict(range=(med-5.*sig1a, med+5.*sig1a), bins=50, histtype='step')
            plt.hist((img - skymod).ravel(), label='Vanilla', **ha)
            plt.hist((img - skymod - pattern).ravel(), label='Pattern noise removed', **ha)
            plt.xlabel('Image pixel values (ADU)')
            plt.legend()
            self.ps.savefig()
            

        # Estimate noise on sky- and pattern-subtracted image.
        sig1b = blanton_sky(img - skymod - pattern)

        # Estimate noise on 2x2-binned sky-sub image
        sig2b = blanton_sky(self.bin_image(img - skymod - pattern, 2))

        skymod += pattern

        print('Estimated noise before & after pattern noise removal: %.1f, %.1f' % (sig1a, sig1b))
        print('Estimated noise on 2x2 binned image, before & after pattern noise removal: %.1f, %.1f' % (sig2a, sig2b))
        sig1 = sig1b

        return skymod, sky1, sig1

    def bin_image(self, img, N):
        h,w = img.shape
        sh,sw = h//N, w//N
        sub = np.zeros((sh,sw), np.float32)
        for i in range(N):
            for j in range(N):
                sub += img[i::N, j::N]
        sub /= N**2
        return sub

    # Function to clean pattern noise from the input image
    # from Doug Finkbeiner, 2024-11-03
    def estimate_pattern_noise(self, img, rolloff_fraction=0.04, percentile=95.0,
                               clipping=20.):
        Ny, Nx = img.shape
        Nh = Ny // 2  # half height of image
        # clip bright pixels
        imgmed = np.median(img.ravel())
        img = np.minimum(img, imgmed + clipping)
        # Pattern noise is symmetric about the center, so average the left and right halves
        tot = 0.5 * (img + img[:, ::-1])
        # Subtract mean of each row (could do better detrending here)
        tot -= np.mean(tot, axis=1)[:, np.newaxis]
        # Now apodize the bottom half of the image
        apod = np.ones(Nh, dtype=np.float32)
        ax = int(round(rolloff_fraction * Nh))
        apod[:ax + 1] = 0.5 + 0.5 * np.cos(np.pi * (1 - np.arange(ax + 1) / ax))
        apod[-(ax + 1):] = 0.5 + 0.5 * np.cos(np.pi * (np.arange(ax + 1) / ax))
        # Apply apodization
        tot[:, :Nh] *= apod[np.newaxis, :]
        tot[:, Nh:] = 0.0
        # Row by row FFT using real FFT (for speed improvement)
        fimg = pyfftw.interfaces.numpy_fft.rfft(tot, axis=1)
        # Fourier power
        fpow = fimg.real**2 + fimg.imag**2
        # if self.debug:
        #     plt.clf()
        #     plt.imshow(fpow, interpolation='nearest', origin='lower')
        #     plt.title('Fourier power')
        #     self.ps.savefig()
        # Select the brightest pixels based on the given percentile for threshold
        thresh = np.percentile(fpow[::8,:].ravel(), percentile)
        # Mask for right half of Fourier domain
        fmsk = fpow < thresh
        fmsk[:, :16] = True  # Zero out the DC component
        fmsk[:, -1] = True   # Zero out the Nyquist frequency
        # Set values to zero based on the mask
        fimg[fmsk] = 0
        # Now do the inverse FFT
        pattern = pyfftw.interfaces.numpy_fft.irfft(fimg, n=Nx, axis=1)
        # Replace right half of pattern with left half, flipped
        pattern[:, Nh:] = pattern[:, :Nh][:, ::-1]
        return pattern

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

def compute_shift_all(roi_settings):
    print('roi_settings:', roi_settings)
    setup = roi_settings

    # Defaults from GCS.py
    roi_size = [50,50]
    #add rows below ROI to the readout, to be discarded by roi_masker 
    rows_discard = 5
    #active rows in the CCDs
    nrows_pan = 2048

    dely = roi_size[1]
    delx = roi_size[0]
    rows = nrows_pan
    half_delx = delx/2
    half_dely = dely/2

    x1N = int(setup['roi']['GN1'][0]) - half_delx
    x2N = int(setup['roi']['GN2'][0]) - half_delx
    x1S = int(setup['roi']['GS1'][0]) - half_delx
    x2S = int(setup['roi']['GS2'][0]) - half_delx

    y1N = int(setup['roi']['GN1'][1]) - half_dely - rows_discard
    y2N = int(setup['roi']['GN2'][1]) - half_dely - rows_discard
    y1S = int(setup['roi']['GS1'][1]) - half_dely - rows_discard
    y2S = int(setup['roi']['GS2'][1]) - half_dely - rows_discard

    # max row and column ROI in each CCD
    #mx1N = int(setup['roi']['GN1'][0]) + half_delx
    #mx2N = int(setup['roi']['GN2'][0]) + half_delx
    #mx1S = int(setup['roi']['GS1'][0]) + half_delx
    #mx2S = int(setup['roi']['GS2'][0]) + half_delx
    #my1N = int(setup['roi']['GN1'][1]) + half_dely
    #my2N = int(setup['roi']['GN2'][1]) + half_dely
    #my1S = int(setup['roi']['GS1'][1]) + half_dely
    #my2S = int(setup['roi']['GS2'][1]) + half_dely          

    if(x1N<1): x1N=1
    if(x2N<1): x2N=1
    if(x1S<1): x1S=1
    if(x2S<1): x2S=1
    if(y1N<2): y1N=2
    if(y2N<2): y2N=2
    if(y1S<2): y1S=2
    if(y2S<2): y2S=2               

    row = (y1N, y2N, y1S, y2S)
    maxrow = max(row)
    minrow = min(row)

    xstart = 1               # starting column for the panview image calculation
    ystart = maxrow          # starting row    for the panview image calculation

    #the number of rows the MCB sequencer: skips shift_all; clears after readout after_Rows
    shift_all = max(maxrow - 1,1)   # MCB sequencer loop register

    dely = dely + rows_discard
    after_Rows = rows - minrow - dely + 1

    skip={}
    skip['GN1']= maxrow - y1N
    skip['GN2']= maxrow - y2N
    skip['GS1']= maxrow - y1S
    skip['GS2']= maxrow - y2S

    return dict(shift_all=shift_all, after_rows=after_Rows, skip=skip)

    '''
As an example, I looked for the logs on the image number you mention, exposure number 1278659:

2024-02-28T23:56:21 command: set exptime 600.000000
2024-02-28T23:56:21 imediateReply: DONE
2024-02-28T23:56:21 command: set seqdelay 200

2024-02-28T23:59:22 command: set SHIFT_ALL 1884
2024-02-28T23:59:22 imediateReply: DONE
2024-02-28T23:59:22 command: set AFTER_ROWS 1277
2024-02-28T23:59:22 imediateReply: DONE
2024-02-28T23:59:22 command: memory write 4 none 0x1 loc 0x0490
2024-02-28T23:59:22 imediateReply: DONE
2024-02-28T23:59:22 command: memory write 2 none 0x3 loc 0x0000
2024-02-28T23:59:22 imediateReply: DONE
2024-02-28T23:59:22 command: memory write 4 none 0x3 loc 0x02f4
2024-02-28T23:59:22 imediateReply: DONE
2024-02-28T23:59:22 command: memory write 4 none 0x2 loc 0x015d
'''

    #'guide_ccds' : ['GN1','GN2','GS1','GS2'],
    #"slot":{'GN1':4,  'GN2': 2, 'GS1': 4, 'GS2' : 4},                    # DHE slot for CB controlling the CCDs
    #"skipReg":{'GN1':1,  'GN2': 3, 'GS1': 3, 'GS2' : 2},                 # CB registers for skip rows for the CCDs


if __name__ == '__main__':
    import json
    import pickle

    if False:
        from measure_raw import DECamMeasurer
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('meas')
        # raw image for 1336362
        imgfn = 'c4d_241029_012100_ori.fits.fz'
        meas = DECamMeasurer(imgfn, 'N4', nominal_cal)
        R = meas.run(ps=ps)
        print(R.keys())
        sys.exit(0)

    # 1336362: sky levels
    # ACQ:
    # GS1: 0.302
    # GS2: 0.264
    # GN1: 0.271
    # GN2: 0.253

    # ROI: cumulative:
    # GS1: 0.71
    # GS2: 0.67
    # GN1: 0.68
    # GN2: 0.65

    # RAW image:
    # N4: 1.154, but gains = 4.025 4.077
    # -> 0.284




    metadata = {}
    from obsdb import django_setup
    django_setup(database_filename='decam.sqlite3')
    from obsdb.models import MeasuredCCD
    for m in MeasuredCCD.objects.all():
        metadata[m.expnum] = dict(radec_boresight=(m.rabore, m.decbore),
                                  airmass=m.airmass)
    print('Grabbed metadata for', len(metadata), 'exposures from copilot db')

    procdir = 'data-processed2'
    if not os.path.exists(procdir):
        try:
            os.makedirs(procdir)
        except:
            pass

    astrometry_config_file = '~/data/INDEXES/5200/cfg'

    # expnum = 1336362
    # # 1336360:
    # #kwa = dict(
    # #    radec_boresight=(351.5373, -1.539),
    # #    airmass = 1.15)
    # # 1336361: M438
    # kwa = dict(
    #     radec_boresight=(34.8773, -6.181),
    #     airmass = 1.65)

    # 0.9-second GEXPTIME
    target_gexptime = 0.9
    for expnum in [1336362]:
    #for expnum in range(1336348, 1336450+1):
    #for expnum in range(1336348, 1336436+1):
    # 2.0-second GEXPTIME
    #target_gexptime = 2.0
    #for expnum in range(1336976, 1337017+1):
        print('Expnum', expnum)
        # Maybe no Astrometry.net index files... (not XMM field)
        if expnum in [1336375, 1336397, 1336407, 
                      1336376, 1336413, 1336437, 1336438, 1336439, 1336440, 1336441, 1336442,
                      1337014, 1337015, 1337016, 1337017]:
            print('Skip')
            continue

        roi_settings = json.load(open('/Users/dstn/ibis-data-transfer/guider-acq/roi_settings_%08i.dat' % expnum))
        S = compute_shift_all(roi_settings)
        print(S)

        # (SHIFT_ALL + AFTER_ROWS + 55) * PAR_SHIFT_TIME  + 438 (roi serial read time + sequence delay) + EXPTIME

        shift_all = S['shift_all']
        after_rows = S['after_rows']

        par_shift_time = 160e-6
        pixel_read_time = 4e-6
        seq_delay_time = 200e-3
        gexptime = target_gexptime

        roi_read = 55 * (par_shift_time + 1080 * pixel_read_time)
        print('ROI read time:    %8.3f ms' % (roi_read * 1e3))
        print('Shift time:       %8.3f ms' % (shift_all * par_shift_time * 1e3))
        print('After shift time: %8.3f ms' % (after_rows * par_shift_time * 1e3))
        print('Seq delay time:   %8.3f ms' % (seq_delay_time * 1e3))
        print('exposure time:    %8.3f ms' % (gexptime * 1e3))
        
        total_time = (shift_all + after_rows) * par_shift_time + roi_read + seq_delay_time + gexptime
        print('Total time:       %8.3f ms' % (total_time * 1e3))

        kwa = metadata[expnum]

        acqfn = '~/ibis-data-transfer/guider-acq/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum)
        acqfn = os.path.expanduser(acqfn)
        
        if not os.path.exists(acqfn):
            print('Does not exist:', acqfn)
            continue

        hdr = fitsio.read_header(acqfn)
        if float(hdr['GEXPTIME']) != target_gexptime:
            continue
        print('Filter', hdr['FILTER'])

        statefn = 'state-%i.pickle' % expnum
        if not os.path.exists(statefn):
        #if True:

            from astrometry.util.starutil import ra2hmsstring, dec2dmsstring
            phdr = fitsio.read_header(acqfn)
            fake_header = dict(RA=ra2hmsstring(kwa['radec_boresight'][0], separator=':'),
                            DEC=dec2dmsstring(kwa['radec_boresight'][1], separator=':'),
                            AIRMASS=kwa['airmass'],
                            SCI_UT=phdr['UTSHUT'])

            etc = IbisEtc()
            etc.configure(procdir, astrometry_config_file)
            etc.set_plot_base('acq-%i' % expnum)
            etc.process_guider_acq_image(acqfn, fake_header)

            f = open(statefn,'wb')
            pickle.dump(etc, f)
            f.close()
        else:
            etc = pickle.load(open(statefn, 'rb'))

    #sys.exit(0)

        state2fn = 'state2-%i.pickle' % expnum
        if not os.path.exists(state2fn):

            roi_settings = json.load(open('/Users/dstn/ibis-data-transfer/guider-acq/roi_settings_%08i.dat' % expnum))

            for roi_num in range(1, 100):
            #for roi_num in range(1, 10):
                roi_filename = '~/ibis-data-transfer/guider-sequences/%i/DECam_guider_%i_%08i.fits.gz' % (expnum, expnum, roi_num)
                roi_filename = os.path.expanduser(roi_filename)
                if not os.path.exists(roi_filename):
                    print('Does not exist:', roi_filename)
                    break
                etc.set_plot_base('roi-%03i' % roi_num)
                #etc.set_plot_base(None)
                etc.process_roi_image(roi_settings, roi_num, roi_filename)

            etc.shift_all = shift_all
            etc.after_rows = after_rows

            f = open(state2fn,'wb')
            pickle.dump(etc, f)
            f.close()
        else:
            #continue
            etc = pickle.load(open(state2fn, 'rb'))

    #sys.exit(0)

        if etc.acc_strips is None:
            print('No ROI images')
            continue

        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(os.path.join(procdir, 'roi-summary-%i' % expnum))

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
        mn = 1e6
        for i,chip in enumerate(etc.chipnames):
            m = np.median(etc.acc_strips[chip], axis=0)
            mn = min(mn, min(m))
            plt.plot(m, '-')
        plt.ylim(mn, mn+100)
        plt.title('Accumulated strips')
        ps.savefig()

        plt.clf()
        plt.subplots_adjust(hspace=0)
        mx = np.percentile(np.hstack([x.ravel() for x in etc.acc_strips.values()]), 98)
        for i,chip in enumerate(etc.chipnames):
            plt.subplot(4,1,i+1)
            plt.imshow(etc.sci_acc_strips[chip], interpolation='nearest', origin='lower',
                       aspect='auto', vmin=0, vmax=mx)
            plt.xticks([]); plt.yticks([])
            plt.ylabel(chip)
        plt.suptitle('Science-weighted accumulated strips')
        ps.savefig()

        plt.clf()
        cmap = matplotlib.cm.jet
        for i,chip in enumerate(etc.chipnames):
            plt.subplot(2,2,i+1)
            N = len(etc.all_sci_acc_strips[chip])
            #pcts = np.arange(10, 100, 10)
            lo,hi = 0,0
            for j in range(N):
                l,h = np.percentile(etc.all_sci_acc_strips[chip][j].ravel(), [5,95])
                lo = min(lo, l)
                hi = max(hi, h)
            for j in range(N):
                h,e = np.histogram(etc.all_sci_acc_strips[chip][j].ravel(), range=(lo,hi),
                                   bins=25)
                #plt.hist(etc.all_sci_acc_strips[chip][j].ravel(), range=(lo,hi), bins=25,
                #         histtype='step', color=cmap(j/N))
                plt.plot(e[:-1] + (e[1]-e[0])/2., h, '-', alpha=0.2, color=cmap(j/N))
                plt.axvline(np.median(etc.all_sci_acc_strips[chip][j].ravel()),
                            color=cmap(j/N), alpha=0.1)
            plt.title(chip)
        plt.suptitle('Science-weighted pixel histograms')
        ps.savefig()

        plt.clf()
        plt.subplots_adjust(hspace=0.1)
        for i,chip in enumerate(etc.chipnames):
            for k,side in enumerate(['_L','_R']):
                plt.subplot(2,4,2*i+k+1)
                plt.imshow(etc.all_acc_biases[chip+side][-1], interpolation='nearest',
                           origin='lower')
                plt.title(chip+side)
        plt.suptitle('Bias images')
        ps.savefig()

        plt.clf()
        for i,chip in enumerate(etc.chipnames):
            for k,side in enumerate(['_L','_R']):
                b = etc.all_acc_biases[chip+side][-1]
                plt.plot(np.median(b, axis=0), '-')
                plt.plot(np.median(b, axis=1), '-')
        #plt.yscale('symlog')
        plt.suptitle('Bias image medians')
        ps.savefig()

        plt.clf()
        plt.subplots_adjust(hspace=0.1)
        cmap = matplotlib.cm.jet
        for i,chip in enumerate(etc.chipnames):
            lo,hi = 0,0
            for side in ['_L','_R']:
                N = len(etc.all_acc_biases[chip+side])
                for j in range(N):
                    b = etc.all_acc_biases[chip+side][j]
                    b = b[25:-1,:]
                    l,h = np.percentile(b.ravel(), [0,95])
                    lo = min(lo, l)
                    hi = max(hi, h)
            for k,side in enumerate(['_L','_R']):
                plt.subplot(2,4,2*i+k+1)
                for j in range(N):
                    b = etc.all_acc_biases[chip+side][j]
                    b = b[25:-1,:]
                    h,e = np.histogram(b.ravel(),
                                       range=(lo,hi), bins=25)
                    plt.plot(e[:-1] + (e[1]-e[0])/2., h, '-', alpha=0.2, color=cmap(j/N))
                    #plt.axvline(np.median(etc.all_acc_biases[chip+side][j].ravel()),
                    #            color=cmap(j/N), alpha=0.1)
                plt.title(chip+side)
        plt.suptitle('Bias pixel histograms')
        ps.savefig()

        # plt.clf()
        # for i,chip in enumerate(etc.chipnames):
        #     pix = etc.acc_strips[chip].ravel()
        #     n,b,p = plt.hist(pix)#, **ha)
        #     c = p[0].get_edgecolor()
        #     plt.hist(etc.acc_biases[chip+'_L'].ravel(), histtype='step', linestyle='--')
        #     plt.hist(etc.acc_biases[chip+'_R'].ravel(), histtype='step', linestyle=':')
        # plt.suptitle('Accumulated strips & biases')
        # ps.savefig()

        plt.clf()
        plt.subplots_adjust(hspace=0)
        mx = np.percentile(np.hstack([x.ravel() for x in etc.acc_whole_strips.values()]), 95)
        for i,chip in enumerate(etc.chipnames):
            plt.subplot(4,1,i+1)
            plt.imshow(etc.acc_whole_strips[chip], interpolation='nearest', origin='lower',
                       aspect='auto', vmin=0, vmax=mx)
            plt.xticks([]); plt.yticks([])
            plt.ylabel(chip)
        plt.suptitle('Accumulated strips (full)')
        ps.savefig()

        plt.clf()
        for i,chip in enumerate(etc.chipnames):
            img, ie, mod = etc.acc_rois[chip]
            mn,mx = np.percentile(img, [25,98])
            ima = dict(interpolation='nearest', origin='lower', vmin=0, vmax=mx)
            plt.subplot(4,5, i*5+1)
            plt.imshow(img, **ima)
            plt.title(chip)
            plt.xticks([]); plt.yticks([])
            plt.subplot(4,5, i*5+2)
            plt.imshow(mod, **ima)
            plt.xticks([]); plt.yticks([])
            plt.subplot(4,5, i*5+3)
            plt.imshow((img - mod) * ie, interpolation='nearest', origin='lower',
                       vmin=-10, vmax=+10)
            plt.xticks([]); plt.yticks([])
            plt.subplot(4,5, i*5+4)
            plt.imshow(etc.acc_biases[chip+'_L'], interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])
            plt.subplot(4,5, i*5+5)
            plt.imshow(etc.acc_biases[chip+'_R'], interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])
            
        plt.suptitle('Accumulated ROIs')
        ps.savefig()

        plt.clf()
        for chip in etc.chipnames:
            plt.plot(etc.sci_times, etc.inst_seeing[chip], '.-', label=chip)
        plt.plot(etc.sci_times, etc.cumul_seeing, 'k.-', label='Average')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Accumulated Seeing (arcsec)')
        ps.savefig()

        plt.clf()
        for chip in etc.chipnames:
            plt.plot(etc.sci_times, etc.inst_seeing_2[chip], '.-', label=chip)
        plt.plot(etc.sci_times, etc.cumul_seeing, 'k.-', label='Cumulative')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Instantaneous Seeing (arcsec)')
        ps.savefig()

        plt.clf()
        for chip in etc.chipnames:
            plt.plot(etc.sci_times, etc.inst_transparency[chip], '.-', label=chip)
        plt.plot(etc.sci_times, etc.cumul_transparency, 'k.-', label='Cumulative')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Transparency (instantaneous)')
        ps.savefig()

        # plt.clf()
        # plt.plot(etc.sci_times, etc.cumul_transparency, 'k.-')
        # plt.legend()
        # plt.xlabel('Science exposure time (sec)')
        # plt.ylabel('Transparency (cumulative)')
        # ps.savefig()

        plt.clf()
        for chip in etc.chipnames:
            plt.plot(etc.sci_times, etc.inst_sky[chip], '.-', label=chip)
        plt.plot(etc.sci_times, etc.cumul_sky, 'k.-', label='Cumulative')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Sky brightness (instantaneous) (mag/arcsec^2)')
        ps.savefig()

        # plt.clf()
        # plt.xlabel('Science exposure time (sec)')
        # plt.ylabel('Sky brightness (cumulative) (mag/arcsec^2)')
        # ps.savefig()

        # Copilot terminology:
        # efftime = exptime / expfactor
        fid = nominal_cal.fiducial_exptime(etc.filt)
        ebv = etc.ebv
        exptimes = np.array(etc.sci_times)
        transp = np.vstack([etc.inst_transparency[chip] for chip in etc.chipnames]).mean(axis=0)
        skybr  = np.vstack([etc.inst_sky[chip] for chip in etc.chipnames]).mean(axis=0)
        seeing = np.vstack([etc.inst_seeing_2[chip] for chip in etc.chipnames]).mean(axis=0)
        expfactor_inst = np.zeros(len(exptimes))
        for i in range(len(exptimes)):
            expfactor = exposure_factor(fid, nominal_cal, etc.airmass, ebv,
                                        seeing[i], skybr[i], transp[i])
            expfactor_inst[i] = expfactor
        pixsc = nominal_cal.pixscale
        plt.clf()
        neff_fid = Neff(fid.seeing, pixsc)
        #neff     = Neff(np.array(etc.cumul_seeing), pixsc)
        neff     = Neff(seeing, pixsc)
        efftime_seeing = neff_fid / neff
        efftime_trans = transp**2
        efftime_airmass = 10.**-(0.8 * fid.k_co * (etc.airmass - 1.))
        efftime_sky = 10.**(0.4 * (skybr - fid.skybright))
        efftime_ebv = 10.**(-0.8 * fid.A_co * ebv)
        plt.semilogy(exptimes, efftime_seeing, '-', label='Seeing')
        plt.semilogy(exptimes, efftime_trans, '-', label='Transparency')
        plt.semilogy(exptimes, efftime_sky, '-', label='Sky brightness')
        plt.axhline(efftime_airmass, color='r', label='Airmass')
        plt.axhline(efftime_ebv, color='0.5', label='Dust extinction')
        plt.semilogy(exptimes, 1. / expfactor_inst, 'k-', label='Total')
        plt.xlim(exptimes.min(), exptimes.max())
        plt.legend()
        plt.ylabel('Efftime factor')
        plt.xlabel('Science exposure time (sec)')
        plt.title('Instantaneous')
        ps.savefig()

        expfactor_cumul = np.zeros(len(exptimes))
        for i in range(len(exptimes)):
            expfactor = exposure_factor(fid, nominal_cal, etc.airmass, ebv,
                                        etc.cumul_seeing[i], etc.cumul_sky[i],
                                        etc.cumul_transparency[i])
            expfactor_cumul[i] = expfactor
        plt.clf()
        neff     = Neff(np.array(etc.cumul_seeing), pixsc)
        efftime_seeing = neff_fid / neff
        efftime_trans = np.array(etc.cumul_transparency)**2
        efftime_airmass = 10.**-(0.8 * fid.k_co * (etc.airmass - 1.))
        efftime_sky = 10.**(0.4 * (np.array(etc.cumul_sky) - fid.skybright))
        efftime_ebv = 10.**(-0.8 * fid.A_co * ebv)
        plt.semilogy(exptimes, efftime_seeing, '-', label='Seeing')
        plt.semilogy(exptimes, efftime_trans, '-', label='Transparency')
        plt.semilogy(exptimes, efftime_sky, '-', label='Sky brightness')
        plt.axhline(efftime_airmass, color='r', label='Airmass')
        plt.axhline(efftime_ebv, color='0.5', label='Dust extinction')
        plt.semilogy(exptimes, 1. / expfactor_cumul, 'k-', label='Total')
        plt.xlim(exptimes.min(), exptimes.max())
        plt.legend()
        plt.ylabel('Efftime factor')
        plt.xlabel('Science exposure time (sec)')
        plt.title('Cumulative')
        ps.savefig()

        plt.clf()
        plt.plot(exptimes, exptimes / expfactor_cumul, '.-')
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Effective time (sec)')
        ps.savefig()

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




