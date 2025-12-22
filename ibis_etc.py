import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pylab as plt
import matplotlib
import time

#import pyfftw

import fitsio

from astrometry.util.util import Sip, Tan
from astrometry.util.multiproc import multiproc

#sys.path.insert(0, 'legacypipe/py')
from legacypipe.ps1cat import ps1cat

# Obsbot isn't a proper module
sys.path.insert(0, 'obsbot')
from measure_raw import RawMeasurer
from camera_decam import nominal_cal
from obsbot import exposure_factor, Neff

import scipy.optimize
from legacypipe.ps1cat import ps1_to_decam
from legacypipe.gaiacat import GaiaCatalog, gaia_to_decam

import tractor
import tractor.dense_optimizer

from photutils.aperture import CircularAperture, aperture_photometry

#from tractor.sfd import SFDMap
#print('Reading SFD maps, complaints about END on the following lines are expected.')
#sfd = SFDMap()

import line_profiler

# Remove a V-shape pattern (MAGIC number)
guider_horiz_slope = 0.00168831

TRACTOR_PARAM_PSFSIGMA = 0
TRACTOR_PARAM_SKY = 1
TRACTOR_PARAM_X = 2
TRACTOR_PARAM_Y = 3
TRACTOR_PARAM_FLUX = 4

class IbisEtc(object):

    def __init__(self, assume_photometric=False):
        self.ps = None
        self.debug = False
        self.astrometry_net = False
        self.assume_photometric = assume_photometric
        self.target_efftime = None
        self.prev_times = None
        self.db = None

    def set_db(self, db):
        self.db = db

    def set_plot_base(self, base):
        if base is None:
            self.ps = None
            self.debug = False
        else:
            from astrometry.util.plotutils import PlotSequence
            self.ps = PlotSequence(os.path.join(self.procdir, base))
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
        if astrometry_config_file is not None:
            self.astrometry_net = True

    def clear_after_exposure(self):
        # Clear all the data associated with the current science exposure
        self.sci_datetime = None
        self.acq_datetime = None
        self.acq_exptime = None
        self.roi_exptime = None
        self.expnum = None
        self.radec = None
        #self.ebv = None
        self.filt = None
        self.airmass = None
        self.chipnames = None
        self.imgs = None
        self.chipmeas = None
        self.transparency = None
        #self.transmission = None
        self.nom_zp = None
        self.wcschips = None
        #self.goodchips = None
        # ROIs actually containg a star with flux>1000
        self.starchips = None
        self.flux0 = None
        self.acc_strips = None
        self.acc_biases = None
        self.strip_skies = None
        self.strip_sig1s = None
        self.acc_strip_skies = None
        self.acc_strip_sig1s = None
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
        self.inst_transparency_roi = None
        self.cumul_seeing = None
        self.cumul_sky = None
        self.cumul_transparency = None
        self.efftimes = None
        self.first_roi_datetime = None
        self.rois = None
        self.ran_first_roi = False

    def process_guider_acq_image(self, acqfn, roi_settings, mp=None):
        '''
        * acqfn: string, filename of guider acquisition (first exposure) FITS file
        '''
        self.clear_after_exposure()
        print('Reading', acqfn)
        t0 = time.time()
        chipnames,imgs,phdr,biases,_,_ = assemble_full_frames(acqfn, fit_exp=False)
        t1 = time.time()
        #print('assemble_full_frames took %.3f sec' % (t1-t0))
        # ASSUME that science image starts at the same time as the guider acq image
        self.acq_datetime = datetime_from_header(phdr)
        self.sci_datetime = self.acq_datetime
        self.acq_exptime = float(phdr['GEXPTIME'])
        self.expnum = int(phdr['EXPNUM'])
        self.filt = phdr['FILTER']
        print('Expnum', self.expnum, 'Filter', self.filt)

        if 'RA' in roi_settings and 'dec' in roi_settings and roi_settings['RA'] is not None:
            ra = float(roi_settings['RA'])
            dec = float(roi_settings['dec'])
            self.radec = (ra, dec)
            #self.ebv = sfd.ebv(ra, dec)[0]
        else:
            self.radec = None
            #print('Warning: E(B-V) not known')
            #self.ebv = 0.

        if 'airmass' in roi_settings:
            self.airmass = float(roi_settings['airmass'])
        else:
            print('Warning: airmass not known')
            self.airmass = 1.

        self.chipnames = chipnames
        self.imgs = dict(zip(chipnames, imgs))

        if self.debug:
            plt.clf()

        wcsfns = {}
        imgfns = {}
        self.wcschips = []
        any_img = False
        commands_to_run = []
        for i,(chip,img,biaslr) in enumerate(zip(chipnames, imgs, biases)):
            imgfn = os.path.join(self.procdir, '%s-acq-%s.fits' % (self.expnum, chip))
            imgfns[chip] = imgfn

            if self.astrometry_net:
                # HACK - speed up re-runs
                wcsfn = os.path.join(self.procdir, '%s-acq-%s.wcs' % (self.expnum, chip))
                wcsfns[chip] = wcsfn
                if os.path.exists(wcsfn):
                    self.wcschips.append(chip)
                    continue
                axyfn = os.path.join(self.procdir, '%s-acq-%s.axy' % (self.expnum, chip))
                if os.path.exists(axyfn):
                    print('Exists:', axyfn, '-- assuming it will not solve')
                    continue
            else:
                self.wcschips.append(chip)

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
    
            if self.astrometry_net:
                cmd = ('solve-field ' +
                       '--config %s ' % self.astrometry_config_file +
                       '--scale-low 0.25 --scale-high 0.27 --scale-units app ' +
                       '--solved none --match none --corr none --new-fits none ' +
                       '--no-tweak ' +
                       '--continue ' +
                       '--depth 30 ' +
                       '--crpix-center ' +
                       '--nsigma 6 ')
                if self.debug:
                    cmd = cmd + '--plot-scale 0.5 '
                else:
                    cmd = cmd + '--no-plots '
                if self.radec is not None:
                    ra,dec = self.radec
                    cmd = cmd + '--ra %.4f --dec %.4f --radius 5 ' % (ra, dec)
                cmd = cmd + imgfn
                #cmd = cmd + ' -v --no-delete-temp'
                commands_to_run.append((cmd, wcsfn, chip))

            if self.debug:
                any_img = True
                plt.subplot(2,2, i+1)
                plt.imshow(img, origin='lower', interpolation='nearest', vmin=-30, vmax=+50)
                plt.xticks([]); plt.yticks([])
                plt.title(chip)

        if mp is not None:
            t0 = time.time()
            res = mp.map(run_command, [c for c,_,_ in commands_to_run])
            t1 = time.time()
            print('Got astrometry.net return values:', res)
            print('Runtime: %.3f sec' % (t1-t0))
        else:
            for cmd,_,_ in commands_to_run:
                print(cmd)
                rtn = os.system(cmd)
                print('rtn:', rtn)

        for _,wcsfn,chip in commands_to_run:
            if self.astrometry_net and os.path.exists(wcsfn):
                self.wcschips.append(chip)

        if self.debug and any_img:
            plt.suptitle(acqfn)
            self.ps.savefig()

        flatmap = {}
        flatfn = os.path.join('guideflats-v2', 'flat-%s.fits' % self.filt.lower())
        if os.path.exists(flatfn):
            print('Reading flats from', flatfn)
            chipnames,flats,_,_,_,_ = assemble_full_frames(flatfn,
                                               subtract_bias=False, fit_exp=False)
            #trim_first=False, trim_last=False)
            flatmap.update(dict(list(zip(chipnames, flats))))

        t0 = time.time()
        self.chipmeas = {}
        for chip in chipnames:
            print('Measuring', chip)
            imgfn = imgfns[chip]
            print('Using image file', imgfn)
            # if chip in self.wcschips:
            #     wcs = Sip(wcsfns[chip])
            #     p = ps1cat(ccdwcs=wcs)
            #     stars = p.get_stars()
            # else:
            #     wcs = None

            if not self.astrometry_net and self.radec is not None:
                ra,dec = self.radec
                # Dead-reckon WCS
                chip_offsets = dict(
                    GN1 = (-0.7038, -0.7401),
                    GN2 = (-0.5486, -0.9044),
                    GS1 = (-0.7019,  0.7393),
                    GS2 = (-0.5464,  0.9033),)
                dra,ddec = chip_offsets[chip]
                print('Dead-reckoning WCS')
                wcs = Tan(ra + dra, dec + ddec, 1024.5, 1024.5,
                          0., 7.3e-5, -7.3e-5, 0., 2048.0, 2048.0)
            elif self.astrometry_net and chip in self.wcschips:
                wcs = Sip(wcsfns[chip])
            else:
                wcs = None

            ext = 0
            meas = DECamGuiderMeasurer(imgfn, ext, nominal_cal)
            meas.airmass = self.airmass
            print('Airmass', self.airmass)
            meas.wcs = wcs
            # max astrometric shift, in arcsec (assume astrometry.net solution is pretty good)
            #meas.maxshift = 5.
            meas.maxshift = 30.
    
            kw = dict(ref='gaia')
            if self.debug:
                #kw.update(ps=self.ps)
                #meas.debug = True
                #meas.ps = self.ps
                pass
            kw.update(get_image=True)

            if chip in flatmap:
                # Apply flat-field
                kw.update(flat=flatmap[chip])

            R = meas.run(**kw)
            self.chipmeas[chip] = (meas, R)
        t1 = time.time()
        print('Measuring chips took %.3f sec' % (t1-t0))

        zp0 = None
        kx = None
        dmags = []
        seeings = []
        for chip in self.wcschips:
            meas,R = self.chipmeas[chip]
            if not 'refstars' in R:
                print('No reference stars in chip', chip)
                continue
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
        if len(dmags) == 0:
            print('No matched stars found in any of the guide chips')
            return
        dmags = np.hstack(dmags)
        seeings = np.hstack(seeings)
        seeing = np.median(seeings)
        zpt = -np.median(dmags)
        self.transparency = 10.**(-0.4 * (zp0 - zpt - kx * (self.airmass - 1.)))
        print()
        print('All chips:')
        print('Zeropoint:     %.3f   (with %i stars)' % (zpt, len(dmags)))
        print('Nom Zeropoint: %.3f' % zp0)
        print('Transparency:  %.3f' % self.transparency)
        print('Seeing:        %.2f arcsec' % seeing)
        del dmags
        #self.transmission = 10.**(-0.4 * (zp0 - zpt))
        #print('Transmission:  %.3f' % self.transmission)
        self.nom_zp = zp0

        for chip in self.wcschips:
            meas,R = self.chipmeas[chip]
            ref = R['refstars']
            if meas.use_ps1:
                for i,b in enumerate('grizy'):
                    ref.set('ps1_mag_%s' % b, ref.median[:,i])
                #ref.color = ref.ps1_mag_g - ref.ps1_mag_i
                meas.color_name = 'PS1 g-i'
                ref.base_mag = ref.ps1_mag_g
                meas.base_band_name = 'PS1 g'
                meas.ref_survey_name = 'PS1'
            else:
                #ref.color = ref.bprp_color
                meas.color_name = 'Gaia BP-RP'
                ref.base_mag = ref.phot_g_mean_mag
                meas.base_band_name = 'Gaia G'
                meas.ref_survey_name = 'Gaia'

        if self.debug:
            plt.clf()
            diffs = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
                ref = R['refstars']
                apflux = R['apflux']
                exptime = R['exptime']
                apmag = -2.5 * (np.log10(apflux / exptime) - 9)
                plt.plot(ref.color,  apmag - ref.base_mag, '.', label=chip)
                diffs.append(apmag - (ref.base_mag + R['colorterm']))
            xl,xh = plt.xlim()
            gi = np.linspace(xl, xh, 100)

            from astrometry.util.fits import fits_table
            if meas.use_ps1:
                fakestars = fits_table()
                fakestars.median = np.zeros((len(gi),3))
                fakestars.median[:,0] = gi
                cc = meas.colorterm_ps1_to_observed(fakestars.median, self.filt)
            else:
                fakestars = fits_table()
                # We would use zeros, but gaia_to_decam treats zeros specially!
                mag_offset = 0.001
                fakestars.phot_g_mean_mag = np.zeros(len(gi)) + mag_offset
                fakestars.phot_bp_mean_mag = gi + mag_offset
                fakestars.phot_rp_mean_mag = np.zeros(len(gi)) + mag_offset
                m = gaia_to_decam(fakestars, [self.filt], only_color_term=True)
                cc = m[0] - mag_offset
                del m
                #fakestars.bprp_color = gi
            #cc = meas.get_color_term(fakestars, self.filt)
            #cc = meas.colorterm_ref_to_observed(fakemag, self.filt)
            offset = np.median(np.hstack(diffs))
            plt.plot(gi, offset + cc, '-')
            plt.xlim(xl,xh)
            m = np.mean(cc) + offset
            yl,yh = plt.ylim()
            #plt.ylim(max(yl, m-1), min(yh, m+1))
            plt.legend()
            plt.xlabel('%s (mag)' % meas.color_name)
            plt.ylabel('%s - %s (mag)' % (self.filt, meas.base_band_name))
            self.ps.savefig()

            plt.clf()
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
                ref = R['refstars']
                apflux = R['apflux']
                exptime = R['exptime']
                apmag = -2.5 * np.log10(apflux / exptime)
                plt.plot(ref.color,  apmag - ref.mag, '.', label=chip)
            yl,yh = plt.ylim()
            #plt.ylim(max(yl, m-1), min(yh, m+1))
            plt.axhline(-zpt, color='k', linestyle='--')
            plt.legend()
            plt.xlabel('%s (mag)' % meas.color_name)
            plt.ylabel('%s - ref (%s+color term) (mag)' % (self.filt, meas.base_band_name))
            self.ps.savefig()

            plt.clf()
            rr = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
                ref = R['refstars']
                apflux = R['apflux']
                exptime = R['exptime']
                apmag = -2.5 * np.log10(apflux / exptime)
                apflux_err = R['apflux_err_poisson']
                sn = apflux / apflux_err
                mag_err = np.abs(-2.5 / np.log(10.) / sn)
                #plt.plot(ref.mag, apmag + zpt, '.', label=chip)
                plt.errorbar(ref.mag, apmag + zpt, fmt='.', label=chip,
                             yerr=mag_err)
                rr.append(ref.mag)
            rr = np.hstack(rr)
            mn,mx = np.min(rr), np.max(rr)
            lohi = [mn-0.5, mx+0.5]
            plt.plot(lohi, lohi, 'k-', alpha=0.5)
            plt.axis(lohi*2)
            plt.legend()
            plt.xlabel('%s ref (mag)' % meas.ref_survey_name)
            plt.ylabel('%s (mag)' % self.filt)
            self.ps.savefig()

            plt.clf()
            rr = []
            for chip in self.wcschips:
                meas,R = self.chipmeas[chip]
                ref = R['refstars']
                apflux = R['apflux']
                exptime = R['exptime']
                #apflux_err = R['apflux_err']
                apflux_err = R['apflux_err_poisson']
                sn = apflux / apflux_err
                mag_err = np.abs(-2.5 / np.log(10.) / sn)
                apmag = -2.5 * np.log10(apflux / exptime)
                #plt.plot(ref.mag, apmag + zpt - ref.mag, '.', label=chip)
                plt.errorbar(ref.mag, apmag + zpt - ref.mag, fmt='.', label=chip,
                             yerr=mag_err)
                rr.append(ref.mag)
            rr = np.hstack(rr)
            mn,mx = np.min(rr), np.max(rr)
            lohi = [mn-0.5, mx+0.5]
            plt.xlim(*lohi)
            plt.axhline(0, color='k', alpha=0.1)
            plt.axhline(+0.1, color='k', linestyle='--', alpha=0.1)
            plt.axhline(-0.1, color='k', linestyle='--', alpha=0.1)
            plt.xlim(11, 20)
            plt.ylim(-0.5, +0.5)
            plt.legend(loc='upper left')
            plt.xlabel('%s ref (mag)' % meas.ref_survey_name)
            plt.ylabel('%s - %s ref (mag)' % (self.filt, meas.ref_survey_name))
            plt.title('Expnum %i' % self.expnum)
            self.ps.savefig()

    @line_profiler.profile
    def process_roi_image(self, roi_settings, roi_num, roi_filename,
                          debug=False, mp=None):
        if mp is None:
            mp = multiproc()
        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0.2)

        if self.rois is None:
            self.rois = roi_settings['roi']
            
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
            return

        first_time = False
        #if roi_num == 2:
        if not self.ran_first_roi:
            first_time = True
            roi = roi_settings['roi']
            #goodchips = []
            #flux0 = {}
            #for i,chip in enumerate(self.chipnames):
            #    x,y = roi[chip]
            #    meas,R = self.chipmeas[chip]
            #    # Stars detected in the acq. image
            #    if not 'all_x' in R:
            #        print('Chip', chip, ': no stars')
            #        continue
            #    trim_x0, trim_y0 = R['trim_x0'], R['trim_y0']
            #    det_x = R['all_x'] + trim_x0
            #    det_y = R['all_y'] + trim_y0
            #    # Match to ROI
            #    d = np.hypot(x - det_x, y - det_y)
            #    j = np.argmin(d)
            #    if d[j] > 5:
            #        print('Chip', chip, ': No close match found for ROI star at (x,y) = %.1f, %.1f' % (x,y))
            #        continue
            #    goodchips.append(chip)
            #    print('Matched a star detected in acq image, %.1f pix away' % d[j])
            #    #acqflux = R['all_apflux'][j]
            #    #print('Flux in acq image:', acqflux)
            #    #print('Transmission:', self.transmission)
            #    #if self.transmission is not None:
            #    #    flux0[chip] = acqflux / self.transmission
            #self.goodchips = goodchips
            #self.acq_flux = flux0

            if self.debug:
                plt.clf()
                for i,chip in enumerate(self.chipnames):
                    meas,R = self.chipmeas[chip]
                    x,y = roi[chip]
                    trim_x0, trim_y0 = R['trim_x0'], R['trim_y0']
                    det_x = R['all_x'] + trim_x0
                    det_y = R['all_y'] + trim_y0

                    plt.subplot(3, 4, i+1)
                    img = self.imgs[chip]
                    mn,mx = np.percentile(img.ravel(), [25,98])
                    ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
                    plt.imshow(img, **ima)
                    S = 50
                    ax = plt.axis()
                    refstars = None
                    if 'all_refstars' in R:
                        refstars = R['all_refstars']
                    elif 'refstars' in R:
                        refstars = R['refstars']
                    if refstars is not None:
                        plt.plot(refstars.x, refstars.y, 'o', mec='r', mfc='none')
                    plt.plot([x-S, x+S, x+S, x-S, x-S], [y-S, y-S, y+S, y+S, y-S], 'c-')
                    plt.axis(ax)
                    plt.xticks([]); plt.yticks([])
                    plt.title(chip)
    
                    ix,iy = int(x), int(y)
                    #print('Chip', chip, '- x,y', ix,iy)
                    #print('WCS:', meas.wcs)
                    #print('R keys:', R.keys())
                    H,W = img.shape
                    ix = np.clip(ix, S, W-S)
                    iy = np.clip(iy, S, H-S)
                    x0,y0 = ix-S,iy-S
                    slc = slice(iy-S, iy+S+1), slice(ix-S, ix+S+1)
                    plt.subplot(3, 4, 4 + i+1)
                    plt.imshow(img[slc], **ima)
                    plt.xticks([]); plt.yticks([])
                    plt.subplot(3, 4, 8 + i+1)
                    plt.imshow(img[slc], **ima)
                    ax = plt.axis()
                    if refstars is not None:
                        kw = {}
                        if chip == 'GN2':
                            kw.update(label='ref star')
                        plt.plot(refstars.x-x0, refstars.y-y0, 'o',
                                 mec='r', mfc='none', ms=20, mew=3, **kw)
                    kw = {}
                    if chip == 'GN2':
                        kw.update(label='det star')
                    plt.plot(det_x-x0, det_y-y0, 's',
                             mec='m', mfc='none', ms=10, mew=2, **kw)
                    plt.axis(ax)
                    plt.xticks([]); plt.yticks([])
                    if chip == 'GN2':
                        plt.figlegend(loc='center right')
                self.ps.savefig()

            # Init ROI data structures

            self.starchips = []
            self.roiflux = {}
            self.acc_strips = {}
            self.acc_biases = {}
            self.strip_skies_2  = dict((chip,[]) for chip in self.chipnames)
            self.strip_skies_2.update((chip+'_L',[]) for chip in self.chipnames)
            self.strip_skies_2.update((chip+'_R',[]) for chip in self.chipnames)
            self.strip_skies  = dict((chip,[]) for chip in self.chipnames)
            self.strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_sig1s  = dict((chip,[]) for chip in self.chipnames)
            self.acc_strip_skies  = dict((chip,[]) for chip in self.chipnames)

            self.acc_bias_medians = {}
            self.acc_rowwise_skies = {}

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
            self.inst_transparency_roi = dict((chip,[]) for chip in self.chipnames)
            self.cumul_seeing = []
            self.cumul_sky = []
            self.cumul_transparency = []
            self.efftimes = []
            self.dt_walls = []

            # Find reference star corresponding to each chosen ROI star
            ## FIXME -- we *could* try to match *all* reference stars within the
            # strip, rather than just the rectangle!
            self.roi_ref_stars = {}
            self.roi_star_mags = {}
            for ichip,chip in enumerate(self.chipnames):
                meas,R = self.chipmeas[chip]
                if 'all_refstars' in R:
                    refstars = R['all_refstars']
                elif 'refstars' in R:
                    refstars = R['refstars']
                else:
                    continue
                x,y = roi[chip]
                dists = np.hypot(refstars.x-1 - x, refstars.y-1 - y)
                print('Chip', chip, ': min dist from ROI center to a ref star:', min(dists), 'pix')
                i = np.argmin(dists)
                if dists[i] >= 5:
                    print('Chip', chip,
                          ': Closest reference star is %.1f pix from ROI center -- too far away.' %
                          dists[i])
                    continue
                print('Chip', chip, ': ref star mag is', refstars.mag[i])
                self.roi_ref_stars[chip] = refstars[np.array([i])]
                self.roi_star_mags[chip] = refstars.mag[i]

        if self.debug and False:
            F = fitsio.FITS(roi_filename, 'r')
            self.roi_debug_plots(F)
        kw = {}
        if self.debug and first_time:
            kw.update(ps=self.ps)
        chips,imgs,phdr,biasvals,biasimgs,data_offs = assemble_full_frames(roi_filename, mp=mp, **kw)

        if first_time:
            self.roi_exptime = float(phdr['GEXPTIME'])
        else:
            assert(self.roi_exptime == float(phdr['GEXPTIME']))
        troi = datetime_from_header(phdr)
        if self.first_roi_datetime is None:
            self.first_roi_datetime = troi
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
        #print('dt sci: %.3f, dt wall: %.3f' % (dt_sci, dt_wall))
        self.dt_walls.append(dt_wall)

        # Record things about the full ROI strips...
        for ichip,(chip,img,biases,data_off) in enumerate(zip(chips,imgs,biasvals,data_offs)):
            # The chip-assembly function trims off the top & bottom row of the ROI, so
            # the "img" arrays here are 53 x 2048.

            # The fitted data offset (sky level), averaged between the two amps
            self.strip_skies_2[chip].append(np.mean(data_off))
            self.strip_skies_2[chip+'_L'].append(data_off[0])
            self.strip_skies_2[chip+'_R'].append(data_off[1])

            self.strip_skies[chip].append(np.median(img))
            self.strip_sig1s[chip].append(blanton_sky(img, step=3))
            bl,br = biases

            if first_time:
                self.acc_strips[chip] = img.copy()
                self.acc_biases[chip+'_L'] = (biasimgs[ichip*2  ].copy() - bl)
                self.acc_biases[chip+'_R'] = (biasimgs[ichip*2+1].copy() - br)
                self.acc_bias_medians[chip+'_L'] = []
                self.acc_bias_medians[chip+'_R'] = []
                self.acc_strip_skies[chip+'_L'] = []
                self.acc_strip_skies[chip+'_R'] = []
                self.acc_rowwise_skies[chip+'_L'] = []
                self.acc_rowwise_skies[chip+'_R'] = []

            else:
                # HACK extraneous .copy(), don't think we need them
                self.acc_strips[chip] += img.copy()
                self.acc_biases[chip+'_L'] += (biasimgs[ichip*2  ].copy() - bl)
                self.acc_biases[chip+'_R'] += (biasimgs[ichip*2+1].copy() - br)

            acc = self.acc_strips[chip]
            self.acc_strip_sig1s[chip].append(blanton_sky(acc, step=3))
            self.acc_strip_skies[chip].append(np.median(acc))

            # Left and right half ROI medians
            h,w = acc.shape
            acc_l = acc[:, :w//2]
            acc_r = acc[:, w//2:]
            self.acc_strip_skies[chip+'_L'].append(np.median(acc_l))
            self.acc_strip_skies[chip+'_R'].append(np.median(acc_r))

            acc_bl = self.acc_biases[chip+'_L']
            acc_br = self.acc_biases[chip+'_R']
            self.acc_bias_medians[chip+'_L'].append(np.median(acc_bl))
            self.acc_bias_medians[chip+'_R'].append(np.median(acc_br))

            bias_l_rowmed = np.median(acc_bl, axis=1)
            bias_r_rowmed = np.median(acc_br, axis=1)
            self.acc_rowwise_skies[chip+'_L'].append(
                np.median(acc_l - bias_l_rowmed[:,np.newaxis]))
            self.acc_rowwise_skies[chip+'_R'].append(
                np.median(acc_r - bias_r_rowmed[:,np.newaxis]))

        if self.debug:
            plt.clf()
            plt.subplots_adjust(hspace=0)
            mn,md = np.percentile(np.hstack([img.ravel() for img in imgs]), [1,50])
            for ichip,(chip,img) in enumerate(zip(chips,imgs)):
                plt.subplot(8, 1, ichip+1)
                plt.imshow(img, interpolation='nearest', origin='lower',
                           vmin=mn, vmax=md+(md-mn), aspect='auto')
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            mn,md = np.percentile(np.hstack([img.ravel() for img in self.acc_strips.values()]),
                                  [1,50])
            for ichip,(chip,img) in enumerate(zip(chips,imgs)):
                plt.subplot(8, 1, ichip+5)
                plt.imshow(self.acc_strips[chip], interpolation='nearest', origin='lower',
                           vmin=mn, vmax=md+(md-mn), aspect='auto')
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            plt.suptitle('ROI images and Accumulated')
            self.ps.savefig()

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

        # Trim some pixels off the bottom
        # ... choose 3 to result in 50 x 51 cutouts
        Ntrim = 3
        orig_h = imgs[0].shape[0]
        imgs = [img[Ntrim:, :] for img in imgs]

        roi_xsize = 25
        roi_imgs = {}
        roi = roi_settings['roi']

        for i,(img,chip) in enumerate(zip(imgs, chips)):
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
            #aprad_pix = 15.
            aprad_pix = 10.
            aper = CircularAperture(apxy, aprad_pix)
            p = aperture_photometry(roi_img - sky, aper)
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
            plt.subplots_adjust(hspace=0.25)

        orig_roi_imgs = dict((k,v.copy()) for k,v in roi_imgs.items())
        pixsc = nominal_cal.pixscale

        # If no star was found in first ROI frame, don't try to do tractor fitting
        if first_time:
            tractor_chips = chips
        else:
            tractor_chips = self.starchips

        opt_tractors = []

        for ichip,chip in enumerate(tractor_chips):
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
                # Set PSF parameter bounds: [sigmas, weights]
                # set sigmas for 0.5 to 3" seeing?
                tim.psf.sigmas.lowers = [0.5 / 2.35 / pixsc]
                tim.psf.sigmas.uppers = [3.0 / 2.35 / pixsc]
                # print('PSF parameter bounds:', tim.psf.getLowerBounds(), tim.psf.getUpperBounds())
                tim.sig1 = sig1
                h,w = roi_img.shape
                sky = self.strip_skies[chip][-1]
                flux = np.sum(roi_img) - sky * h*w
                #print('chip', chip, ': initialized flux to', flux)
                flux = max(flux, 100)
                src = tractor.PointSource(tractor.PixPos(roi_starx, roi_stary),
                                          tractor.Flux(flux))
                tr = tractor.Tractor([tim], [src])
                tr.optimizer = tractor.dense_optimizer.ConstrainedDenseOptimizer()
                self.tractors[chip] = tr
                # a second tractor object for instantaneous fitting
                # (fitting each new ROI image)
                # do this so that the source params start from last frame's values
                itr = tr.copy()
                tim = itr.images[0]
                tim.psf.freezeParam('weights')
                # Set PSF parameter bounds: [sigmas, weights]
                # set sigmas for 0.5 to 3" seeing?
                tim.psf.sigmas.lowers = [0.5 / 2.35 / pixsc]
                tim.psf.sigmas.uppers = [3.0 / 2.35 / pixsc]
                itr.optimizer = tractor.dense_optimizer.ConstrainedDenseOptimizer()
                src = itr.catalog[0]
                self.inst_tractors[chip] = itr
            else:
                # we already accumulated the image into tim.data above.
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1

            opt_tractors.append(itr)
            opt_tractors.append(tr)

        tr_params = mp.map(tractor_opt, opt_tractors)

        SEEING_CORR = DECamGuiderMeasurer.SEEING_CORRECTION_FACTOR

        for ichip,chip in enumerate(tractor_chips):
            # unpack results
            iparams = tr_params[ichip*2 + 0]
            params = tr_params[ichip*2 + 1]

            itr = self.inst_tractors[chip]
            tr = self.tractors[chip]
            tim = tr.images[0]

            itr.setParams(iparams)
            tr.setParams(params)

            isee = itr.getParams()[TRACTOR_PARAM_PSFSIGMA] * 2.35 * pixsc
            # HACK -- seeing correction to match copilot
            isee *= SEEING_CORR
            self.inst_seeing_2[chip].append(isee)
            del isee

            s = tim.psf.getParams()[0]
            if s < 0:
                ### Wtf
                tim.psf.setParams([np.abs(s)])
                tr.optimize_loop(**opt_args)

            if self.debug:
                #mx = np.percentile(orig_roi_imgs[chip].ravel(), 99)
                mx = np.max(orig_roi_imgs[chip].ravel())
                plt.subplot(4, 4, 1+ichip)
                plt.imshow(orig_roi_imgs[chip], interpolation='nearest', origin='lower',
                           vmin=-5, vmax=mx)
                plt.xticks([]); plt.yticks([])
                plt.title(chip + ' new ROI')
                mx = np.percentile(orig_roi_imgs[chip].ravel(), 95)
                plt.subplot(4, 4, 5+ichip)
                plt.imshow(orig_roi_imgs[chip], interpolation='nearest', origin='lower',
                           vmin=-5, vmax=mx)
                plt.xticks([]); plt.yticks([])
                plt.title(chip + ' new ROI')
                plt.subplot(4, 4, 9+ichip)
                mx = np.percentile(roi_imgs[chip].ravel(), 95)
                plt.imshow(roi_imgs[chip], interpolation='nearest', origin='lower',
                           vmin=sky-3.*sig1, vmax=mx)
                plt.xticks([]); plt.yticks([])
                plt.title(chip + ' acc ROI')
                plt.subplot(4, 4, 13+ichip)
                mod = tr.getModelImage(0)
                plt.imshow(mod, interpolation='nearest', origin='lower',
                           vmin=sky-3.*sig1, vmax=mx)
                plt.title(chip + ' fit mod')

            self.acc_rois[chip] = (roi_imgs[chip], tim.inverr, tr.getModelImage(0))

            self.tractor_fits[chip].append(tr.getParams())
            #  images.image0.psf.sigmas.param0 = 2.2991302175858706
            #  images.image0.sky.sky = 6.603832232554123
            #  catalog.source0.pos.x = 24.43478222069596
            #  catalog.source0.pos.y = 10.50428514888774
            #  catalog.source0.brightness.Flux = 55101.9692526979

            if first_time:
                p = tr.getParams()
                flux = p[TRACTOR_PARAM_FLUX]
                #if (flux > 1000) and chip in goodchips:
                if flux > 1000:
                    self.starchips.append(chip)
                    self.roiflux[chip] = flux
                else:
                    print('Warning: chip', chip, 'got small tractor flux %.1f - ignoring' % flux)

        if self.debug:
            plt.suptitle('Expnum %i guider frame %i' % (self.expnum, roi_num))
            self.ps.savefig()

        # Cumulative measurements
        # save individual-chip values for db
        csees = {}
        for chip in self.starchips:
            s = self.tractor_fits[chip][-1][TRACTOR_PARAM_PSFSIGMA]
            s *= 2.35 * pixsc
            s *= SEEING_CORR
            csees[chip] = s
        sees = np.array(list(csees.values()))
        sees = clip_outliers(sees, SEEING_MAXRANGE)
        seeing = np.mean(sees)

        skyrate = np.mean([self.acc_strip_skies[chip][-1]
                           for chip in self.chipnames]) / sum(self.dt_walls)
        if self.nom_zp is None:
            self.nom_zp = nominal_cal.zeropoint(self.filt)
        skybr = -2.5 * np.log10(skyrate /pixsc/pixsc) + self.nom_zp
        # HACK -- arbitrary sky correction to match copilot
        skybr += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
        skybr = np.mean(skybr)
        # save individual-chip values for db
        cskies = {}
        for chip in self.chipnames:
            s = self.acc_strip_skies[chip][-1] / sum(self.dt_walls)
            s = -2.5 * np.log10(s /pixsc/pixsc) + self.nom_zp
            s += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
            cskies[chip] = s

        # transparency
        trs = []
        for chip in self.starchips:
            # instantaneous
            flux_now = self.tractor_fits[chip][-1][TRACTOR_PARAM_FLUX]
            if len(self.dt_walls) > 1:
                flux_prev = self.tractor_fits[chip][-2][TRACTOR_PARAM_FLUX]
                # the flux is cumulative, so "now - prev" is the increment
                tr = (flux_now - flux_prev) / self.roiflux[chip]
            else:
                # we just set roiflux = flux_now above
                tr = 1.
            T = self.transparency or 0.0
            self.inst_transparency[chip].append(tr * T)

            # cumulative:
            # assume the time chunks are equal and we're just expecting N
            # increments x the first-frame flux
            tr = flux_now / len(self.dt_walls) / self.roiflux[chip]
            trs.append(tr * T)
        trans = np.mean(trs)

        # transparency on ROI frames alone
        roi_trs = {}
        roi_inst_trs = {}
        S = compute_shift_all(roi_settings)
        shift_all = S['shift_all']
        after_rows = S['after_rows']
        skip_rows = S['skip']
        fid = nominal_cal.fiducial_exptime(self.filt)
        zp0 = self.nom_zp - fid.k_co * (self.airmass - 1.)
        #print('Nominal zeropoint:', self.nom_zp, ', airmass-corrected:', zp0)
        #zp0 = meas.zeropoint_for_exposure(self.filt, ext=meas.ext, exptime=exptime,
        #   primhdr=R['primhdr'])

        for chip in self.starchips:
            if not chip in self.roi_star_mags:
                continue

            ### from Untitled405.ipynb (!) notebook
            t_off = 0.126
            delay_time = 0.2
            row_shift_time = 123.e-6

            chip_offsets = dict(GS1 = 0.020,
                                GS2 = 0.085,
                                GN1 = -0.051,
                                GN2 = 0.101,)

            # "exptime3"
            et = (t_off + self.roi_exptime + delay_time +
                  row_shift_time * (skip_rows[chip] - after_rows))
            mag = self.roi_star_mags[chip]
            refflux = 10.**((zp0 - mag) / 2.5)

            # instantaneous
            apflux = self.roi_apfluxes[chip][-1]
            dmag = 2.5 * np.log10(apflux / (refflux * et))
            dmag += chip_offsets[chip]
            tr = 10.**(dmag / 2.5)
            roi_inst_trs[chip] = tr
            self.inst_transparency_roi[chip].append(tr)

            # cumulative (average flux)
            #apflux = np.sum(self.roi_apfluxes[chip]) / len(self.roi_apfluxes[chip])
            apflux = np.mean(self.roi_apfluxes[chip])
            dmag = 2.5 * np.log10(apflux / (refflux * et))
            dmag += chip_offsets[chip]
            tr = 10.**(dmag / 2.5)
            roi_trs[chip] = tr

        inst_str = []

        if len(roi_trs):
            #print('Instantaneous transparency (from ROIs):', ', '.join(['%.1f' % (tr*100) for tr in roi_inst_trs]), '%')
            #print('Cumulative transparency (from ROIs):', ', '.join(['%.1f' % (tr*100) for tr in roi_trs]), '%')
            roi_trans = np.mean(list(roi_trs.values()))
            #print('Mean transparency (from ROIs): %.1f %%' % (roi_trans*100))
            trans = roi_trans

        if self.assume_photometric:
            #print('--photometric was set, assuming 100% transparency.')
            trans = 1.0

        # Clamp...
        trans = min(trans, 1.05)

        # Instantaneous seeing
        isees = []
        for chip in self.starchips:
            isee = self.tractor_fits[chip][-1][TRACTOR_PARAM_PSFSIGMA] * 2.35 * pixsc
            # HACK -- seeing correction to match copilot
            isee *= SEEING_CORR
            self.inst_seeing[chip].append(isee)
            isees.append(isee)

        # Instantaneous sky
        iskies = []
        for chip in self.chipnames:
            if len(self.dt_walls) > 1:
                iskyrate = ((self.acc_strip_skies[chip][-1] - self.acc_strip_skies[chip][-2]) /
                            dt_wall)
            else:
                iskyrate = self.acc_strip_skies[chip][-1] / dt_wall

            iskybr = -2.5 * np.log10(iskyrate /pixsc/pixsc) + self.nom_zp
            # HACK -- sky correction to match copilot
            iskybr += DECamGuiderMeasurer.SKY_BRIGHTNESS_CORRECTION
            self.inst_sky[chip].append(iskybr)
            iskies.append(iskybr)

        isee = None
        isky = None
        itran = None
        if len(isees):
            isees = clip_outliers(isees, SEEING_MAXRANGE)
            isee = np.mean(isees)
            inst_str.append('see %4.2f"' % isee)
        if len(iskies):
            #print('Sky estimates: [ %s ]' % (', '.join(['%.2f' % s for s in iskies])))
            isky = np.mean(iskies)
            inst_str.append('sky %4.2f' % isky)
        if len(roi_inst_trs):
            itran = np.mean(list(roi_inst_trs.values())) * 100.
            inst_str.append('trans %5.1f %%' % itran)

        fid = nominal_cal.fiducial_exptime(self.filt)
        ### Note -- for IBIS, we have folded the Galactic E(B-V) extinction into
        ### the requested "efftime"s, so here we do *not* include the extinction
        ### factor.
        #ebv = self.ebv
        ebv = 0.
        expfactor = exposure_factor(fid, nominal_cal, self.airmass, ebv,
                                    seeing, skybr, trans)
        efftime = self.sci_times[-1] / expfactor

        ispeed = None
        if self.prev_times is not None:
            (exp_prev, eff_prev) = self.prev_times
            deff_dt = (efftime - eff_prev) / (self.sci_times[-1] - exp_prev)
            ispeed = 100. * deff_dt
            inst_str.append('speed %5.1f %%' % ispeed)
        self.prev_times = (self.sci_times[-1], efftime)

        if self.target_efftime:
            et_target = ' / %5.1f' % self.target_efftime
        else:
            et_target = ''
        print('Exp', self.expnum, '/ %3i,' % roi_num)
        exptime = self.sci_times[-1]
        print('   cumulative:',
              'see %4.2f",' % seeing,
              'sky %4.2f,' % skybr,
              'trans %5.1f %%,' % (100.*trans),
              'exp %5.1f,' % exptime,
              'eff %5.1f%s sec' % (efftime, et_target))
        if len(inst_str):
            #inst = '(inst: ' + ', '.join(inst_str) + ')'
            #inst = '    instantaneous: ' + ', '.join(inst_str)
            #print(inst)
            print('instantaneous: ' + ', '.join(inst_str))

        self.cumul_sky.append(skybr)
        self.cumul_transparency.append(trans)
        self.cumul_seeing.append(seeing)
        self.efftimes.append(efftime)

        # Insert into db
        if self.db:
            print('DB:', self.db)
            with conn.cursor() as cur:
                sql = (
                    'INSERT INTO guider_chip (time,expnum,frame,chip,seeing_cumul,' +
                    'seeing_inst,transparency_cumul,transparency_inst,sky_cumul,sky_inst' +
                    ') values(' +
                    ','.join(['%s'] * 10) +
                    ');')
                data = []
                for chip in self.chipnames:
                    # sql none or 0.0 ?
                    see_inst = None
                    if chip in tractor_chips:
                        see_inst = self.inst_seeing_2[chip][-1]
                    see_cumul = csees.get(chip, None)
                    sky_inst = None
                    if chip in self.inst_sky:
                        sky_inst = self.inst_sky[chip][-1]
                    sky_cumul = cskies.get(chip, None)
                    tran_inst = roi_inst_trs.get(chip, None)
                    tran_cumul = roi_trs.get(chip, None)
                    thisrow = ([troi, self.expnum, roi_num, chip] +
                               [float(x) if x is not None else None for x in
                                [see_cumul, see_inst, tran_cumul, tran_inst,
                                 sky_cumul, sky_inst]])
                    data.append(thisrow)
                #print('Inserting data:', data)
                cur.executemany(sql, data)

                sql = (
                    'INSERT INTO guider_frame (' +
                    'time,expnum,frame,' +
                    'seeing_cumul,seeing_inst,transparency_cumul,transparency_inst,' +
                    'sky_cumul,sky_inst,speed_cumul,speed_inst,' +
                    'airmass,efftime,efftime_target,exptime' +
                    ') values(' +
                    ','.join(['%s'] * 15) +
                    ');')
                cspeed = 1./expfactor
                data = [[troi, self.expnum, roi_num,] +
                        [float(x) if x is not None else None for x in [
                            seeing, isee, trans * 100., itran, skybr, isky, cspeed, ispeed,
                            self.airmass, efftime, self.target_efftime, exptime]]]
                #print('Inserting data:', data)
                cur.executemany(sql, data)
                conn.commit()

        if first_time:
            self.ran_first_roi = True

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
                datasec1 = hdr1['DATASEC'].strip('[]').split(',')
                assert(len(datasec1) == 2)
                (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec1]]
                data = im1[y0-1:y1, x0-1:x1]
                dataimgs.append(data)
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

# for multiprocessing
def tractor_opt(tr):
    # ugh, work around a tractor pickling bug
    tr.model_kwargs = {}
    tr.optimize_loop(shared_params=False)
    return tr.getParams()

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

def process_amp(X):
    j, img, hdr, trim_last, trim_first, fit_exp, drop_bias_rows, subtract_bias = X

    chipname = hdr['DETPOS']
    # Grab the data section
    datasec = hdr['DATASEC'].strip('[]').split(',')
    assert(len(datasec) == 2)
    (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec]]
    if trim_last:
        # Trim off last row -- it sometimes glitches!
        y1 -= 1
    if trim_first:
        # Also trim off the first row -- it's not glitchy in the same way, but does
        # seem to have slightly different statistics than the remaining rows.
        y0 += 1
    maxrow = y1
    dataslice = slice(y0-1, y1), slice(x0-1, x1)
    data_x0, data_y0 = x0-1, y0-1

    # Grab the overscan/"bias" section
    biassec = hdr['BIASSEC'].strip('[]').split(',')
    assert(len(biassec) == 2)
    (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in biassec]]
    if trim_last:
        # Trim off last row -- it sometimes glitches!
        y1 -= 1
    if trim_first:
        # And first row
        y0 += 1
    maxrow = max(maxrow, y1)
    biasslice = slice(y0-1, y1), slice(x0-1, x1)
    bias_x0, bias_y0 = x0-1, y0-1

    ampimg  = img[dataslice]
    biasimg = img[biasslice]

    # the ROI images are 2048  x 1080 - with only the bottom 55 rows non-zero,
    # so trim down to 55 x 1080 here.
    # There is some prescan before the image, and some overscan (bias) after.
    # Note, the logic in the exponential-decay fitting below depends on this
    # having the same origin as the full image
    fitimg = img[:maxrow, :].astype(np.float32)

    bias_level = 0
    data_level = 0

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
            #print('obj %.3f, %.3f %.3f, %.3f -> %.3f' %
            # (data_offset, bias_offset, eamp, escale, r))
            return r

        # Flip the image left-right if it's the right-hand amp, because the readout
        # is in the opposite direction
        rev = (j == 1)

        # Model the V-shaped pattern
        h,w = fitimg.shape
        xx = np.arange(w)
        if not rev:
            xx = xx[::-1]
        v_model = xx * guider_horiz_slope

        ystride = w
        if rev:
            data_pixel0 = data_y0 * w + (w-1 - data_x0)
            bias_pixel0 = bias_y0 * w + (w-1 - bias_x0)
            xstride = -1
        else:
            data_pixel0 = data_y0 * w + data_x0
            bias_pixel0 = bias_y0 * w + bias_x0
            xstride = +1

        #t0 = time.time()
        rowmed = np.median(fitimg[biasslice], axis=1)
        med = np.median(rowmed[len(rowmed)//2:])
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
        if not r.success:
            print('Warning: Fitting result:', r)
        #     from astrometry.util.plotutils import PlotSequence
        #     ps = PlotSequence('fit-fail')
        #assert(r.success)
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
        data_level = data_offset - bias_offset

        # Subtract the exponential decay part of the model from the returned images
        ampimg = ampimg - data_model
        biasimg = biasimg - bias_model
        # For the data image, subtract off the bias level.
        ampimg -= bias_level

        # these are cool, but a lot of plots...
        if False and ps is not None:
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
            plt.suptitle('Chip %s, amp %i' % (chipname, j+1))
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
            # Row plots
            plt.clf()
            plt.subplot(2,2,1)
            off = 2
            plt.axhline(off, color='k', alpha=0.5)
            plt.plot(off - bias_offset + fitimg[biasslice], 'k-', alpha=0.1)
            plt.plot(off + bias_model[:,0], '-', color='orange')
            plt.plot(off + bias_model[:,-1], '-', color='orange')
            plt.yscale('log')
            plt.title('Bias')
            plt.subplot(2,2,2)
            plt.plot(bias_resid, 'k-', alpha=0.1)
            plt.title('Bias resid')
            plt.subplot(2,2,3)
            plt.axhline(off, color='k', alpha=0.5)
            plt.plot(off - data_offset + fitimg[dataslice], 'k-', alpha=0.01)
            plt.plot(off + data_model[:,0], '-', color='orange')
            plt.plot(off + data_model[:,-1], '-', color='orange')
            plt.yscale('log')
            plt.title('Data')
            plt.xlabel('Row number')
            plt.subplot(2,2,4)
            mn,md = np.percentile(data_resid.ravel(), [1,50])
            plt.plot(data_resid, 'k-', alpha=0.01)
            plt.ylim(mn, md+(md-mn))
            plt.title('Data resid')
            ps.savefig()

        #assert(r.success)

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
        if subtract_bias:
            ampimg = ampimg - bias_level

    return (chipname, biasimg.astype(np.float32), bias_level,
            ampimg.astype(np.float32), data_level)


@line_profiler.profile
def assemble_full_frames(fn, drop_bias_rows=48, fit_exp=True, ps=None,
                         subtract_bias=True, trim_first=True, trim_last=True, mp=None):
    F = fitsio.FITS(fn, 'r')
    phdr = F[0].read_header()
    chipnames = []
    imgs = []
    biases = []
    biasimgs = []
    data_offsets = []

    if mp is None:
        mp = multiproc()

    amp_args = []
    # 4 guide chips
    for i in range(4):
        # 2 amps per chip
        for j in range(2):
            hdu = i*2 + j + 1
            img = F[hdu].read()
            hdr = F[hdu].read_header()
            amp_args.append((j, img, hdr, trim_last, trim_first, fit_exp, drop_bias_rows, subtract_bias))
    R = mp.map(process_amp, amp_args)

    # 4 guide chips
    for i in range(4):
        # two amps per chip (assumed in Left-Right order)
        ampimgs = []
        biasvals = []
        data_offs = []
        for j in range(2):
            ri = i*2 + j
            chipname, biasimg, bias_level, ampimg, data_level = R[ri]
            biasimgs.append(biasimg)
            biasvals.append(bias_level)
            data_offs.append(data_level)
            ampimgs.append(ampimg)
        biases.append(biasvals)
        data_offsets.append(data_offs)
        chipnames.append(chipname)
        imgs.append(np.hstack(ampimgs))

    return chipnames, imgs, phdr, biases, biasimgs, data_offsets

def run_command(cmd):
    # used for multi-processing to run a given command line
    print('Running:', cmd)
    rtn = os.system(cmd)
    print('Return value:', rtn)
    return rtn

class DECamGuiderMeasurer(RawMeasurer):

    ZEROPOINT_OFFSET = -2.5 * np.log10(3.23) - 0.30
    SKY_BRIGHTNESS_CORRECTION = -0.588
    SEEING_CORRECTION_FACTOR = 0.916

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
        # Reference catalog
        self.use_ps1 = False

    def remove_sky_gradients(self, img):
        pass

    # def get_reference_stars(self, wcs, band):
    #     if self.use_ps1:
    #         return super().get_reference_stars(wcs, band)
    #     # Gaia
    #     gaia = GaiaCatalog().get_catalog_in_wcs(wcs)
    #     assert(gaia is not None)
    #     assert(len(gaia) > 0)
    #     gaia = GaiaCatalog.catalog_nantozero(gaia)
    #     assert(gaia is not None)
    #     print(len(gaia), 'Gaia stars')
    #     #gaia.about()
    #     return gaia

    # def cut_reference_catalog(self, stars):
    #     if self.use_ps1:
    #         # Cut to stars with good g-i colors
    #         stars.gicolor = stars.median[:,0] - stars.median[:,2]
    #         keep = (stars.gicolor > 0.2) * (stars.gicolor < 2.7)
    #         return keep
    #     keep = (stars.phot_bp_mean_mag != 0) * (stars.phot_rp_mean_mag != 0)
    #     print(sum(keep), 'of', len(stars), 'Gaia stars have BP-RP color')
    #     stars.bprp_color = stars.phot_bp_mean_mag - stars.phot_rp_mean_mag
    #     stars.bprp_color *= keep
    #     # Arjun's color terms are G + colorterm(BP-RP)
    #     stars.mag = stars.phot_g_mean_mag
    #     return keep
    # 
    # def get_color_term(self, stars, band):
    #     if self.use_ps1:
    #         return super().get_color_term(stars, band)
    # 
    #     polys = dict(
    #         M411 = [-0.3464, 1.9527,-2.8314, 3.7463,-1.7361, 0.2621],
    #         M438 = [-0.1806, 0.8371,-0.2328, 0.6813,-0.3504, 0.0527],
    #         M464 = [-0.3263, 1.4027,-1.3349, 1.1068,-0.3669, 0.0424],
    #         M490 = [-0.2287, 1.6287,-2.7733, 2.6698,-1.0101, 0.1330],
    #         M517 = [-0.1937, 1.2866,-2.4744, 2.7437,-1.1472, 0.1623],
    #         )
    #     coeffs = polys[band]
    #     color = stars.bprp_color
    #     colorterm = np.zeros(len(color))
    #     I = np.flatnonzero(stars.bprp_color != 0.0)
    #     for power,coeff in enumerate(coeffs):
    #         colorterm[I] += coeff * color[I]**power
    #     return colorterm
    # 
    # def get_ps1_band(self, band):
    #     print('Band', band)
    #     return ps1cat.ps1band[band]

    # def colorterm_ref_to_observed(self, mags, band):
    #     if self.use_ps1:
    #         print('Using PS1 color term')
    #         return ps1_to_decam(mags, band)
    #     print('Gaia color term')
    #     cc = self.get_color_term(stars, band)
    #     return cc

    def zeropoint_for_exposure(self, band, **kwa):
        zp0 = super().zeropoint_for_exposure(band, **kwa)
        # the superclass applies a GAIN
        if zp0 is None:
            return zp0
        # HACK -- correct for GAIN difference.... shouldn't this be ~3.5 ???
        return zp0 + DECamGuiderMeasurer.ZEROPOINT_OFFSET

    def read_raw(self, F, ext):
        img,hdr = super().read_raw(F, ext)
        print('read_raw: image shape', img.shape)
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
        del xx,hbias

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

        print('Ramp rate: %.3f counts/sec, offset -> %.3f sec' % (m / row_exptime, b/m * row_exptime))
        
        # Remove sky ramp
        skymod += b + (m * x)[:, np.newaxis]

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
            plt.plot(x, ramp, '-', label='Row-wise median')
            plt.plot(x, b + m*x, '-', label='Ramp model')
            plt.legend()
            self.ps.savefig()

        # Estimate noise on sky-subtracted image.
        sig1a = blanton_sky(img - skymod)
        sig1 = sig1a

        # pattern = self.estimate_pattern_noise(img - skymod)
        # # Estimate noise on sky- and pattern-subtracted image.
        # sig1b = blanton_sky(img - skymod - pattern)
        # print('Estimated noise before & after pattern noise removal: %.1f, %.1f' % (sig1a, sig1b))
        # sig1 = sig1b
        # # Estimate noise on 2x2-binned sky-sub image
        # sig2a = blanton_sky(self.bin_image(img - skymod, 2))
        # sig2b = blanton_sky(self.bin_image(img - skymod - pattern, 2))
        # print('Estimated noise on 2x2 binned image, before & after pattern noise removal: %.1f, %.1f' % (sig2a, sig2b))
        # Patter-noise plots
        if False and self.debug:
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
        #skymod += pattern

        #print('Median sky residual:', np.median((img - skymod).ravel()))
        #print('fit offset b:', b)
        
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

import json
import pickle

def run_expnum(args):
    E, metadata, procdir, astrometry_config_file, db = args
    for expnum in [E]:
        print('Expnum', expnum)
        for roi_fn in ['~/ibis-data-transfer/guider-acq/roi_settings_%08i.dat' % expnum,
                       'data-ETC/%ixxx/roi_settings_%08i.dat' % (expnum//1000, expnum)]:
            roi_fn = os.path.expanduser(roi_fn)
            if os.path.exists(roi_fn):
                break

        for acq_fn in ['~/ibis-data-transfer/guider-acq/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum, expnum),
                       'data-ETC/%ixxx/DECam_guider_%i/DECam_guider_%i_00000000.fits.gz' % (expnum//1000, expnum, expnum),
                       ]:
            acq_fn = os.path.expanduser(acq_fn)
            if os.path.exists(acq_fn):
                break
        if not os.path.exists(acq_fn):
            print('Does not exist:', acq_fn)
            continue
        if not os.path.exists(roi_fn):
            print('Does not exist:', roi_fn)
            continue
        
        roi_settings = json.load(open(roi_fn, 'r'))
        S = compute_shift_all(roi_settings)
        # #print(S)
        shift_all = S['shift_all']
        after_rows = S['after_rows']
        skip_rows = S['skip']
        # par_shift_time = 160e-6
        # pixel_read_time = 4e-6
        # seq_delay_time = 200e-3
        # gexptime = target_gexptime
        # roi_read = 55 * (par_shift_time + 1080 * pixel_read_time)
        # print('ROI read time:    %8.3f ms' % (roi_read * 1e3))
        # print('Shift time:       %8.3f ms' % (shift_all * par_shift_time * 1e3))
        # print('After shift time: %8.3f ms' % (after_rows * par_shift_time * 1e3))
        # print('Seq delay time:   %8.3f ms' % (seq_delay_time * 1e3))
        # print('exposure time:    %8.3f ms' % (gexptime * 1e3))
        # total_time = (shift_all + after_rows) * par_shift_time + roi_read + seq_delay_time + gexptime
        # print('Total time:       %8.3f ms' % (total_time * 1e3))

        hdr = fitsio.read_header(acq_fn)
        print('Filter', hdr['FILTER'])

        eprocdir = os.path.join(procdir, '%s' % expnum)
        if not os.path.exists(eprocdir):
            try:
                os.makedirs(eprocdir)
            except:
                pass

        statefn = 'state-%i.pickle' % expnum
        if not os.path.exists(statefn):
        #if True:
            if ('RA' not in roi_settings) or (roi_settings['RA'] == 'null'):
                kwa = metadata[expnum]
                roi_settings['RA'] = kwa['ra']
                roi_settings['dec'] = kwa['dec']
                roi_settings['airmass'] = kwa['airmass']

            etc = IbisEtc()
            etc.configure(eprocdir, astrometry_config_file)
            etc.set_db(db)
            etc.set_plot_base('acq-%i' % expnum)
            #etc.set_plot_base('acq-noflat-%i' % expnum)
            print('Processing acq image', acq_fn)
            etc.process_guider_acq_image(acq_fn, roi_settings)

            f = open(statefn,'wb')
            # not picklable
            etc.db = None
            pickle.dump(etc, f)
            f.close()
        else:
            print('Reading', statefn)
            etc = pickle.load(open(statefn, 'rb'))

        etc.set_db(db)
        # Drop from the state pickle
        for chip in etc.chipnames:
            meas,R = etc.chipmeas[chip]
            del R['image']

        if not hasattr(etc, 'remote_client'):
            etc.remote_client = None
        #if not hasattr(etc, 'stop_efftime'):
        #    etc.stop_efftime = 200.
        if not hasattr(etc, 'ran_first_roi'):
            etc.ran_first_roi = False
        if not hasattr(etc, 'prev_times'):
            etc.prev_times = None

        if 'efftime' in roi_settings:
            try:
                etc.target_efftime = float(roi_settings['efftime'])
            except:
                pass

        state2fn = 'state2-%i.pickle' % expnum
        if not os.path.exists(state2fn):

            for roi_num in range(1, 300):
            #for roi_num in [1,2]+list(range(170, 300)):
            #for roi_num in range(1, 10):

                found = False
                for roi_filename in [
                        ('~/ibis-data-transfer/guider-sequences/%i/DECam_guider_%i_%08i.fits.gz' %
                         (expnum, expnum, roi_num)),
                        ('data-ETC/%ixxx/DECam_guider_%i/DECam_guider_%i_%08i.fits.gz' %
                         (expnum//1000, expnum, expnum, roi_num)),
                        ]:
                    roi_filename = os.path.expanduser(roi_filename)
                    if os.path.exists(roi_filename):
                        found = True
                        break
                if not found:
                    print('Does not exist:', roi_filename)
                    continue
                if roi_num == 2:
                    etc.set_plot_base('roi-first-%i' % (expnum))
                else:
                    etc.set_plot_base('roi-%i-%03i' % (expnum, roi_num))
                #etc.set_plot_base(None)
                try:
                    etc.process_roi_image(roi_settings, roi_num, roi_filename)
                except Exception as e:
                    print('Error reading', roi_filename, ':', str(e))
                    import traceback
                    traceback.print_exc()
                    continue

            etc.shift_all = shift_all
            etc.after_rows = after_rows
            etc.skip_rows = skip_rows

            f = open(state2fn,'wb')
            # not picklable
            etc.db = None
            pickle.dump(etc, f)
            f.close()
        else:
            #continue
            print('Reading', state2fn)
            etc = pickle.load(open(state2fn, 'rb'))

        if etc.acc_strips is None:
            print('No ROI images')
            continue

        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(os.path.join(eprocdir, 'roi-summary-%i' % expnum))

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
        plt.subplots_adjust(hspace=0)
        mx = np.percentile(np.hstack([x.ravel() for x in etc.acc_strips.values()]), 98)
        for i,chip in enumerate(etc.chipnames):
            plt.subplot(4,1,i+1)
            rowmed = np.median((etc.acc_biases[chip+'_L'] + etc.acc_biases[chip+'_R'])/2,
                               axis=1)
            plt.imshow(etc.acc_strips[chip] - rowmed[:, np.newaxis],
                       interpolation='nearest', origin='lower',
                       aspect='auto', vmin=0, vmax=mx)
            plt.xticks([]); plt.yticks([])
            plt.ylabel(chip)
        plt.suptitle('Accumulated strips - row-wise bias')
        ps.savefig()

        flatfn = os.path.join('guideflats', 'flat-%s.fits' % etc.filt.lower())
        if os.path.exists(flatfn):
            chipnames,flats,_,_,_,_ = assemble_full_frames(flatfn,
                                                           subtract_bias=False, fit_exp=False)
            assert(chipnames == etc.chipnames)

            plt.clf()
            plt.subplots_adjust(hspace=0)
            for i,chip in enumerate(etc.chipnames):
                plt.subplot(4,1,i+1)
                x,y = etc.rois[chip]
                ix = int(x)
                iy = int(y)
                ylo = max(0, iy-25)
                plt.imshow(flats[i][ylo:ylo+51, :],
                           interpolation='nearest', origin='lower',
                           aspect='auto', vmin=0.8, vmax=1.2)
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            plt.suptitle('Flats - ROI strips')
            ps.savefig()

            plt.clf()
            plt.subplots_adjust(hspace=0)
            for i,chip in enumerate(etc.chipnames):
                plt.subplot(4,1,i+1)
                S = 150
                x,y = etc.rois[chip]
                ix = int(x)
                iy = int(y)
                xlo = max(0, ix-S)
                ylo = max(0, iy-25)
                plt.imshow(flats[i][ylo:ylo+51, xlo:xlo+2*S],
                           interpolation='nearest', origin='lower',
                           #aspect='auto',
                           vmin=0.8, vmax=1.2)
                plt.xticks([]); plt.yticks([])
                plt.ylabel(chip)
            plt.suptitle('Flats - around ROIs')
            ps.savefig()

        plt.clf()
        plt.subplots_adjust(hspace=0)
        for i,chip in enumerate(etc.chipnames):
            plt.subplot(4,1,i+1)
            S = 150
            x,y = etc.rois[chip]
            ix = int(x)
            iy = int(y)
            xlo = max(0, ix-S)
            plt.imshow(etc.acc_strips[chip][:, xlo:xlo+2*S],
                       interpolation='nearest', origin='lower',
                       #aspect='auto',
                       vmin=0, vmax=mx)
            plt.xticks([]); plt.yticks([])
            plt.ylabel(chip)
        plt.suptitle('Accumulated strips - around ROIs')
        ps.savefig()

        # # Re-fit the V model...
        # def get_v_model(slope, w):
        #     xx = np.arange(w)
        #     v_model = np.abs(xx - (w/2 - 0.5)) * slope
        #     return v_model
        # def objective(params, strip):
        #     offset, slope = params
        #     w = len(strip)
        #     v_model = get_v_model(slope, w)
        #     return np.sum(np.abs(strip - (offset + v_model)))
        # plt.clf()
        # mn = 1e6
        # slopes = []
        # for i,chip in enumerate(etc.chipnames):
        #     m = np.median(etc.acc_strips[chip], axis=0)
        #     mn = min(mn, min(m))
        #     plt.plot(m, '-', alpha=0.2)
        #     r = scipy.optimize.minimize(objective, [0.,0.], args=(m,),
        #                                 method='Nelder-Mead')
        #     offset,slope = r.x
        #     v_model = get_v_model(slope, len(m))
        #     print('Slope', slope)
        #     plt.plot(v_model + offset, '--')
        #     slopes.append(slope)
        # print('Avg slope:', np.mean(slopes) / len(etc.acc_strip_skies[chip]))
        # plt.ylim(mn, mn+100)
        # plt.title('Accumulated strips')
        # ps.savefig()

        dt_wall = np.mean(etc.dt_walls)
        #print('Average dt_wall:', dt_wall)

        plt.clf()
        plt.suptitle('ROI sky estimates (counts/pix/sec)')
        for chip in etc.chipnames:
            plt.subplot(2,1,1)
            plt.plot(etc.strip_skies[chip]/dt_wall, '.-', alpha=0.5, label=chip)
            plt.ylabel('median (cps)')
            plt.subplot(2,1,2)
            plt.plot(etc.strip_skies_2[chip]/dt_wall, '.-', alpha=0.5, label=chip)
            plt.ylabel('exp fit (cps)')
        plt.legend()
        ps.savefig()

        # omit last sample
        dt_final = np.sum(etc.dt_walls[:-1])

        plt.clf()
        plt.suptitle('Accumulated images: medians')
        for chip in etc.chipnames:
            for iside,side in enumerate(['L','R']):
                plt.subplot(2,1,iside+1)
                s = etc.acc_strip_skies[chip+'_'+side]
                plt.plot(s, '-', alpha=0.5, label=chip + ': %.3f cps' % (s[-2]/dt_final))
        for iside,side in enumerate(['L','R']):
            plt.subplot(2,1,iside+1)
            plt.title('Amp: ' + side)
            plt.legend()
        ps.savefig()

        plt.clf()
        plt.suptitle('Accumulated image medians - bias medians')
        for chip in etc.chipnames:
            for iside,side in enumerate(['L','R']):
                plt.subplot(2,1,iside+1)
                s = (np.array(etc.acc_strip_skies[chip+'_'+side]) -
                    np.array(etc.acc_bias_medians[chip+'_'+side]))
                plt.plot(s, '-', alpha=0.5, label=chip + ': %.3f cps' % (s[-2]/dt_final))
        for iside,side in enumerate(['L','R']):
            plt.subplot(2,1,iside+1)
            plt.title('Amp: ' + side)
            plt.legend()
        ps.savefig()

        plt.clf()
        plt.suptitle('Accumulated image - row-wise bias: medians')
        for chip in etc.chipnames:
            for iside,side in enumerate(['L','R']):
                plt.subplot(2,1,iside+1)
                s = etc.acc_rowwise_skies[chip+'_'+side]
                plt.plot(s, '-', alpha=0.5, label=chip + ': %.3f cps' % (s[-2]/dt_final))
        for iside,side in enumerate(['L','R']):
            plt.subplot(2,1,iside+1)
            plt.title('Amp: ' + side)
            plt.legend()
        ps.savefig()

        # Median bias levels (in accumulated bias images)
        plt.clf()
        plt.suptitle('Accumulated bias images: median levels')
        for iside,side in enumerate(['L','R']):
            plt.subplot(2,1,iside+1)
            for chip in etc.chipnames:
                s = etc.acc_bias_medians[chip + '_' + side]
                plt.plot(s, '-', alpha=0.5, label=chip + ': %.3f cps' % (s[-2]/dt_final))
        for iside,side in enumerate(['L','R']):
            plt.subplot(2,1,iside+1)
            plt.title('Amp: ' + side)
            plt.legend()
        ps.savefig()

        # plt.clf()
        # plt.subplots_adjust(hspace=0)
        # mx = np.percentile(np.hstack([x.ravel() for x in etc.acc_strips.values()]), 98)
        # for i,chip in enumerate(etc.chipnames):
        #     plt.subplot(4,1,i+1)
        #     plt.imshow(etc.sci_acc_strips[chip], interpolation='nearest', origin='lower',
        #                aspect='auto', vmin=0, vmax=mx)
        #     plt.xticks([]); plt.yticks([])
        #     plt.ylabel(chip)
        # plt.suptitle('Science-weighted accumulated strips')
        # ps.savefig()

        # plt.clf()
        # cmap = matplotlib.cm.jet
        # for i,chip in enumerate(etc.chipnames):
        #     plt.subplot(2,2,i+1)
        #     N = len(etc.all_sci_acc_strips[chip])
        #     #pcts = np.arange(10, 100, 10)
        #     lo,hi = 0,0
        #     for j in range(N):
        #         l,h = np.percentile(etc.all_sci_acc_strips[chip][j].ravel(), [5,95])
        #         lo = min(lo, l)
        #         hi = max(hi, h)
        #     for j in range(N):
        #         h,e = np.histogram(etc.all_sci_acc_strips[chip][j].ravel(), range=(lo,hi),
        #                            bins=25)
        #         #plt.hist(etc.all_sci_acc_strips[chip][j].ravel(), range=(lo,hi), bins=25,
        #         #         histtype='step', color=cmap(j/N))
        #         plt.plot(e[:-1] + (e[1]-e[0])/2., h, '-', alpha=0.2, color=cmap(j/N))
        #         plt.axvline(np.median(etc.all_sci_acc_strips[chip][j].ravel()),
        #                     color=cmap(j/N), alpha=0.1)
        #     plt.title(chip)
        # plt.suptitle('Science-weighted pixel histograms')
        # ps.savefig()

        # plt.clf()
        # plt.subplots_adjust(hspace=0.1)
        # allpix = np.hstack([v[-1].ravel() for v in etc.all_acc_biases.values()])
        # mn,mx = np.percentile(allpix, [1, 99])
        # for i,chip in enumerate(etc.chipnames):
        #     for k,side in enumerate(['_L','_R']):
        #         plt.subplot(2,4,2*i+k+1)
        #         plt.imshow(etc.all_acc_biases[chip+side][-1], interpolation='nearest',
        #                    origin='lower', vmin=mn, vmax=mx)
        #         plt.title(chip+side)
        # plt.suptitle('Accumulated bias images')
        # ps.savefig()
        # 
        # plt.clf()
        # for i,chip in enumerate(etc.chipnames):
        #     for k,side in enumerate(['_L','_R']):
        #         b = etc.all_acc_biases[chip+side][-1]
        #         plt.plot(np.median(b, axis=0), '-')
        #         plt.plot(np.median(b, axis=1), '-')
        # #plt.yscale('symlog')
        # plt.suptitle('Bias image medians')
        # ps.savefig()

        # plt.clf()
        # plt.subplots_adjust(hspace=0.1)
        # cmap = matplotlib.cm.jet
        # for i,chip in enumerate(etc.chipnames):
        #     lo,hi = 0,0
        #     for side in ['_L','_R']:
        #         N = len(etc.all_acc_biases[chip+side])
        #         for j in range(N):
        #             b = etc.all_acc_biases[chip+side][j]
        #             b = b[25:-1,:]
        #             l,h = np.percentile(b.ravel(), [0,95])
        #             lo = min(lo, l)
        #             hi = max(hi, h)
        #     for k,side in enumerate(['_L','_R']):
        #         plt.subplot(2,4,2*i+k+1)
        #         for j in range(N):
        #             b = etc.all_acc_biases[chip+side][j]
        #             b = b[25:-1,:]
        #             h,e = np.histogram(b.ravel(),
        #                                range=(lo,hi), bins=25)
        #             plt.plot(e[:-1] + (e[1]-e[0])/2., h, '-', alpha=0.2, color=cmap(j/N))
        #             #plt.axvline(np.median(etc.all_acc_biases[chip+side][j].ravel()),
        #             #            color=cmap(j/N), alpha=0.1)
        #         plt.title(chip+side)
        # plt.suptitle('Bias pixel histograms')
        # ps.savefig()

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
        for chip in etc.starchips:
            plt.plot(etc.sci_times[:-1], etc.inst_seeing[chip][:-1], '.-', label=chip)
        plt.plot(etc.sci_times, etc.cumul_seeing, 'k.-', label='Average')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Accumulated Seeing (arcsec)')
        ps.savefig()

        plt.clf()
        for chip in etc.starchips:
            plt.plot(etc.sci_times[:-1], etc.inst_seeing_2[chip][:-1], '.-', label=chip)
        plt.plot(etc.sci_times[:-1], etc.cumul_seeing[:-1], 'k.-', label='Cumulative')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Instantaneous Seeing (arcsec)')
        ps.savefig()

        plt.clf()
        for chip in etc.starchips:
            plt.plot(etc.sci_times[:-1], etc.inst_transparency[chip][:-1], '.-', label=chip)
        for chip in etc.starchips:
            plt.plot(etc.sci_times[:-1], etc.inst_transparency[chip][:-1] / etc.transparency,
                     '.-', lw=2, alpha=0.5)
        plt.plot(etc.sci_times[:-1], etc.cumul_transparency[:-1], 'k.-', label='Cumulative')
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
            plt.plot(etc.sci_times[:-1], etc.inst_sky[chip][:-1], '.-', label=chip)
        plt.plot(etc.sci_times[:-1], etc.cumul_sky[:-1], 'k.-', label='Cumulative')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Sky brightness (instantaneous) (mag/arcsec^2)')
        ps.savefig()

        plt.clf()
        avgsky = 0.
        for chip in etc.chipnames:
            isky = np.array(etc.strip_skies_2[chip]) / np.array(etc.dt_walls)
            avgsky += isky
            plt.plot(etc.sci_times[:-1], isky[:-1], '.-', alpha=0.5, label=chip)
        plt.plot(etc.sci_times[:-1], avgsky[:-1]/len(etc.chipnames), 'k-', label='Average')
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Sky brightness (instantaneous) (counts/pixel/sec)')
        plt.title('sky from fitting')
        ps.savefig()

        plt.clf()
        for chip in etc.chipnames:
            isky = np.array(etc.strip_skies_2[chip]) / np.array(etc.dt_walls)
            pixsc = nominal_cal.pixscale
            skybr = -2.5 * np.log10(isky /pixsc/pixsc) + etc.nom_zp
            plt.plot(etc.sci_times[:-1], skybr[:-1], '.-', label=chip)
        plt.legend()
        plt.xlabel('Science exposure time (sec)')
        plt.ylabel('Sky brightness (instantaneous) (mag/arcsec^2)')
        plt.title('sky from fitting')
        ps.savefig()


        # plt.clf()
        # plt.xlabel('Science exposure time (sec)')
        # plt.ylabel('Sky brightness (cumulative) (mag/arcsec^2)')
        # ps.savefig()

        # Copilot terminology:
        # efftime = exptime / expfactor
        fid = nominal_cal.fiducial_exptime(etc.filt)
        #ebv = etc.ebv
        ebv = 0.
        exptimes = np.array(etc.sci_times)
        if len(etc.starchips) == 0:
            return
        transp = np.vstack([etc.inst_transparency[chip] for chip in etc.starchips]).mean(axis=0)
        skybr  = np.vstack([etc.inst_sky[chip] for chip in etc.chipnames]).mean(axis=0)
        seeing = np.vstack([etc.inst_seeing_2[chip] for chip in etc.starchips]).mean(axis=0)
        expfactor_inst = np.zeros(len(exptimes))
        for i in range(len(exptimes)):
            expfactor = exposure_factor(fid, nominal_cal, etc.airmass, ebv,
                                        seeing[i], skybr[i], transp[i])
            expfactor_inst[i] = expfactor
        pixsc = nominal_cal.pixscale
        plt.clf()
        neff_fid = Neff(fid.seeing, pixsc)
        neff     = Neff(seeing, pixsc)
        efftime_seeing = neff_fid / neff
        efftime_trans = transp**2
        efftime_airmass = 10.**-(0.8 * fid.k_co * (etc.airmass - 1.))
        efftime_sky = 10.**(0.4 * (skybr - fid.skybright))
        efftime_ebv = 10.**(-0.8 * fid.A_co * ebv)
        plt.semilogy(exptimes[:-1], efftime_seeing[:-1], '-', label='Seeing')
        plt.semilogy(exptimes[:-1], efftime_trans[:-1], '-', label='Transparency')
        plt.semilogy(exptimes[:-1], efftime_sky[:-1], '-', label='Sky brightness')
        plt.axhline(efftime_airmass, color='r', label='Airmass')
        plt.axhline(efftime_ebv, color='0.5', label='Dust extinction')
        plt.semilogy(exptimes[:-1], 1. / expfactor_inst[:-1], 'k-', label='Total')
        plt.xlim(exptimes.min(), exptimes.max())
        plt.legend()
        plt.ylabel('Efftime factor')
        plt.xlabel('Science exposure time (sec)')
        plt.title('Instantaneous')
        ps.savefig()

        expfactor_cumul = np.zeros(len(exptimes))
        for i in range(len(exptimes)):
            expfactor = exposure_factor(fid, nominal_cal, etc.airmass, ebv,
                                        etc.cumul_seeing[i],
                                        etc.cumul_sky[i],
                                        etc.cumul_transparency[i])
            expfactor_cumul[i] = expfactor
        plt.clf()
        neff     = Neff(np.array(etc.cumul_seeing), pixsc)
        efftime_seeing = neff_fid / neff
        efftime_trans = np.array(etc.cumul_transparency)**2
        efftime_airmass = 10.**-(0.8 * fid.k_co * (etc.airmass - 1.))
        efftime_sky = 10.**(0.4 * (np.array(etc.cumul_sky) - fid.skybright))
        efftime_ebv = 10.**(-0.8 * fid.A_co * ebv)
        plt.semilogy(exptimes[:-1], efftime_seeing[:-1], '-', label='Seeing')
        plt.semilogy(exptimes[:-1], efftime_trans[:-1], '-', label='Transparency')
        plt.semilogy(exptimes[:-1], efftime_sky[:-1], '-', label='Sky brightness')
        plt.axhline(efftime_airmass, color='r', label='Airmass')
        plt.axhline(efftime_ebv, color='0.5', label='Dust extinction')
        plt.semilogy(exptimes[:-1], 1. / expfactor_cumul[:-1], 'k-', label='Total')
        plt.xlim(exptimes.min(), exptimes.max())
        plt.legend()
        plt.ylabel('Efftime factor')
        plt.xlabel('Science exposure time (sec)')
        plt.title('Cumulative')
        ps.savefig()

        plt.clf()
        plt.plot(exptimes[:-1], (exptimes / expfactor_cumul)[:-1], '.-')
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




def batch_main(db=None):
    global astrometry_config_file
    global procdir

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

    # RAW image:
    # N4: 1.154, but gains = 4.025 4.077
    # -> 0.284

    metadata = {}
    from obsdb import django_setup
    django_setup(database_filename='decam.sqlite3')
    from obsdb.models import MeasuredCCD
    for m in MeasuredCCD.objects.all():
        metadata[m.expnum] = dict(ra=m.rabore, dec=m.decbore, airmass=m.airmass)
    print('Grabbed metadata for', len(metadata), 'exposures from copilot db')
    print('Max expnum:', max(metadata.keys()))

    for fn in ['~/data/INDEXES/5200/cfg',
               '~/cosmo/work/users/dstn/index-5200/cfg']:
        fn = os.path.expanduser(fn)
        if os.path.exists(fn):
            astrometry_config_file = fn
            break

    # expnum = 1336362
    # # 1336360:

    # 0.9-second GEXPTIME
    #target_gexptime = 0.9
    #for expnum in [1336362]:
    #for expnum in range(1336348, 1336450+1):
    #for expnum in range(1336348, 1336436+1):
    # 2.0-second GEXPTIME
    #target_gexptime = 2.0
    #expnums = list(range(1336976, 1337017+1))
    #expnums = list(range(1336984, 1337017+1))

    #expnums = [1336980, 1336983, 1336993, 1337001]
    #expnums = [1336981, 1336982, 1336990, 1337000, 1337006, 1337007]
    #expnums = [1337001]

    # Maybe no Astrometry.net index files... (not XMM field)
    # if expnum in [1336375, 1336397, 1336407, 
   #               1336376, 1336413, 1336437, 1336438, 1336439, 1336440, 1336441, 1336442,
    #               1337014, 1337015, 1337016, 1337017]:

    # 2024-11-23
    #expnums = list(range(1342565, 1342587))
    # 2024-11-24
    #expnums = list(range(1342719, 1342792))
    # 2024-11-26
    #expnums = list(range(1343416, 1343480))
    #expnums = list(range(1343454, 1343480))
    # 2024-11-27
    #expnums = list(range(1343611, 1343665))

    # Repeated pointings
    #expnums = list(range(1336350, 1336356+1))


    # M464, photometric, wide survey, 2024-07-05
    #expnums = list(range(1309133, 1309210))

    # M517, photometric, XMM, 2024-10-29-ish + 2
    # expnums = [1336418, 1336419, 1336420, 1336421, 1336422, 1336423, 1336424,
    #    1336425, 1336426, 1336427, 1336428, 1336429, 1336430, 1336431,
    #    1336432, 1336433, 1336434, 1336435, 1336436, 1336643, 1336644,
    #    1336645, 1336646, 1336647, 1336648, 1336649, 1336650, 1336976,
    #    1336982, 1336984, 1336985, 1336995, 1337004, 1337007, 1337008]
    # All
    #expnums = list(range(1301441, 1342797+1))

    # 2025-02-28 - photometric - M411
    #expnums = list(range(1369580, 1369619))

    #expnums = list(range(1370237, 1370333+1))
    #expnums = [1370243]
    #expnums = [1370244]
    #expnums = [1370577]

    #expnums = [e for e in expnums if e in metadata]

    # 2025-03-27
    # expnums = (list(range(1374551, 1374558+1)) +
    #            list(range(1374561, 1374564)) +
    #            list(range(1374565, 1374603+1)) +
    #            [1374617])

    # 2025-03-28
    #expnums = (#list(range(1374752, 1374816+1)) +
    #           list(range(1374792, 1374816+1)) +
    #           list(range(1374818, 1374823+1)) +
    #           [1374826])

    # 2025-09-23: guider selected a CR in GS2 / GN1?
    #expnums = [1419704, 1419736]
    #expnums = [1419711]


    # 2025-10-15: overexposed
    #expnums = [1426091,]

    # 2025-12-10: lost guiding
    #expnums = [1439986,]

    # 2025-12-14: distorted PSFs
    #expnums = [1441399, 1441400,]

    # 2025-12-15: long exposures
    #expnums = [ 1441555, 1441588]

    expnums = [1441800,]

    mp = multiproc(1)

    for e in expnums:
        run_expnum((e, metadata, procdir, astrometry_config_file, db))

    #mp = multiproc(128)
    #mp.map(run_expnum, [(e, metadata, procdir, astrometry_config_file) for e in expnums])

from obsbot import NewFileWatcher

class EtcFileWatcher(NewFileWatcher):
    def __init__(self, *args, procdir='.', astrometry_config_file=None,
                 remote_client=None, assume_photometric=False, mp=None, db=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mp = mp
        self.procdir = procdir
        self.astrometry_config_file = astrometry_config_file
        self.remote_client = remote_client
        self.assume_photometric = assume_photometric
        self.expnum = None
        self.last_roi = None
        self.etc = None
        self.roi_settings = None
        self.out_of_order = {}
        self.stop_efftime = None
        self.stopped_exposure = False
        self.db = db

    def filter_backlog(self, backlog):
        return []

    # quieter
    def log(self, *args, **kwargs):
        self.debug(*args, **kwargs)

    def get_newest_file(self, newfiles=None):
        if newfiles is None:
            newfiles = self.get_new_files()
        if len(newfiles) == 0:
            return None
        # Take the one with the newest timestamp.
        # --> OLDEST instead here
        latest = None
        newestfile = None
        for fn in newfiles:
            try:
                st = os.stat(fn)
            except OSError as e:
                print('Failed to stat filename', fn, ':', e)
                continue
            t = st.st_mtime
            if latest is None or t < latest:
                newestfile = fn
                latest = t
        return newestfile

    def start_exposure(self, path, expnum):
        dirnm = os.path.dirname(path)
        roi_fn = os.path.join(dirnm, 'roi_settings_%08i.dat' % expnum)
        print('Looking for ROI settings file:', roi_fn)
        self.roi_settings = json.load(open(roi_fn, 'r'))
        print('Got settings:', self.roi_settings)

        # Starting a new exposure!

        eprocdir = os.path.join(self.procdir, '%s' % expnum)
        if not os.path.exists(eprocdir):
            try:
                os.makedirs(eprocdir)
            except:
                pass

        etc = IbisEtc(assume_photometric=self.assume_photometric)
        etc.configure(eprocdir, astrometry_config_file)
        #etc.set_plot_base('acq-%i' % expnum)

        if 'efftime' in self.roi_settings:
            # "None" in some (testing?) files
            try:
                self.stop_efftime = float(self.roi_settings['efftime'])
            except:
                print('Failed to parse "efftime": "%s"' % self.roi_settings['efftime'])
                self.stop_efftime = None
        self.stopped_exposure = False
        etc.target_efftime = self.stop_efftime
        if self.db is not None:
            etc.set_db(self.db)
        etc.process_guider_acq_image(path, self.roi_settings, mp=self.mp)
        self.etc = etc
        self.expnum = expnum
        self.last_roi = 0
        # clear the out-of-order list of previous exposures
        self.out_of_order = dict([((e,r), p) for (e,r),p in self.out_of_order.items()
                                  if e == self.expnum])

        # HACK - testing
        #etc.stop_efftime = 30.

    def process_file(self, path):
        #print('process_file:', path)
        fn = os.path.basename(path)
        dirnm = os.path.dirname(path)
        if fn.startswith('roi_settings'):
            return False
        # DEBUGGING
        if fn == 'quit':
            sys.exit(0)
        if not (fn.startswith('DECam_guider_') and fn.endswith('.fits.gz')):
            print('Unexpect filename pattern:', fn)
            return False
        trim = fn[len('DECam_guider_'):]
        trim = trim[:-len('.fits.gz')]
        #print('Trimmed filename:', trim)
        words = trim.split('_')
        expnum = int(words[0])
        roinum = int(words[1])
        #print('Expnum', expnum, 'ROI num', roinum)
        if roinum == 0:
            print('Starting a new exposure!')
            self.start_exposure(path, expnum)

        elif expnum == self.expnum:
            if roinum != self.last_roi + 1:
                print('The last ROI frame we saw was', self.last_roi, 'but this one is', roinum)
                self.out_of_order[(expnum, roinum)] = path
                return False
            # Catch exceptions and move on the next ROI frame!!
            try:
                self.etc.process_roi_image(self.roi_settings, roinum, path, mp=self.mp)
            except Exception as e:
                print('Error handling', path, ':', str(e))
                import traceback
                traceback.print_exc()
            self.last_roi = roinum

            if self.stop_efftime is not None and self.etc.efftimes is not None and len(self.etc.efftimes) > 1:
                efftime = self.etc.efftimes[-1]
                if efftime > self.stop_efftime:
                    print('Reached the target EFFTIME!')
                    self.stop_exposure()

        elif self.expnum is None:
            # We've started up partway through an exposure.  Try to catch up!
            print('Starting partway through exposure %i.  Trying to catch up!' % expnum)
            self.expnum = expnum
            self.last_roi = -1
            # Add the previous ROI frames to the out-of-order list, if they exist.
            for roi in range(roinum+1):
                fn = os.path.join(dirnm, 'DECam_guider_%i_%08i.fits.gz' % (expnum, roi))
                if os.path.exists(fn):
                    if roi == 0:
                        # First frame -- process it to start this exposure!
                        self.start_exposure(fn, expnum)
                    else:
                        self.out_of_order[(expnum, roi)] = fn
        else:
            print('Unexpected: we were processing expnum', self.expnum,
                  'and new expnum is', expnum, 'and ROI frame number', roinum)
            self.out_of_order[(expnum, roinum)] = path
        return True

    def stop_exposure(self):
        if self.remote_client is not None:
            if self.stopped_exposure:
                print('We already stopped this exposure, don\'t stop again')
            else:
                print('Stopping exposure!')
                #self.remote_client.stopexposure()
                self.remote_client.stoprequested()
                self.stopped_exposure = True

    def heartbeat(self):
        # Check if any of the files in the out-of-order list match the next frame we expect!
        if not len(self.out_of_order):
            return

        print('Checking backlog... on expnum %s, last ROI was %s' %
              (self.expnum, self.last_roi))
        # fixme -- we could just check the dict for the expnum,roi we're looking forw,
        # rather than iterating through it - but it's kind of nice to print out the
        # backlog
        for (e,r),p in self.out_of_order.items():
            if e == self.expnum:
                print('  exp', e, 'roi', r, '->', p)
            if e == self.expnum and r == self.last_roi+1:
                print('Popping an exposure from the backlog')
                try:
                    self.process_file(p)
                    del self.out_of_order[(e,r)]
                    self.run_loop_sleep = False
                except (IOError,OSError) as e:
                    print('Failed to process file: %s (%s)' % (p, str(e)))
                return

SEEING_MAXRANGE = 0.5

# Detect the one-wild-outlier case when we have 3-4 stars.
def clip_outliers(x, maxrange):
    x = np.array(x)
    while len(x) >= 3 and np.max(x) - np.min(x) > maxrange:
        # drop the most outlying one
        i = np.argmax(np.abs(x - np.median(x)))
        #print('Dropping seeing estimate %.2f from [ %s ]' %
        #      (sees[i], ', '.join(['%.2f'%s for s in sees])))
        keep = np.ones(len(x), bool)
        keep[i] = False
        x = x[keep]
        #print('Now: median %.2f, mean %.2f (diff %.2f)' %
        #(np.median(sees), np.mean(sees), np.abs(np.median(sees) - np.mean(sees))))
    return x

if __name__ == '__main__':
    procdir = '/tmp/etc/'
    astrometry_config_file='/data/declsp/astrometry-index-5200/cfg'
    watchdir = '/home3/guider_nfs/ETC/'
    #watchdir = '/tmp/watch'

    #from RemoteClient import RemoteClient
    #rc = RemoteClient()
    from RunRemoteClient import RunRemoteClient
    rc = RunRemoteClient()

    # NERSC
    if 'NERSC_HOST' in os.environ:
        #procdir = 'data-processed2'
        procdir = 'data-processed3'
        astrometry_config_file = os.path.expanduser('~/cosmo/work/users/dstn/index-5200/cfg')
        watchdir = 'temp-data'
        rc = None

    if not os.path.exists(procdir):
        os.makedirs(procdir)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--photometric', action='store_true', help='Assume the night is photometric (transparency = 100%%)')
    parser.add_argument('--no-stop-exposure', default=False, action='store_true',
                        help='Do not actually try to stop exposures')
    parser.add_argument('--watch-dir', default=watchdir, help='Watch this directory for new guider images')
    parser.add_argument('--astrometry', default=astrometry_config_file,
                        help='Astrometry.net config file, default %(default)s')
    parser.add_argument('--batch', default=False, action='store_true', help='Batch mode')
    parser.add_argument('--db', default=False, action='store_true', help='Store results in db')

    opt = parser.parse_args()

    if opt.no_stop_exposure:
        rc = None
    watchdir = opt.watch_dir
    astrometry_config_file = opt.astrometry

    kw = {}
    if opt.db:
        import psycopg2
        conn = psycopg2.connect('dbname=declsp')
        kw.update(db=conn)

    if opt.batch:
        batch_main(**kw)
        sys.exit(0)

    # 8-way multiprocessing (4 guide chips x 2 amps for assemble_full_frames)
    mp = multiproc(8)

    etc = EtcFileWatcher(watchdir,
                         procdir=procdir,
                         astrometry_config_file=astrometry_config_file,
                         remote_client=rc,
                         assume_photometric=opt.photometric,
                         mp=mp,
                         **kw)
    etc.sleeptime = 1.
    etc.run()
