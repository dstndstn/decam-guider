import os
import numpy as np
import pylab as plt
import fitsio
from glob import glob
from datetime import datetime
import photutils

## For astrometry.net -- can use the boresight RA,Dec to speed up the solve...
#> for x in ~/guider/decam-2024-02-28/*_ooi_*; do echo $x; listhead $x | grep EXPNUM; listhead $x | grep 'TEL\(RA\|DEC\)'; done
#/global/homes/d/dstn/guider/decam-2024-02-28/cf9c7fd624758fec840f2997040ea7a8_c4d_240229_013804_ooi_r_ls11.fits.fz
#EXPNUM  =              1278635 / DECam exposure number
#TELRA   = '08:14:54.240'       / [HH:MM:SS] Telescope RA
#TELDEC  = '-04:28:05.902'      / [DD:MM:SS] Telescope DEC
RADECS = {
    1278635: ('08:14:54.240', '-04:28:05.902'),
    1278647: ('09:44:41.996', '-08:44:30.982'),
    }

FILTERS = {
    1278647: 'r',
}    

AIRMASSES = {
    1278647: 1.17,
}

from astrometry.util.fits import fits_table
T = fits_table('obsdb.fits')
T.cut(T.expnum < 1300000)
T.cut(T.extension == 'N4')
_,I = np.unique(T.expnum, return_index=True)
T.cut(I)
RADECS.update(dict([(k,(r,d)) for k,r,d in zip(T.expnum, T.rabore, T.decbore)]))
FILTERS.update(dict(list(zip(T.expnum, T.band))))
AIRMASSES.update(dict(list(zip(T.expnum, T.airmass))))

gaia_color_terms = dict(
    # From Copilot on expnum 1278647 / c4d_240229_021855_ori.fits.fz, eg
    # python copilot.py --save-phot '~/decam-guider/data-2024-02-28/copilot-phot/phot-(EXPNUM)-(EXT).fits' ~/decam-guider/data-2024-02-28/decam/c4d_240229_021855_ori.fits.fz
    #   --ext N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19,S20 --threads 16
    # filter = (gaia_band, bp_color_range, bp_poly)
    r = ('G', (0.7, 2.0), [ 0.20955288, -0.43624861,  0.07023776,  0.05471446]),
    g = ('G', (0.7, 2.0), [ 0.63551392, -1.68398593,  1.98642154, -0.46866005]),
    z = ('G', (0.7, 2.0), [ 0.27618418, -0.39479863, -0.27949407,  0.06148139]),
    N673 = ('G', (0.7, 2.0), [ 0.17001402,  0.12573145, -0.53929908,  0.22958496]),
)

nominal_zeropoints = dict(
    g = 25.15,
    r = 25.30,
    z = 25.00,
    N673 = 24.00,
)
airmass_extinctions = dict(
    g = 0.17,
    r = 0.10,
    z = 0.060,
    ## HACK
    N673 = 0.10,
)


def assemble_full_frames(fn):
    F = fitsio.FITS(fn, 'r')
    chipnames = []
    imgs = []
    # 4 guide chips
    for i in range(4):
        #DATASEC = '[7:1030,1:2048]'    / Data section to display
        #DETPOS  = 'GS1     '
        # two amps per chip
        ampimgs = []
        for j in range(2):
            hdu = i*2 + j + 1
            img = F[hdu].read()
            #print(img.shape)
            hdr = F[hdu].read_header()
            datasec = hdr['DATASEC'].strip('[]').split(',')
            assert(len(datasec) == 2)
            #print(datasec)
            #v = [w.split(':') for w in datasec]
            (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec]]
            #print(v)
            #(x0,x1),(y0,y1) = [int(x) for x in [w.split(':') for w in datasec]]
            ampimgs.append(img[y0-1:y1, x0-1:x1])
        chipnames.append(hdr['DETPOS'])
        imgs.append(np.hstack(ampimgs).astype(np.float32))
        #print(imgs[-1].shape)
    return chipnames, imgs

def plot_from_npz(expnum, fn):
    print('Reading', fn)
    R = np.load(fn, allow_pickle=True)
    #print('Keys:', R.keys())
    chipnames = R['chipnames']
    filt = str(R['filter'])
    ref_mags = R['ref_mags']
    instmags = R['instmags']
    use_for_zpt = R['use_for_zpt']

    if filt != 'r':
        return
    if not ('gaia_mags' in R and 'apfluxes' in R):
        print('No Gaia mags in', fn)
        return

    nframes, nguide = instmags.shape
    apfluxes = R['apfluxes']
    apskies = R['apskies']

    nominal_zpt = nominal_zeropoints[filt]
    k_airmass_ext = airmass_extinctions.get(filt, 0.)
    airmass = AIRMASSES.get(expnum, 1.0)
    if k_airmass_ext == 0 or airmass == 1.0:
        print('WARNING, no airmass correction for expnum', expnum,
              '(airmass %.2f, k_co %.3f)' % (airmass, k_airmass_ext))

    expected_zpt = nominal_zpt - k_airmass_ext * (airmass - 1.0)
    #mags = instmags + expected_zpt
    ref_inst = ref_mags - expected_zpt

    print('Ref mags:', ', '.join(['%.3g' % m for m in ref_mags]))
    print('Median inst_mags + clear-sky zeropoint:', np.median(instmags + expected_zpt, axis=1))
    print('Median mag diffs:',
          np.median(instmags - ref_inst[np.newaxis,:], axis=1))

    transp = 10.**((instmags - ref_inst[np.newaxis, :])/-2.5)
    print('Transparency:', np.median(transp, axis=1))

    gaia_mags = R['gaia_mags']
    g  = gaia_mags[:,0]
    bp = gaia_mags[:,1]
    rp = gaia_mags[:,2]
    print('Gaia mags:', gaia_mags)
    # for j in range(4):
    #     if use_for_zpt[j]:
    #         t = transp[j,:]
    #         plt.scatter([ref_mags[j]]*len(t), t, c=(bp-rp)[j])
    xys = R['guide_xy']
    yy = [x[1] for x in xys]

    if False:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('guider-exp%i' % expnum)
        ps.skipto(7)
    
        plt.clf()
        for j in range(nguide):
            sty = '-'
            ll = ' (pred. mag %.2f)' % (ref_mags[j])
            if not use_for_zpt[j]:
                sty = ':'
                ll = ''
            p = plt.plot(100. * transp[:,j], sty, label=chipnames[j] + ll)
            # + ' (G = %.2f, BP-RP = %.2f)' % bprps[j])
        ax = plt.axis()
        plt.legend()
        plt.ylabel('Guide star implied transparency (%)')
        plt.xlabel('Guider frame number')
        plt.title('Transparency (exp %i: %s)' % (expnum, filt))
        plt.ylim(0, 120)
        ps.savefig()
    
    return (ref_mags, transp, (bp-rp), use_for_zpt, instmags, expnum, g, yy,
            apfluxes, apskies)

def main():
    basedir = 'data-2024-02-28/guider_images/'
    tempdir = 'temp'

    #an_config = '~/cosmo/work/users/dstn/index-5200/cfg'
    an_config = '/Users/dstn/data/INDEXES/5200/cfg'
    
    from astrometry.util.plotutils import PlotSequence

    from astrometry.util.util import Sip
    from legacypipe.gaiacat import GaiaCatalog

    import sys
    
    fns = glob(os.path.join(basedir, 'DECam_guider_*_00000000.fits.gz'))
    fns.sort()
    expnums = []
    for fn in fns:
        words = fn.split('_')
        expnums.append(int(words[-2]))
    print(len(expnums), 'exposures found')

    if True:
        XX = [ [] for i in range(10) ]
        for expnum in expnums:
            npzfn = 'guider-tractor-fit-%i.npz' % expnum
            if os.path.exists(npzfn):
                R = np.load(npzfn, allow_pickle=True)
                # print('Keys:', R.keys())
                # arr = R['arr_0']
                # arr = arr.tolist()
                # print('arr:', type(arr), arr.keys())
                # np.savez(npzfn, **arr)
                X = plot_from_npz(expnum, npzfn)
                if X is None:
                    continue
                for i,xi in enumerate(X):
                    XX[i].append(xi)

        nguide = 4
        rr, tt, bb, ii, ee, gg, yy,apff,apss = [],[],[],[], [], [], [], [], []
        for refs,tr,bprp,use,imags,enum,gmag,y,apf,aps in zip(*XX):
            for j in range(nguide):
                if not use[j]:
                    continue
                print('y', y)
                t = tr[:,j]
                tt.append(t)
                ii.append(imags[:,j])
                rr.append([refs[j]]*len(t))
                bb.append([bprp[j]]*len(t))
                ee.append([enum]*len(t))
                gg.append([gmag[j]]*len(t))
                yy.append([y[j]]*len(t))
                apff.append(apf[:,j])
                apss.append(aps[:,j])
        plt.scatter(np.hstack(rr), np.hstack(tt), c=np.hstack(yy), s=1)
        #plt.scatter(np.hstack(rr), np.hstack(tt), c=np.hstack(bb), s=1)
        #plt.scatter(np.hstack(rr), np.hstack(tt), c=np.hstack(ee), s=1)
        #plt.scatter(np.hstack(gg), np.hstack(tt), c=np.hstack(bb), s=1)
        plt.xlabel('Gaia-predicted mag')
        #plt.xlabel('Gaia G mag')
        plt.ylabel('Transparency')
        cb = plt.colorbar()
        #cb.set_label('Gaia BP-RP')
        cb.set_label('y coordinate')
        #cb.set_label('Expnum')
        #plt.title('r-band data from 2024-02-28')
        plt.title('r-band data from 2024-02-28')
        #plt.scatter(np.hstack(rr), np.hstack(ii), c=np.hstack(bb), s=1)
        plt.savefig('trends.png')
        return

    for expnum in expnums:
        #     #if expnum != 1278687:
        #     #if expnum < 1278723:
        #     #if expnum < 1278712:
        #     #    continue
        #     #1278687
        if expnum != 1278635:
            continue

        npzfn = 'guider-tractor-fit-%i.npz' % expnum
        # If updating NPZ file...
        # old_r = False
        # if os.path.exists(npzfn):
        #     R = np.load(npzfn, allow_pickle=True)
        #     keys = R.keys()
        #     R = dict([(k, R[k]) for k in keys])
        #     old_r = True

        filt = FILTERS.get(expnum, None)
        print('Looked up filter for expnum,', expnum, ':', filt)

        # if filt == 'r':
        #     continue

        ps = PlotSequence('guider-exp%i' % expnum)
        fns = glob(os.path.join(basedir, 'DECam_guider_%i_????????.fits.gz' % expnum))
        fns.sort()
        print('Expnum', expnum, ': found', len(fns), 'guider frames')

        fn = fns[0]
        chipnames,imgs = assemble_full_frames(fn)
        print('Chips:', chipnames)
        nguide = len(chipnames)
        assert(nguide == 4)

        #print([i.shape for i in imgs])
        t = fitsio.read_header(fn)['UTSHUT'] #= '2024-02-29T01:45:30.524' / exp start
        time0 = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")

        for i,img in enumerate(imgs):
            # Remove ampwise medians
            bl = np.median(img[:,:1024])
            br = np.median(img[:,1024:])
            img[:,:1024] -= bl
            img[:,1024:] -= br

        # Remove row-wise median (to remove sky gradient)
        skygrads = []
        for i,img in enumerate(imgs):
            g = np.median(img, axis=1)
            img -= g[:, np.newaxis]
            skygrads.append(g)
            #plt.plot(g)

        # Save to temp image, run astrometry
        wcses = []
        for img,name in zip(imgs, chipnames):
            fn = os.path.join(tempdir, '%i-%s.fits' % (expnum, name))
            fitsio.write(fn, img, clobber=True)

            # FAKE
            radec = RADECS.get(expnum, None)
            rd = ''
            if radec is not None:
                rd = ' --ra %s --dec %s --radius 5' % radec

            cmd = ('solve-field --config %s ' % an_config
                   + '--scale-low 0.25 --scale-high 0.27 --scale-units app '
                   + rd + ' --continue ' 
                   + ' --no-plots ' #' --plot-scale 0.5 # --tag-all')
                   + fn)
            print(cmd)
            wcsfn = fn.replace('.fits', '.wcs')
            if os.path.exists(wcsfn):
                print('Exists:', wcsfn)
            else:
                rtn = os.system(cmd)
                print(rtn)
                assert(rtn == 0)
            assert(os.path.exists(wcsfn))
            wcses.append(Sip(wcsfn))
            
        gaia = GaiaCatalog()
        cats = []
        for wcs,name in zip(wcses, chipnames):
            print('WCS', name, wcs)
            cat = gaia.get_catalog_in_wcs(wcs)
            print(name, len(cat), 'Gaia')
            cats.append(cat)

        ## The "_roi" image headers are useless...
        # The non-roi image header has:
        # ROICOLS = '57 1224 263 1276 50 5' / roi
        fn = fns[1]
        roifn = fn.replace('.fits.gz', '_roi.fits.gz')
        hdr = fitsio.read_header(fn)
        roicols = hdr['ROICOLS']
        roicols = [int(x) for x in roicols.split()]

        # Guess the vertical ROI
        y_star = []
        for i,img in enumerate(imgs):
            strip = img[:, roicols[i]:roicols[i]+50]
            imx = np.argmax(strip.ravel())
            iy,ix = np.unravel_index(imx, strip.shape)
            #print('max in strip:', ix, iy)
            # ASSUME it's in the middle!
            ix = 24
            y_star.append(np.argmax(strip[:,ix]))

        xys = []
        rds = []
        for i in range(nguide):
            x,y = roicols[i], y_star[i]
            x += 24
            xys.append((x,y))
            r,d = wcses[i].pixelxy2radec(x, y)
            rds.append((r,d))

        # Match Gaia catalog to selected guide stars.
        use_for_zpt = np.ones(nguide, bool)
        mags = []
        for j,((x,y),cat) in enumerate(zip(xys, cats)):
            d = np.hypot(x - (cat.x-1), y - (cat.y-1))
            i = np.argmin(d)
            print('Min distance to Gaia star: %.2f pixels' % d[i])
            if d[i] > 5.:
                print('WARNING: failed to find Gaia star near supposed guide star')
                use_for_zpt[j] = False
                m = (0., 0., 0.)
            else:
                m = (cat.phot_g_mean_mag[i], cat.phot_bp_mean_mag[i],
                     cat.phot_rp_mean_mag[i])
                print('Gaia star: G %.2f, BP %.2f, RP %.2f' % m)
            mags.append(m)
        mags = np.array(mags)

        # Apply color terms...
        # (default to G)
        ref_mags = mags[:,0]
        bprps = []
        if filt is not None:
            bp = mags[:,1]
            rp = mags[:,2]
            bprp = bp - rp
            bprps = bprp
            colorterm = gaia_color_terms.get(filt, None)
            if colorterm is not None:
                gaia_band, (c_lo,c_hi), poly = colorterm
                ind = {'G':0, 'BP':1, 'RP':2}.get(gaia_band)
                base = mags[:, ind]
                use_for_zpt[np.logical_or(bp == 0, rp == 0)] = False
                # Drop stars outside the color term range
                use_for_zpt[np.logical_or(bprp < c_lo, bprp > c_hi)] = False
                bprp = np.clip(bprp, c_lo, c_hi)
                colorterm = 0.
                for i,c in enumerate(poly):
                    colorterm = colorterm + c * bprp
                
                ref_mags[use_for_zpt] = (base + colorterm)[use_for_zpt]
                print('Color term corrected mags:',
                      ', '.join(['%.2f'%m for m in ref_mags]))
            else:
                print('No COLOR TERM for filter', filt, '!')
        else:
            print('Unknown filter for expnum', expnum, '!')
            continue

        # if old_r:
        #     R.update(gaia_mags=mags, wcs=wcses, guide_xy=xys, guide_rd=rds)
        #     print('Updating', npzfn)
        #     np.savez(npzfn, R)
        #     continue

        plt.clf()
        for i,img in enumerate(imgs):
            plt.subplot(2,2,i+1)
            mn,mx = np.percentile(img.ravel(), [25,99])
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            plt.title(chipnames[i]);
            if i in [0, 1]:
                plt.xticks([])
        ps.savefig()

        plt.clf()
        for i,(img,(x,y)) in enumerate(zip(imgs, xys)):
            plt.subplot(2,4,i+1)
            mn,mx = np.percentile(img.ravel(), [25,99])
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            ax = plt.axis()
            x0,x1 = x-24, x+24
            y0,y1 = y-24, y+24
            plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'r-')
            plt.title(chipnames[i]);
            plt.axis(ax)
            plt.subplot(2,4,4+i+1)
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            plt.axis([x0,x1,y0,y1])
        ps.savefig()

        ### Question - should we just discard the first ROI frame?  It has different
        # exposure properties than the rest due to the readout timing patterns!
        # [0] is the full-frame image
        # [1] is the first ROI, with funny exposure properties
        # [2] is the firt normal ROI
        startframe = 2
        
        # Tractor fit for the cumulative image after each guider frame
        fns = fns[startframe:]
        nframes = len(fns)
        print('Reading', nframes, 'guider frames...')
        gstack = []
        strip_skies = np.zeros((nframes, nguide), np.float32)
        ut_times = []
        for i,fn in enumerate(fns):
            roifn = fn.replace('.fits.gz', '_roi.fits.gz')
            if not os.path.exists(roifn):
                print('No such file:', roifn)
                break
            F = fitsio.FITS(roifn, 'r')
            t = F[0].read_header()['UTSHUT'] #= '2024-02-29T01:45:30.524' / exp start
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
            ut_times.append(dt)
            gimgs = []
            for j in range(nguide):
                # Trim one pixel off each side of the guider
                # images... otherwise, we sometimes get 'glitches'
                # where the bottom row is hot.
                im = F[j+1].read()
                im = im[1:-1, 1:-1]
                gimgs.append(im)
            gstack.append(gimgs)

            if False and i<3:
                plt.clf()
                for j in range(4):
                    plt.subplot(2,2,j+1)
                    plt.imshow(gimgs[j])
                plt.show()

            F = fitsio.FITS(fn, 'r')
            for j in range(nguide):
                sky = 0
                # 2 amps
                for k in range(2):
                    hdu = j*2 + k + 1
                    im1  = F[hdu].read()
                    hdr1 = F[hdu].read_header()
                    biassec1 = hdr1['BIASSEC'].strip('[]').split(',')
                    assert(len(biassec1) == 2)
                    (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in biassec1]]
                    bias = im1[y0-1:y1, x0-1:x1]
                    # The bottom ~10 rows are much brighter; but use 25 to match data
                    bias = bias[25:, :]
                    bias = np.median(bias)
                    datasec1 = hdr1['DATASEC'].strip('[]').split(',')
                    assert(len(datasec1) == 2)
                    (x0,x1),(y0,y1) = [[int(x) for x in vi] for vi in [w.split(':') for w in datasec1]]
                    data = im1[y0-1:y1, x0-1:x1]
                    # The bottom ~25 (!) rows are significantly larger.
                    data = data[25:, :]
                    sky += np.median(data) - bias
                strip_skies[i, j] = sky/2.

        print('Read', len(gstack), 'guider ROI frames')

        if len(gstack) < nframes:
            nframes = len(gstack)
            strip_skies = strip_skies[:nframes, :]

        from tractor import PixPos, Flux, NCircularGaussianPSF, Image, PointSource, ConstantSky, Tractor

        # Mask out star pixels before computing median (sky)
        h,w = gstack[0][0].shape
        rad = 8
        x = np.arange(w)
        y = np.arange(h)
        starmask = np.hypot(x[np.newaxis,:] - w/2, y[:,np.newaxis] - h/2) > rad

        paramnames = ['PSF-sigma', 'Sky', 'X', 'Y', 'Flux']
        nparams = len(paramnames)
        allskies = np.zeros((nframes, nguide), np.float32)
        allparams = np.zeros((nframes, nguide, nparams), np.float32)
        sig1s = np.zeros((nframes, nguide), np.float32)

        apfluxes = np.zeros((nframes, nguide), np.float32)
        apskies = np.zeros((nframes, nguide), np.float32)
        
        tractors = []
        for j,img in enumerate(gstack[0]):
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            step = 3
            slice1 = (slice(0,-5,step),slice(0,-5,step))
            slice2 = (slice(5,None,step),slice(5,None,step))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
        
            med = np.median(img[starmask])
            tim = Image(img.copy(), inverr=np.ones_like(img)/sig1, psf=NCircularGaussianPSF([2.], [1.]),
                        sky=ConstantSky(med))
            tim.psf.freezeParam('weights')
            tim.sig1 = sig1

            allskies[0, j] = med
            sig1s[0, j] = sig1
            
            h,w = img.shape
            flux = np.sum(img) - med * h*w
            #print('flux estimate:', flux)
            flux = max(flux, 100)
            src = PointSource(PixPos(25, 25), Flux(flux))
            tr = Tractor([tim], [src])
            #print(tr.getParamNames())
            mod0 = tr.getModelImage(0)
            X = tr.optimize_loop()
            #print('opt results:', X)

            #print('Initial fit params:')
            #tr.printThawedParams()

            s = tim.psf.getParams()[0]
            if s < 0:
                ### Wtf
                tim.psf.setParams([np.abs(s)])
                tr.optimize_loop()
            mod1 = tr.getModelImage(0)
            tractors.append(tr)

            allparams[0, j, :] = tr.getParams()

        # Initial tractor fits
        plt.clf()
        for j,tr in enumerate(tractors):
            tim = tr.images[0]
            plt.subplot(4,4, j + 1)
            plt.imshow(tim.data, interpolation='nearest', origin='lower')
            plt.title(chipnames[j])
            plt.subplot(4,4, j + 5)
            #plt.subplot(3,4, j + 1)
            med = np.median(tim.data)
            sig1 = tim.sig1
            ima = dict(interpolation='nearest', origin='lower', vmin=med-3*sig1, vmax=med+10*sig1)
            plt.imshow(tim.data, **ima)
            plt.subplot(4,4, j + 9)
            #plt.subplot(3,4, j + 5)
            mod = tr.getModelImage(0)
            plt.imshow(mod, **ima)
            #plt.subplot(3,4, j + 9)
            plt.subplot(4,4, j + 13)
            plt.imshow((tim.data - mod)/sig1, interpolation='nearest', origin='lower', vmin=-10, vmax=+10)
        plt.suptitle('Initial tractor fits')
        ps.savefig()

        for i,gims in enumerate(gstack):
            if i == 0:
                continue
            for j,(tr,im) in enumerate(zip(tractors,gims)):
                tim = tr.images[0]
                tim.data += im

                allskies[i, j] = np.median(tim.data[starmask])

                # Estimate per-pixel noise via Blanton's "5"-pixel MAD
                step = 3
                slice1 = (slice(0,-5,step),slice(0,-5,step))
                slice2 = (slice(5,None,step),slice(5,None,step))
                mad = np.median(np.abs(tim.data[slice1] - tim.data[slice2]).ravel())
                sig1 = 1.4826 * mad / np.sqrt(2.)
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1
                tr.optimize_loop(shared_params=False)
                #print('Params:', tr.getParams())

                allparams[i, j, :] = tr.getParams()
                sig1s[i, j] = sig1

                # Aperture photometry
                apxy = np.array([[(w-1)/2, (h-1)/2],])
                ap = []
                aprad_pix = 15.
                aper = photutils.aperture.CircularAperture(apxy, aprad_pix)
                p = photutils.aperture_photometry(tim.data, aper)
                apflux = p.field('aperture_sum')
                apflux = float(apflux.data[0])

                #print('Tractor flux:', tr.catalog[0].getBrightness())
                #print('Aperture flux:', apflux)
                #print('Strip_skies shape:', strip_skies.shape)
                #print('strip sky:', ss)
                ss = np.sum(strip_skies[:i+1, j])
                sky = ss * np.pi * aprad_pix**2
                #print('Apflux - ss:', float(apflux) - sky)
                apskies[i, j] = sky
                apfluxes[i, j] = apflux
            print('.', end='')
            sys.stdout.flush()
        print()

        plt.clf()
        for j,tr in enumerate(tractors):
            plt.subplot(3,4, j + 1)
            tim = tr.images[0]
            med = np.median(tim.data)
            sig1 = tim.sig1
            ima = dict(interpolation='nearest', origin='lower', vmin=med-3*sig1, vmax=med+10*sig1)
            plt.title(chipnames[j])
            plt.imshow(tim.data, **ima)
            plt.subplot(3,4, j + 5)
            mod = tr.getModelImage(0)
            plt.imshow(mod, **ima)
            plt.subplot(3,4, j + 9)
            plt.imshow((tim.data - mod)/sig1, interpolation='nearest', origin='lower', vmin=-10, vmax=+10)
        plt.suptitle('All summed guider frames')
        ps.savefig()

        # Plot param(s)
        for ip in [0]:
            pname = paramnames[ip]
            plt.clf()
            for j in range(nguide):
                p = allparams[:, j, ip]
                plt.plot(p, label=chipnames[j])
            plt.title(pname)
            plt.legend()
            yl,yh = plt.ylim()
            plt.ylim(max(yl,0), yh)
            ps.savefig()

        # Check that params are positive
        for ip in [0, 1, 4]:
            pname = paramnames[ip]
            for j in range(nguide):
                # If fit goes to negative params (eg PSF-sigma or flux)
                # (eg expnum 1278689), omit that star from subsequent analysis
                p = allparams[:, j, ip]
                if np.any(p < 0):
                    print('Param', pname, 'goes negative for chip', chipnames[j],
                          '-- marking bad')
                    use_for_zpt[j] = False
            
        plt.clf()
        for j in range(nguide):
            plt.plot(strip_skies[:,j], label=chipnames[j])
        plt.title('Strip median sky')
        plt.legend()
        ps.savefig()

        # plt.clf()
        # for j in range(4):
        #     plt.plot(np.diff(allskies[:, j]), label=chipnames[j])
        # plt.title('Delta median ROI sky')
        # plt.legend()
        # ps.savefig()

        # for i in [1,4]:
        #     plt.clf()
        #     for j in range(4):
        #         plt.plot(np.diff(allparams[j, i, :]), label=chipnames[j])
        #     plt.title('Delta ' + paramnames[i])
        #     plt.legend()
        #     ps.savefig()

        pflux = allparams[:, :, 4]
        #print('Flux param:', pflux.shape)
        #print('pflux:', pflux)
        dflux = np.diff(np.vstack(([0]*nguide, pflux)), axis=0)
        #print('Flux diff:', dflux.shape)
        #print('dflux:', dflux)
        plt.clf()
        print('Ref mags:', ref_mags)
        instmags = -2.5 * np.log10(dflux)
        if np.sum(use_for_zpt):
            #print('computing zpt from', (ref_mags[np.newaxis, :] - instmags)[:, use_for_zpt].shape)
            zpt = np.median((ref_mags[np.newaxis, :] - instmags)[:, use_for_zpt])
        else:
            print('Warning: no stars are good for zeropointing')
            # ... so use 'em all anyway
            zpt = np.median(ref_mags[np.newaxis, :] - instmag)
        print('Fit zeropoint: %.3f' % zpt)
        colors = []
        for j in range(nguide):
            #plt.axhline(gmags[j], color='k', label=('Gaia G' if j==3 else None))
            p = plt.plot(zpt + instmags[:, j], label=chipnames[j] + ' (BP-RP = %.2f)' % bprps[j])
            c = p[0].get_color()
            colors.append(c)
            if use_for_zpt[j]:
                plt.axhline(ref_mags[j], color=c, linestyle='--')
        ax = plt.axis()
        # don't let these ones set the axis limits
        for j in range(nguide):
            if not use_for_zpt[j]:
                plt.axhline(ref_mags[j], color=colors[j], linestyle=':')
        plt.axhline(0, color='0.3', linestyle='--', label='Gaia pred')
        plt.axis(ax)
        plt.title('Calibrated mags: zeropoint %s: %.3f' % (filt, zpt))
        plt.ylabel('Guide star mag (after calibration)')
        plt.xlabel('Guider frame number')
        plt.legend()
        ps.savefig()

        # Fit zeropoint: 24.898 --> assume 0.6 sec exposures -> 25.453
        # (-> correcting for airmass = 25.470)

        # vs DR10 r: corrected ~ 25.390
        
        # vs Copilot: eg zp_obs 26.807349479715313 kx 0.1 airmass 1.17
        # gain ~ 4: 1.505 mag --> 25.302

        np.savez(npzfn, paramnames=paramnames, expnum=expnum, chipnames=chipnames,
                 params=allparams, sig1s=sig1s,
                 allskies=allskies, strip_skies=strip_skies, filter=filt,
                 ref_mags=ref_mags, instmags=instmags, zpt=zpt,
                 use_for_zpt=use_for_zpt,
                 gaia_mags=mags, wcs=wcses, guide_xy=xys, guide_rd=rds,
                 apfluxes=apfluxes, apskies=apskies)
        plot_from_npz(expnum, npzfn)

if __name__ == '__main__':
    main()
