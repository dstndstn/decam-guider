import os
import numpy as np
import pylab as plt
import fitsio
from glob import glob
from datetime import datetime

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
    
    for expnum in expnums:

        if expnum != 1278647:
            continue
        #1278687
        
        ps = PlotSequence('guider-exp%i' % expnum)
        fns = glob(os.path.join(basedir, 'DECam_guider_%i_????????.fits.gz' % expnum))
        fns.sort()
        print('Expnum', expnum, ': found', len(fns), 'guider frames')

        fn = fns[0]
        chipnames,imgs = assemble_full_frames(fn)
        print('Chips:', chipnames)
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

        plt.clf()
        for i,img in enumerate(imgs):
            plt.subplot(2,2,i+1)
            mn,mx = np.percentile(img.ravel(), [25,99])
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            plt.title(chipnames[i]);
            if i in [0, 1]:
                plt.xticks([])
        ps.savefig()

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
                   + ' --no-plots -v ' #' --plot-scale 0.5 # --tag-all')
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
            print('max in strip:', ix, iy)
            # ASSUME it's in the middle!
            ix = 24
            y_star.append(np.argmax(strip[:,ix]))

        xys = []
        plt.clf()
        for i,img in enumerate(imgs):
            plt.subplot(2,4,i+1)
            mn,mx = np.percentile(img.ravel(), [25,99])
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            ax = plt.axis()
            x,y = roicols[i], y_star[i]
            x += 24
            xys.append((x,y))
            x0,x1 = x-24, x+24
            y0,y1 = y-24, y+24
            plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'r-')
            plt.title(chipnames[i]);
            plt.axis(ax)
        
            plt.subplot(2,4,4+i+1)
            plt.imshow(img, vmin=mn, vmax=mx, origin='lower', interpolation='nearest');
            plt.axis([x0,x1,y0,y1])
        ps.savefig()

        # Match Gaia catalog to selected guide star.
        mags = []
        for (x,y),cat in zip(xys, cats):
            d = np.hypot(x - cat.x, y - cat.y)
            i = np.argmin(d)
            print('Min distance to Gaia star: %.2f pixels' % d[i])
            m = (cat.phot_g_mean_mag[i], cat.phot_bp_mean_mag[i],
                 cat.phot_rp_mean_mag[i])
            print('Gaia star: G %.2f, BP %.2f, RP %.2f' % m)
            mags.append(m)
        mags = np.array(mags)

        # Tractor fit for the cumulative image after each guider frame
        gstack = []
        strip_skies = []
        ut_times = []
        for i in range(1, len(fns)):
            fn = fns[i]
            roifn = fn.replace('.fits.gz', '_roi.fits.gz')
            if not os.path.exists(roifn):
                print('No such file:', roifn)
                break
            F = fitsio.FITS(roifn, 'r')
            t = F[0].read_header()['UTSHUT'] #= '2024-02-29T01:45:30.524' / exp start
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
            ut_times.append(dt)
            gimgs = []
            for j in range(4):
                # Trim one pixel off each side of the guider images... otherwise, we sometimes get 'glitches' where the bottom row
                # is hot.
                im = F[j+1].read()
                im = im[1:-1, 1:-1]
                gimgs.append(im)
            if False and i<3:
                plt.clf()
                for j in range(4):
                    plt.subplot(2,2,j+1)
                    plt.imshow(gimgs[j])
                plt.show()
            gstack.append(gimgs)

            skies = []
            F = fitsio.FITS(fn, 'r')
            for j in range(4):
                sky = 0
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
                skies.append(sky/2)
            strip_skies.append(skies)
        strip_skies = np.array(strip_skies)

        print('Read', len(gstack), 'guider ROI frames')

        from tractor import PixPos, Flux, NCircularGaussianPSF, Image, PointSource, ConstantSky, Tractor

        # Mask out star pixels before computing median (sky)
        h,w = gstack[0][0].shape
        rad = 8
        x = np.arange(w)
        y = np.arange(h)
        starmask = np.hypot(x[np.newaxis,:] - w/2, y[:,np.newaxis] - h/2) > rad

        allskies = []
        tractors = []
        skies = []
        for img in gstack[0]:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            step = 3
            slice1 = (slice(0,-5,step),slice(0,-5,step))
            slice2 = (slice(5,None,step),slice(5,None,step))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
        
            med = np.median(img)
            tim = Image(img.copy(), inverr=np.ones_like(img)/sig1, psf=NCircularGaussianPSF([2.], [1.]),
                        sky=ConstantSky(med))
            tim.psf.freezeParam('weights')
            tim.sig1 = sig1
            skies.append(np.median(img[starmask]))
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
        allskies.append(skies)

        # Initial tractor fits
        plt.clf()
        for j,tr in enumerate(tractors):
            plt.subplot(4,4, j + 1)
            plt.imshow(tim.data, interpolation='nearest', origin='lower')
            plt.subplot(4,4, j + 5)
            #plt.subplot(3,4, j + 1)
            tim = tr.images[0]
            med = np.median(tim.data)
            sig1 = tim.sig1
            ima = dict(interpolation='nearest', origin='lower', vmin=med-3*sig1, vmax=med+10*sig1)
            plt.title(chipnames[j])
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

        allparams = [np.vstack([tr.getParams() for tr in tractors])]
        sig1s = [np.array([tr.images[0].sig1 for tr in tractors])]
        for j,gims in enumerate(gstack[1:]):
            #print('Frame', j)
            skies = []
            for k,(tr,im) in enumerate(zip(tractors,gims)):
                tim = tr.images[0]
                tim.data += im
                skies.append(np.median(tim.data[starmask]))
                # Estimate per-pixel noise via Blanton's 5-pixel MAD
                step = 3
                slice1 = (slice(0,-5,step),slice(0,-5,step))
                slice2 = (slice(5,None,step),slice(5,None,step))
                mad = np.median(np.abs(tim.data[slice1] - tim.data[slice2]).ravel())
                sig1 = 1.4826 * mad / np.sqrt(2.)
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1
                tr.optimize_loop()
                #print('Params:', tr.getParams())
            allparams.append(np.vstack([tr.getParams() for tr in tractors]))
            sig1s.append(np.array([tr.images[0].sig1 for tr in tractors]))
            allskies.append(skies)
            print('.', end='')
        print()
        allparams = np.dstack(allparams)
        sig1s = np.vstack(sig1s)
        allskies = np.array(allskies)

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

        paramnames = ['PSF-sigma', 'Sky', 'X', 'Y', 'Flux']

        np.savez('guider-tractor-fit-%i.npz' % expnum, paramnames=paramnames, expnum=expnum, chipnames=chipnames, params=allparams, sig1s=sig1s,
                 allskies=allskies, strip_skies=strip_skies)

        for i in [0]:
            pname = paramnames[i]
            plt.clf()
            for j in range(4):
                plt.plot(allparams[j, i, :], label=chipnames[j])
            plt.title(pname)
            plt.legend()
            ps.savefig()

        plt.clf()
        for j in range(4):
            plt.plot(strip_skies[:,j], label=chipnames[j])
        plt.title('Strip median sky')
        plt.legend()
        ps.savefig()

        plt.clf()
        for j in range(4):
            plt.plot(np.diff(allskies[:, j]), label=chipnames[j])
        plt.title('Delta median ROI sky')
        plt.legend()
        ps.savefig()

        for i in [1,4]:
            plt.clf()
            for j in range(4):
                plt.plot(np.diff(allparams[j, i, :]), label=chipnames[j])
            plt.title('Delta ' + paramnames[i])
            plt.legend()
            ps.savefig()

        dflux = np.diff(allparams[:, 4, :])
        plt.clf()
        gmags = mags[:,0]
        instmags = -2.5 * np.log10(dflux)
        zpt = np.median(gmags[:,np.newaxis] - instmags)
        for j in range(4):
            plt.axhline(gmags[j], color='k', label=('Gaia G' if j==0 else None))
            plt.plot(zpt + instmags[j,:], label=chipnames[j])
        plt.title('Calibrated mag')
        plt.legend()
        ps.savefig()

            
if __name__ == '__main__':
    main()
