import os
import sys
from glob import glob
from datetime import datetime

import numpy as np
import pylab as plt
import fitsio
import photutils
from tractor import (PixPos, Flux, NCircularGaussianPSF, Image, PointSource,
                     ConstantSky, Tractor)
from astrometry.util.plotutils import PlotSequence
from astrometry.util.util import Sip
from astrometry.util.fits import fits_table, merge_tables
from astrometry.libkd.spherematch import match_radec
from legacypipe.gaiacat import GaiaCatalog


basedir = 'data-2024-02-28/guider_images/'
tempdir = 'temp'
#an_config = '~/cosmo/work/users/dstn/index-5200/cfg'
an_config = '/Users/dstn/data/INDEXES/5200/cfg'

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
    r = ('G', (0.7, 2.0), [ 0.02752588,  0.03825722, -0.3158438 ,  0.14626219]),
    #g = ('G', (0.7, 2.0), [ 0.63551392, -1.68398593,  1.98642154, -0.46866005]),
    g = ('G', (0.7, 2.0), [ 0.36423847, -0.98522136,  1.36913521, -0.31177573]),
    z = ('G', (0.7, 2.0), [ 0.51604316, -0.99860037,  0.18527867, -0.03979571]),
    N673 = ('G', (0.7, 2.0), [ 0.17001402,  0.12573145, -0.53929908,  0.22958496]),
)

#Median trans for N673 : 0.3772193640470505
#Median trans for g : 1.0593839883804321
#Median trans for r : 0.7122441828250885
#Median trans for z : 0.5655636541027943
for f,corr in [
    ('N673', 0.377219),
    ('g',  1.059383988),
    ('r',   0.712244182),
    ('z',   0.565563654)]:
    (b,cr,poly) = gaia_color_terms[f]
    poly[0] += -2.5*np.log10(corr)
    gaia_color_terms[f] = (b, cr, poly)


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

# Thanks, https://stackoverflow.com/questions/20601872/numpy-or-scipy-to-calculate-weighted-median
def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]
def weighted_quantiles(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

def fit_line_sad(x, y):
    from scipy.optimize import minimize
    # Fit a line, minimizing the sum of absolute differences
    b = np.median(y)
    # assume x sorted
    assert(np.all(np.diff(x) > 0))
    # Initial slope estimate: medians of first & third third
    third = len(y)//3
    m = ((np.median(y[-third:]) - np.median(y[:third])) /
         (np.median(x[-third:]) - np.median(x[:third])))
    def sad(p):
        (bi,mi) = p
        yi = bi + mi * x
        return np.sum(np.abs(yi - y))
    res = minimize(sad, (b,m))
    print('Optimizer: success?', res.success)
    if not res.success:
        print('Reason:', res.message)
    return res.x

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

def plot_from_npz(expnum, fn, summary=False):
    print('Reading', fn)
    R = np.load(fn, allow_pickle=True)
    #print('Keys:', R.keys())
    chipnames = R['chipnames']
    filt = str(R['filter'])
    ref_mags = R['ref_mags']
    instmags = R['instmags']
    use_for_zpt = R['use_for_zpt']

    if not ('gaiastars' in R):
    #if True:
        gaiastars = []
        gaia = GaiaCatalog(cache=True)
        rds = np.array(R['guide_rd'])
        ra  = rds[:,0]
        dec = rds[:,1]
        for wcs,r,d in zip(R['wcs'], ra, dec):
            gaiacat = gaia.get_catalog_in_wcs(wcs)
            # HACK - distance func
            i = np.argmin(np.hypot(gaiacat.ra - r, gaiacat.dec - d))
            gaiastars.append(gaiacat[np.array([i])])
        gaiastars = merge_tables(gaiastars)
        # print('Gaia stars:')
        # gaiastars.about()
        # print('Number of Gaia stars brighter than 14th:',
        #       np.sum(gaiastars.phot_g_mean_mag < 14))
        keys = R.keys()
        R = dict([(k, R[k]) for k in keys])
        R['gaiastars'] = gaiastars.to_dict()
        np.savez(fn, **R)
        R = np.load(fn, allow_pickle=True)

    # npz mangles fits_tables & dicts    
    gaiastars = R['gaiastars'].tolist()
    gs = fits_table()
    for k,v in gaiastars.items():
        gs.set(k, v)
    gaiastars = gs

    if not ('gaia_mags' in R and 'apfluxes' in R):
        print('No Gaia mags / ap fluxes in', fn)
        return

    nframes, nguide = instmags.shape
    apfluxes = R['apfluxes']
    apskies = R['apskies']
    gaia_mags = R['gaia_mags']
    g  = gaia_mags[:,0]
    bp = gaia_mags[:,1]
    rp = gaia_mags[:,2]
    params = R['params']
    paramnames = R['paramnames']
    print('Params shape:', params.shape)
    # ASSUME the params...
    fit_psf_fwhm = 2.35 * params[:, :, 0]
    fit_sky = params[:, :, 1]
    fit_x = params[:, :, 2]
    fit_y = params[:, :, 3]
    fit_flux = params[:, :, 4]

    nominal_zpt = nominal_zeropoints[filt]
    k_airmass_ext = airmass_extinctions.get(filt, 0.)
    airmass = AIRMASSES.get(expnum, 1.0)
    if k_airmass_ext == 0 or airmass == 1.0:
        print('WARNING, no airmass correction for expnum', expnum,
              '(airmass %.2f, k_co %.3f)' % (airmass, k_airmass_ext))

    expected_zpt = nominal_zpt - k_airmass_ext * (airmass - 1.0)

    # WHOOPS messed up the color term!
    assert(filt is not None)
    bprp = bp - rp
    colorterm = gaia_color_terms.get(filt, None)
    assert(colorterm is not None)
    gaia_band, (c_lo,c_hi), poly = colorterm
    ind = {'G':0, 'BP':1, 'RP':2}.get(gaia_band)
    base = gaia_mags[:, ind]
    use_for_zpt[np.logical_or(bp == 0, rp == 0)] = False
    # Drop stars outside the color term range
    use_for_zpt[np.logical_or(bprp < c_lo, bprp > c_hi)] = False
    bprp = np.clip(bprp, c_lo, c_hi)
    colorterm = 0.
    for i,c in enumerate(poly):
        colorterm = colorterm + c * bprp**i
    ref_mags[use_for_zpt] = (base + colorterm)[use_for_zpt]

    ref_inst = ref_mags - expected_zpt

    transp = 10.**((instmags - ref_inst[np.newaxis, :])/-2.5)

    d_apflux = apfluxes.copy()
    d_apflux[1:,:] = np.diff(d_apflux, axis=0)
    d_apsky = apskies.copy()
    d_apsky [1:,:] = np.diff(d_apsky , axis=0)
    d_apflux -= d_apsky

    ap_instmags = -2.5 * np.log10(d_apflux)
    ap_transp = 10.**((ap_instmags - ref_inst[np.newaxis, :])/-2.5)

    xys = np.array(R['guide_xy'])
    xx = xys[:,0]
    yy = xys[:,1]

    rds = np.array(R['guide_rd'])
    ra  = rds[:,0]
    dec = rds[:,1]

    gaia_bp_snr = gaiastars.phot_bp_mean_flux_over_error
    gaia_rp_snr = gaiastars.phot_rp_mean_flux_over_error

    if not summary:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('guider-exp%i' % expnum)
        ps.skipto(7)

    T = fits_table()
    for k in ['apfluxes', 'apskies', 'instmags']:
        T.set(k, R[k])
    T.guideframe = np.arange(nframes)
    T.transparency = transp
    T.ap_transparency = ap_transp
    loc = locals()
    for k in ['d_apflux', 'ap_instmags',
              'fit_psf_fwhm', 'fit_sky', 'fit_x', 'fit_y', 'fit_flux']:
        T.set(k, loc[k])
    #for k in ['ref_mags', 'use_for_zpt']:
    #    T.set(k, R[k][np.newaxis,:].repeat(nframes, axis=0))
    for k in ['g', 'bp', 'rp', 'xx', 'yy',
              'ref_mags', 'use_for_zpt', 'gaia_bp_snr', 'gaia_rp_snr']:
        T.set(k, loc[k][np.newaxis,:].repeat(nframes, axis=0))
    for k,t in [('expnum', int), ('airmass', np.float32),
                ('expected_zpt', np.float32)]:
        T.set(k, np.zeros(nframes, t) + loc[k])
    T.filter = np.array([filt]*nframes)
    meta = dict(expnum=expnum, chipnames=chipnames, filter=filt,
                expected_zpt=expected_zpt, airmass=airmass)
    return T,meta

def main():
    fns = glob(os.path.join(basedir, 'DECam_guider_*_00000000.fits.gz'))
    fns.sort()
    expnums = []
    for fn in fns:
        words = fn.split('_')
        expnums.append(int(words[-2]))
    print(len(expnums), 'exposures found')

    if False:
        TT = []
        mm = []
        for expnum in expnums:
            npzfn = 'guider-tractor-fit-%i.npz' % expnum
            if os.path.exists(npzfn):
                R = np.load(npzfn, allow_pickle=True)
                # print('Keys:', R.keys())
                # arr = R['arr_0']
                # arr = arr.tolist()
                # print('arr:', type(arr), arr.keys())
                # np.savez(npzfn, **arr)
                X = plot_from_npz(expnum, npzfn, summary=True)
                if X is None:
                    continue
                T,meta = X
                TT.append(T)
                mm.append(meta)

        T = merge_tables(TT)
        T.about()
        nguide = 4

        filts = np.unique(T.filter)
        
        plt.clf()
        plt.scatter(T.expnum[:,np.newaxis].repeat(nguide,axis=1),
                    T.ap_transparency, s=1)
        plt.savefig('trends.png')

        print('max guide frame:', T.guideframe.max())
        plt.clf()
        plt.subplots_adjust(hspace=0.1)
        elo,ehi = T.expnum.min(), T.expnum.max()
        for i,f in enumerate(filts):
            plt.subplot(4,1, i+1)
            I = T.use_for_zpt * (T.filter[:,np.newaxis] == f)
            print('Median trans for', f, ':', np.median(T.ap_transparency[I]))
            I = np.flatnonzero(T.filter == f)
            yl,yh = 0.5, 1.5

            for j,name in enumerate(meta['chipnames']):
                plt.plot((T.expnum + T.guideframe/150)[I],
                         np.clip(T.ap_transparency[I, j], yl, yh),
                         '.', ms=1, label='%s in %s' % (f, name))

            tr = []
            ee = np.unique(T.expnum[T.filter == f])
            for e in ee:
                tr.append(np.median(T.ap_transparency[(T.expnum == e)[:,np.newaxis] * T.use_for_zpt]))
            plt.plot(ee, tr, 'k.-')
            # plt.plot((T.expnum + T.guideframe/150)[I, np.newaxis].repeat(nguide,axis=1),
            #          np.clip(T.ap_transparency[I,:], yl, yh),
            #          '.', ms=1, label=f)
            plt.legend(fontsize=8)
            plt.xlim(elo,ehi)
            plt.ylim(yl,yh)
            if i == 1:
                plt.ylabel('ap Transparency')
            if i == 3:
                plt.xlabel('Expnum')
            else:
                plt.xticks([])
        plt.suptitle('Guider data, 2024-02-28: Transparency')
        plt.savefig('trends2.png')

        d_apsky = T.apskies.copy()
        d_apsky [1:,:] = np.diff(d_apsky , axis=0)
        #print('d_apsky:', d_apsky)

        filts = np.unique(T.filter)
        for f in filts:
            print(np.sum(T.filter == f), 'in', f)
            I = T.use_for_zpt * (T.filter[:,np.newaxis] == f)
            plt.clf()

            plt.subplot(2,1,1)
            y = (T.instmags - T.g)[I]
            m = np.median(y)
            plt.scatter((T.bp - T.rp)[I], np.clip(y, m-1, m+1), s=1,
                        c=T.airmass[:,np.newaxis].repeat(nguide, axis=1)[I])
            plt.xlabel('BP - RP')
            plt.ylabel('Instmag - G')
            cb = plt.colorbar()
            cb.set_label('Airmass')

            plt.subplot(2,1,2)
            #y = (T.instmags - T.ref_mags)[I]
            y = (T.ap_instmags - T.ref_mags)[I]
            m = np.median(y)
            sk = d_apsky[I]
            mn,mx = np.percentile(sk, [5,95])
            plt.scatter((T.bp - T.rp)[I], np.clip(y, m-1, m+1), s=1,
                        c = T.xx[I] % 1024)
            #c = d_apsky[I], vmin=mn, vmax=mx)
            #c = T.g[I])
            #c=np.minimum(T.gaia_bp_snr, T.gaia_rp_snr)[I],
            #vmin=0, vmax=50)
            cb = plt.colorbar()
            #cb.set_label('BP/RP S/N')
            #cb.set_label('Gaia G')
            #cb.set_label('Sky')
            cb.set_label('x (mod 1024)')
            plt.xlabel('BP - RP')
            plt.ylabel('ApInstmag - (G+color)')

            # plt.scatter(T.ref_mags[I], T.transparency[I],
            #             c=T.bp[I]-T.rp[I], s=1)
            # plt.xlabel('Gaia-predicted mag')
            # plt.ylabel('Transparency')
            # cb = plt.colorbar()
            # cb.set_label('Gaia BP-RP')
            plt.suptitle('2024-02-28: filter %s' % f)
            plt.savefig('trends-%s.png' % f)

        T.writeto('guider-2024-02-28.fits')
        return

    threads = 4
    #if threads:
    from astrometry.util.multiproc import multiproc
    mp = multiproc(threads)

    #expnums = [1278635]
    expnums = [1278649]

    #mp.map(bounce_one_expnum, expnums)
    #return

    for expnum in expnums:
        #     #if expnum < 1278723:
        #     #    continue
        #if expnum != 1278635:
        #    continue

        R = run_expnum(expnum)
        if R is None:
            continue

        # npzfn = 'guider-tractor-fit-%i.npz' % expnum
        # np.savez(npzfn, **R)
        # plot_from_npz(expnum, npzfn)

def bounce_one_expnum(expnum):
    npzfn = 'guider-tractor-fit-%i.npz' % expnum
    #if os.path.exists(npzfn):
    #    return
    R = run_expnum(expnum)
    if R is None:
        return
    np.savez(npzfn, **R)
    plot_from_npz(expnum, npzfn)
    
def run_expnum(expnum):
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

    # Measure row-wise median per amp
    #plt.clf()
    for i,img in enumerate(imgs):
        imgl = img[:,:1024]
        imgr = img[:,1024:]
        gl = np.median(imgl, axis=1)
        gr = np.median(imgr, axis=1)
        # omit top/bottom -- they have extra brightness
        trim = 50
        xx = np.arange(trim, len(gl)-trim)
        # fit for slope
        offl,slopel = fit_line_sad(xx, gl[trim:-trim])
        offr,sloper = fit_line_sad(xx, gr[trim:-trim])
        # subtract slope
        img[trim:-trim, :1024] -= (offl + xx * slopel)[:, np.newaxis]
        img[trim:-trim, 1024:] -= (offr + xx * sloper)[:, np.newaxis]
        # top & bottom - just subtract row-wise medians.
        img[:trim,  :1024] -= np.median(img[:trim , :1024], axis=1)[:, np.newaxis]
        img[-trim:, :1024] -= np.median(img[-trim:, :1024], axis=1)[:, np.newaxis]
        img[:trim,  1024:] -= np.median(img[:trim , 1024:], axis=1)[:, np.newaxis]
        img[-trim:, 1024:] -= np.median(img[-trim:, 1024:], axis=1)[:, np.newaxis]
        #p = plt.plot(gl)
        #plt.plot(xx, offl + xx * slopel, 'k--')#, color=p[0].get_color())
        #p = plt.plot(gr)
        #plt.plot(xx, offr + xx * sloper, 'k--')#, color=p[0].get_color())
    #plt.suptitle('Per-amp sky subtracted')
    #ps.savefig()

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

    # Grab the Astrometry.net detected stars list
    # Match to Gaia and photometer them
    plotdata = []
    # subimage half-size
    S = 25
    ss = 2*S+1
    sx = np.arange(ss)
    sy = np.arange(ss)
    rad = 20
    starmask = np.hypot(sx[np.newaxis,:] - S, sy[:,np.newaxis] - S) > rad
    ap_profiles = []

    for j,(img,name,wcs,cat) in enumerate(zip(imgs, chipnames, wcses, cats)):
        fn = os.path.join(tempdir, '%i-%s.axy' % (expnum, name))
        if not os.path.exists(fn):
            print('Does not exist:', fn)
        xy = fits_table(fn)
        print(len(xy), 'stars detected by Astrometry.net star finder')
        print(len(cat), 'Gaia stars')
        ra,dec = wcs.pixelxy2radec(xy.x + 1., xy.y + 1.)
        # Match to Gaia
        I,J,d = match_radec(ra, dec, cat.ra, cat.dec, 1./3600., nearest=True)
        print(len(I), 'stars matched to Gaia')

        # Estimate per-pixel noise via Blanton's "5"-pixel MAD
        step = 5
        slice1 = (slice(0,-5,step),slice(0,-5,step))
        slice2 = (slice(5,None,step),slice(5,None,step))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)

        # Cut stars near the edges
        H,W = img.shape
        K = np.flatnonzero((xy.x[I] >= S) * (xy.y[I] >= S) *
                           (xy.x[I] < (W-S)) * (xy.y[I] < (H-S)) *
                           (cat.phot_g_mean_mag[J] != 0) *
                           (cat.phot_bp_mean_mag[J] != 0) *
                           (cat.phot_rp_mean_mag[J] != 0))
        I = I[K]
        J = J[K]
        K = np.argsort(cat.phot_g_mean_mag[J])
        I = I[K]
        J = J[K]

        med = np.median(img)

        if j == 0:
            plt.clf()
            plt.subplots_adjust(hspace=0, wspace=0)
        fluxes = []
        apfluxes = []
        apfluxes2 = []
        psfw = []
        tractors = []
        for i,(x,y) in enumerate(zip(xy.x[I], xy.y[I])):
            ix,iy = int(x),int(y)
            subimg = img[iy-S:iy+S+1, ix-S:ix+S+1]
            #print('subimg shape', subimg.shape)
            h,w = subimg.shape
            tim = Image(subimg, inverr=np.ones_like(subimg)/sig1,
                        psf=NCircularGaussianPSF([2.], [1.]),
                        sky=ConstantSky(med))
            tim.psf.freezeParam('weights')
            tim.sig1 = sig1
            flux = np.sum(subimg) - med * h*w
            flux = max(flux, 100)
            src = PointSource(PixPos(S, S), Flux(flux))
            tr = Tractor([tim], [src])
            X = tr.optimize_loop()
            fluxes.append(src.getBrightness().val)
            psfw.append(tr.getParams()[0])
            tractors.append(tr)
            
            # Aperture sky
            apsky = np.median(subimg[starmask])
            # Aperture photometry
            apxy = np.array([[S, S],])
            ap = []
            aprad_pix = 15.
            aper = photutils.aperture.CircularAperture(apxy, aprad_pix)
            p = photutils.aperture.aperture_photometry(subimg - apsky, aper)
            ap_profiles.append((subimg - apsky)[S, :])
            apflux = p.field('aperture_sum')
            apflux = float(apflux.data[0])
            apfluxes.append(apflux)

            aprad_pix = 10.
            aper = photutils.aperture.CircularAperture(apxy, aprad_pix)
            p = photutils.aperture.aperture_photometry(subimg - apsky, aper)
            apflux = p.field('aperture_sum')
            apflux = float(apflux.data[0])
            apfluxes2.append(apflux)

            if j == 0 and i < 15:
                #plt.subplot(5, 10, 2*i + 1)
                plt.subplot(5, 9, 3*i + 1)
                mn,mx = np.percentile(subimg.ravel(), [10,98])
                ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
                plt.imshow(subimg, **ima)
                plt.xticks([]); plt.yticks([])
                #plt.subplot(5, 10, 2*i + 2)
                plt.subplot(5, 9, 3*i + 2)
                mod = tr.getModelImage(0)
                plt.imshow(mod, **ima)
                plt.xticks([]); plt.yticks([])
                plt.subplot(5, 9, 3*i + 3)
                chi = (subimg - mod) / tim.sig1
                mx = np.percentile(np.abs(chi.ravel()), 98)
                plt.imshow(chi, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx)
                plt.xticks([]); plt.yticks([])
        if j == 0:
            ps.savefig()

        flux = np.array(fluxes)
        apflux = np.array(apfluxes)
        apflux2 = np.array(apfluxes2)
        instmag = -2.5 * np.log10(flux)
        apinstmag = -2.5 * np.log10(apflux)
        apinstmag2 = -2.5 * np.log10(apflux2)
        g = cat.phot_g_mean_mag[J]
        bp = cat.phot_bp_mean_mag[J]
        rp = cat.phot_rp_mean_mag[J]

        # Repeat the fitting with a fixed PSF width per chip
        medw = np.median(psfw)
        fluxes2 = []
        for tr in tractors:
            tim = tr.images[0]
            #print('PSF params before:', tim.psf.getParams())
            tim.psf.setParams([medw])#, 1.])
            #print('PSF params after:', tim.psf.getParams())
            tim.freezeParam('psf')
            #tr.freezeParam('images')
            #print('Parameters:', tr.getParamNames())
            X = tr.optimize_loop()
            src = tr.catalog[0]
            fluxes2.append(src.getBrightness().val)

        flux2 = np.array(fluxes2)
        instmag2 = -2.5 * np.log10(flux2)
        plotdata.append((instmag, instmag2, apinstmag, apinstmag2,
                         g, bp, rp, np.array(psfw)))
        
    plt.clf()
    for p in ap_profiles:
        plt.plot(p, alpha=0.1)
    plt.title('Aperture photometry: profiles')
    plt.yscale('symlog', linthresh=40)
    ps.savefig()

    colorx = np.linspace(0.4, 2.5, 25)
    colory = np.zeros_like(colorx)
    colorterm = gaia_color_terms.get(filt, None)
    if colorterm is not None:
        gaia_band, (c_lo,c_hi), poly = colorterm
        bprp = np.clip(colorx, c_lo, c_hi)
        colorterm = 0.
        for i,c in enumerate(poly):
            colorterm = colorterm + c * bprp**i
        colory = colorterm

    ylo,yhi = -26, -24.5
    xx = []
    yy = []
    yy2 = []
    yy3 = []
    yy4 = []
    cc = []
    for instmag,instmag2,apinstmag,apinstmag2,g,bp,rp,psfw in plotdata:
        #plt.plot(bp - rp, np.clip(instmag - g, ylo, yhi), '.')
        xx.append(bp-rp)
        yy.append(instmag - g)
        yy2.append(apinstmag - g)
        yy3.append(apinstmag2 - g)
        yy4.append(instmag2 - g)
        cc.append(psfw)
    cc = np.hstack(cc)
    yy = np.hstack(yy)
    yy2 = np.hstack(yy2)
    yy3 = np.hstack(yy3)
    yy4 = np.hstack(yy4)
    clo,chi = np.percentile(cc[(yy > ylo) * (yy < yhi)], [5,95])
    plt.clf()
    plt.scatter(np.hstack(xx), np.clip(yy, ylo, yhi), c=cc,
                vmin=clo, vmax=chi, s=4)
    #ax = plt.axis()
    #print('color', colorx, colory)
    #print('shifted', colory - np.median(colory) + np.median(yy[np.isfinite(yy)]))
    #plt.plot(colorx, colory - np.median(colory) + np.median(yy[np.isfinite(yy)]), 'k-')
    #plt.axis(ax)
    cb = plt.colorbar()
    cb.set_label('PSF size')
    plt.xlabel('Gaia Bp - Rp (mag)')
    plt.ylabel('Guider inst mag. - G (mag)')
    plt.title('Full-frame guider image (tractor phot)')
    plt.ylim(ylo, yhi)
    ps.savefig()

    plt.clf()
    for i,(instmag,instmag2,apinstmag,apinstmag2,g,bp,rp,psfw) in enumerate(plotdata):
        this_xx = bp - rp
        this_yy4 = instmag2 - g
        plt.plot(this_xx, np.clip(this_yy4, ylo, yhi), '.', label=chipnames[i])
    #cb = plt.colorbar()
    #cb.set_label('PSF size')
    plt.legend()
    plt.xlabel('Gaia Bp - Rp (mag)')
    plt.ylabel('Guider inst mag. - G (mag)')
    plt.title('Full-frame guider image (tractor phot 2)')
    plt.ylim(ylo, yhi)
    ps.savefig()

    plt.clf()
    plt.scatter(np.hstack(xx), np.clip(yy2, ylo, yhi), c=cc,
                vmin=clo, vmax=chi, s=4)
    cb = plt.colorbar()
    cb.set_label('PSF size')
    plt.xlabel('Gaia Bp - Rp (mag)')
    plt.ylabel('Guider inst mag. - G (mag)')
    plt.title('Full-frame guider image (ap phot)')
    plt.ylim(ylo, yhi)
    ps.savefig()

    plt.clf()
    plt.scatter(np.hstack(xx), np.clip(yy3, ylo, yhi), c=cc,
                vmin=clo, vmax=chi, s=4)
    cb = plt.colorbar()
    cb.set_label('PSF size')
    plt.xlabel('Gaia Bp - Rp (mag)')
    plt.ylabel('Guider inst mag. - G (mag)')
    plt.title('Full-frame guider image (ap phot 2)')
    plt.ylim(ylo, yhi)
    ps.savefig()

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
    gaiastars = []
    for j,((x,y),cat) in enumerate(zip(xys, cats)):
        d = np.hypot(x - (cat.x-1), y - (cat.y-1))
        i = np.argmin(d)
        print('Min distance to Gaia star: %.2f pixels' % d[i])
        if d[i] > 5.:
            print('WARNING: failed to find Gaia star near supposed guide star')
            use_for_zpt[j] = False
            m = (0., 0., 0.)
            fake = fits_table()
            fake.ra = np.array([0])
            gaiastars.append(fake)
        else:
            m = (cat.phot_g_mean_mag[i], cat.phot_bp_mean_mag[i],
                 cat.phot_rp_mean_mag[i])
            print('Gaia star: G %.2f, BP %.2f, RP %.2f' % m)
            gaiastars.append(cat[np.array([i])])
        mags.append(m)
    mags = np.array(mags)
    gaiastars = merge_tables(gaiastars, columns='fillzero')

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
                colorterm = colorterm + c * bprp**i
            
            ref_mags[use_for_zpt] = (base + colorterm)[use_for_zpt]
            print('Color term corrected mags:',
                  ', '.join(['%.2f'%m for m in ref_mags]))
        else:
            print('No COLOR TERM for filter', filt, '!')
    else:
        print('Unknown filter for expnum', expnum, '!')
        return

    # if old_r:
    #     R.update(gaia_mags=mags, wcs=wcses, guide_xy=xys, guide_rd=rds)
    #     print('Updating', npzfn)
    #     np.savez(npzfn, R)
    #     continue

    plt.clf()
    plt.subplots_adjust(hspace=0.15, wspace=0)    
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
    
    fns = fns[startframe:]
    nframes = len(fns)
    print('Reading', nframes, 'guider frames...')
    gstack = []
    strip_skies = np.zeros((nframes, nguide), np.float32)
    ut_times = []
    for i,fn in enumerate(fns):
        roifn = fn.replace('.fits.gz', '_roi.fits.gz')
        if not os.path.exists(roifn):
            print()
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
        print('.', end='')
        sys.stdout.flush()
    print()

    print('Read', len(gstack), 'guider ROI frames')

    if len(gstack) < nframes:
        nframes = len(gstack)
        strip_skies = strip_skies[:nframes, :]

    # Mask out star pixels before computing median (sky)
    h,w = gstack[0][0].shape
    rad = 8
    x = np.arange(w)
    y = np.arange(h)
    starmask = np.hypot(x[np.newaxis,:] - w/2, y[:,np.newaxis] - h/2) > rad

    # Tractor fit for the cumulative image after each guider frame
    do_fit = [True]*nguide
    paramnames = ['PSF-sigma', 'Sky', 'X', 'Y', 'Flux']
    nparams = len(paramnames)
    allskies = np.zeros((nframes, nguide), np.float32)
    allparams = np.zeros((nframes, nguide, nparams), np.float32)
    sig1s = np.zeros((nframes, nguide), np.float32)
    apfluxes = np.zeros((nframes, nguide), np.float32)
    apskies = np.zeros((nframes, nguide), np.float32)
    tractors = []
    init_tractor_plot = []
    for i,gims in enumerate(gstack):
        for j,img in enumerate(gims):
            if i == 0:
                img = img.copy()
            else:
                tr = tractors[j]
                tim = tr.images[0]
                tim.data += img
                img = tim.data

            med = np.median(img[starmask])
            allskies[i, j] = med

            # Estimate per-pixel noise via Blanton's "5"-pixel MAD
            step = 3
            slice1 = (slice(0,-5,step),slice(0,-5,step))
            slice2 = (slice(5,None,step),slice(5,None,step))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            sig1s[i, j] = sig1

            if not do_fit[j]:
                continue

            if i == 0:
                tim = Image(img, inverr=np.ones_like(img)/sig1,
                            psf=NCircularGaussianPSF([2.], [1.]),
                            sky=ConstantSky(med))
                tim.psf.freezeParam('weights')
                tim.sig1 = sig1

                h,w = img.shape
                flux = np.sum(img) - med * h*w
                flux = max(flux, 100)
                src = PointSource(PixPos(25, 25), Flux(flux))
                tr = Tractor([tim], [src])
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

                init_tractor_plot.append((img.copy(), med, tim.sig1, mod1))
            else:
                tim.sig1 = sig1
                tim.inverr[:,:] = 1./sig1
                tr.optimize_loop(shared_params=False)

            #print('Fit source:', tr.catalog[0], 'and PSF',
            #    tr.images[0].getPsf().sigmas[0])
            allparams[i, j, :] = tr.getParams()

            # Aperture photometry
            apxy = np.array([[(w-1)/2, (h-1)/2],])
            ap = []
            aprad_pix = 15.
            aper = photutils.aperture.CircularAperture(apxy, aprad_pix)
            p = photutils.aperture.aperture_photometry(tim.data, aper)
            apflux = p.field('aperture_sum')
            apflux = float(apflux.data[0])
            # accumulated strip_skies
            ss = np.sum(strip_skies[:i+1, j])
            sky = ss * np.pi * aprad_pix**2
            apskies[i, j] = sky
            apfluxes[i, j] = apflux

            p = allparams[i,j,:]
            psfsig,sky,x,y,flux = p
            if flux < 0 or psfsig < 0 or psfsig > 10:
                print('Fit went wacky: flux', flux, 'psf sigma', psfsig)
                do_fit[j] = False
                use_for_zpt[j] = False

        print('.', end='')
        sys.stdout.flush()
    print()
            
    # Initial tractor fit plot
    plt.clf()
    for j,(img,med,sig1,mod) in enumerate(init_tractor_plot):
        plt.subplot(4,4, j + 1)
        plt.imshow(img, interpolation='nearest', origin='lower')
        plt.title(chipnames[j])
        plt.subplot(4,4, j + 5)
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=med-3*sig1, vmax=med+10*sig1)
        plt.imshow(img, **ima)
        plt.subplot(4,4, j + 9)
        plt.imshow(mod, **ima)
        plt.subplot(4,4, j + 13)
        plt.imshow((img - mod)/sig1, interpolation='nearest', origin='lower',
                   vmin=-10, vmax=+10)
    plt.suptitle('Initial tractor fits')
    ps.savefig()

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
    dflux = np.diff(np.vstack(([0]*nguide, pflux)), axis=0)
    instmags = -2.5 * np.log10(dflux)

    if np.sum(use_for_zpt):
        #print('computing zpt from', (ref_mags[np.newaxis, :] - instmags)[:, use_for_zpt].shape)
        zpt = np.median((ref_mags[np.newaxis, :] - instmags)[:, use_for_zpt])
    else:
        print('Warning: no stars are good for zeropointing')
        # ... so use 'em all anyway
        zpt = np.median(ref_mags[np.newaxis, :] - instmag)
    print('Fit zeropoint: %.3f' % zpt)

    plt.clf()
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

    nominal_zpt = nominal_zeropoints[filt]
    k_airmass_ext = airmass_extinctions.get(filt, 0.)
    airmass = AIRMASSES.get(expnum, 1.0)
    if k_airmass_ext == 0 or airmass == 1.0:
        print('WARNING, no airmass correction for expnum', expnum,
              '(airmass %.2f, k_co %.3f)' % (airmass, k_airmass_ext))

    expected_zpt = nominal_zpt - k_airmass_ext * (airmass - 1.0)
    ref_inst = ref_mags - expected_zpt

    transp = 10.**((instmags - ref_inst[np.newaxis, :])/-2.5)

    d_apflux = apfluxes.copy()
    d_apflux[1:,:] = np.diff(d_apflux, axis=0)
    d_apsky = apskies.copy()
    d_apsky [1:,:] = np.diff(d_apsky , axis=0)
    d_apflux -= d_apsky
    ap_instmags = -2.5 * np.log10(d_apflux)
    ap_transp = 10.**((ap_instmags - ref_inst[np.newaxis, :])/-2.5)
    
    plt.clf()
    for j in range(nguide):
        sty = '-'
        ll = ' (pred. mag %.2f)' % (ref_mags[j])
        if not use_for_zpt[j]:
            sty = ':'
            ll = ''
        p = plt.plot(100. * transp[:,j], sty, label=chipnames[j] + ll)
        plt.plot(100. * ap_transp[:,j], '--', color=p[0].get_color(),
                 label=chipnames[j] + ' (aperture)')
        # + ' (G = %.2f, BP-RP = %.2f)' % bprps[j])
    ax = plt.axis()
    plt.legend()
    plt.ylabel('Guide star implied transparency (%)')
    plt.xlabel('Guider frame number')
    plt.title('Transparency (exp %i: %s)' % (expnum, filt))
    yl,yh = plt.ylim()
    plt.ylim(0, yh)
    #plt.ylim(0, 120)
    ps.savefig()

    plt.clf()
    #flux = 10.**(instmags / -2.5)
    for j in range(nguide):
        p = plt.plot(dflux[:,j])
        c = p[0].get_color()
        plt.plot(d_apflux[:,j], '--', color=c)
    plt.ylabel('flux')
    plt.xlabel('Guider frame number')
    plt.title('(exp %i: %s)' % (expnum, filt))
    ps.savefig()

    
    # Fit zeropoint: 24.898 --> assume 0.6 sec exposures -> 25.453
    # (-> correcting for airmass = 25.470)

    # vs DR10 r: corrected ~ 25.390
    
    # vs Copilot: eg zp_obs 26.807349479715313 kx 0.1 airmass 1.17
    # gain ~ 4: 1.505 mag --> 25.302

    return dict(paramnames=paramnames, expnum=expnum, chipnames=chipnames,
             params=allparams, sig1s=sig1s,
             allskies=allskies, strip_skies=strip_skies, filter=filt,
             ref_mags=ref_mags, instmags=instmags, zpt=zpt,
             use_for_zpt=use_for_zpt,
             gaia_mags=mags, wcs=wcses, guide_xy=xys, guide_rd=rds,
             apfluxes=apfluxes, apskies=apskies,
             gaiastars=gaiastars)

if __name__ == '__main__':
    main()
