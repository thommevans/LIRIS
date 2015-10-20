import numpy as np
import pdb, sys, os
import reduction
import fitsio
import matplotlib.pyplot as plt


# Number of FITS extensions for the images in this dataste:
N_EXTS = 1

RAW_DDIR = '/media/hdisk1/asosx118/evanst/data/wht/liris/2013A'
CAL_DDIR = '/media/hdisk1/data1/wht/liris/2013A'
ADIR = '/home/tevans/analysis/wht/liris/2013A'


def make_master_cals( night='' ):
    """
    This is a fairly bloated routine for LIRIS, mainly because there are dim and
    bright frames, and two different arc lamps (Ar and Xe), both taken at start
    and end of the night.
    """
    

    adir_full = os.path.join( ADIR, night )
    raw_ddir_full = os.path.join( RAW_DDIR, '{0}/raw'.format( night ) )
    cal_ddir_full = os.path.join( CAL_DDIR, night )
    ddir_dark = os.path.join( cal_ddir_full, 'dark' )
    ddir_flat = os.path.join( cal_ddir_full, 'flat' )

    # Create the list of darks:
    if night=='20130517':

        # Darks:
        dark_framen = range( 1907393, 1907432+1 ) # 45sec
        ndark = len( dark_framen )
        dark_frames = []
        for i in range( ndark ):
            dark_frames += [ 'r{0:.0f}.fit'.format( dark_framen[i] ) ]
        dark_list_filename = 'dark.lst'
        mdark_filename = 'mdark.fit'

        # Flats

        flat_1arcsec_dim_framen = range( 1906407, 1906456+1 )
        flat_1arcsec_dim_list_filename = 'domeflat_1arcsec_dim.lst'
        mflat_1arcsec_dim_filename = 'mdomeflat_1arcsec_dim.fit'
        nflat = len( flat_1arcsec_dim_framen )
        flat_1arcsec_dim_frames = []
        for i in range( nflat ):
            flat_1arcsec_dim_frames += [ 'r{0:.0f}.fit'.format( flat_1arcsec_dim_framen[i] ) ]

        flat_1arcsec_bri_framen = range( 1906307, 1906406+1 )
        flat_1arcsec_bri_list_filename = 'domeflat_1arcsec_bri.lst'
        mflat_1arcsec_bri_filename = 'mdomeflat_1arcsec_bri.fit'
        nflat = len( flat_1arcsec_bri_framen )
        flat_1arcsec_bri_frames = []
        for i in range( nflat ):
            flat_1arcsec_bri_frames += [ 'r{0:.0f}.fit'.format( flat_1arcsec_bri_framen[i] ) ]

        flat_10arcsec_dim_framen = range( 1906257, 1906306+1 )
        flat_10arcsec_dim_list_filename = 'domeflat_10arcsec_dim.lst'
        mflat_10arcsec_dim_filename = 'mdomeflat_10arcsec_dim.fit'
        nflat = len( flat_10arcsec_dim_framen )
        flat_10arcsec_dim_frames = []
        for i in range( nflat ):
            flat_10arcsec_dim_frames += [ 'r{0:.0f}.fit'.format( flat_10arcsec_dim_framen[i] ) ]

        flat_10arcsec_bri_framen = range( 1906207, 1906256+1 )
        flat_10arcsec_bri_list_filename = 'domeflat_10arcsec_bri.lst'
        mflat_10arcsec_bri_filename = 'mdomeflat_10arcsec_bri.fit'
        nflat = len( flat_10arcsec_bri_framen )
        flat_10arcsec_bri_frames = []
        for i in range( nflat ):
            flat_10arcsec_bri_frames += [ 'r{0:.0f}.fit'.format( flat_10arcsec_bri_framen[i] ) ]

        # Argon arcs start of night:
        arc_Ar1_dim_framen = range( 1906856, 1906857+1 )
        arc_Ar1_dim_list_filename = 'arc_Ar1_dim.lst'
        marc_Ar1_dim_filename = 'marc_Ar1_dim.fit'
        narc = len( arc_Ar1_dim_framen )
        arc_Ar1_dim_frames = []
        for i in range( narc ):
            arc_Ar1_dim_frames += [ 'r{0:.0f}.fit'.format( arc_Ar1_dim_framen[i] ) ]
        arc_Ar1_bri_framen = range( 1906858, 1906859+1 )
        arc_Ar1_bri_list_filename = 'arc_Ar1_bri.lst'
        marc_Ar1_bri_filename = 'marc_Ar1_bri.fit'
        narc = len( arc_Ar1_bri_framen )
        arc_Ar1_bri_frames = []
        for i in range( narc ):
            arc_Ar1_bri_frames += [ 'r{0:.0f}.fit'.format( arc_Ar1_bri_framen[i] ) ]

        # Argon arcs end of night:
        arc_Ar2_dim_framen = range( 1907383, 1907384+1 )
        arc_Ar2_dim_list_filename = 'arc_Ar2_dim.lst'
        marc_Ar2_dim_filename = 'marc_Ar2_dim.fit'
        narc = len( arc_Ar2_dim_framen )
        arc_Ar2_dim_frames = []
        for i in range( narc ):
            arc_Ar2_dim_frames += [ 'r{0:.0f}.fit'.format( arc_Ar2_dim_framen[i] ) ]
        arc_Ar2_bri_framen = range( 1907385, 1907386+1 )
        arc_Ar2_bri_list_filename = 'arc_Ar2_bri.lst'
        marc_Ar2_bri_filename = 'marc_Ar2_bri.fit'
        narc = len( arc_Ar2_bri_framen )
        arc_Ar2_bri_frames = []
        for i in range( narc ):
            arc_Ar2_bri_frames += [ 'r{0:.0f}.fit'.format( arc_Ar2_bri_framen[i] ) ]

        # Xenon arcs start of night:
        arc_Xe1_dim_framen = range( 1906860, 1906861+1 )
        arc_Xe1_dim_list_filename = 'arc_Xe1_dim.lst'
        marc_Xe1_dim_filename = 'marc_Xe1_dim.fit'
        narc = len( arc_Xe1_dim_framen )
        arc_Xe1_dim_frames = []
        for i in range( narc ):
            arc_Xe1_dim_frames += [ 'r{0:.0f}.fit'.format( arc_Xe1_dim_framen[i] ) ]
        arc_Xe1_bri_framen = range( 1906862, 1906863+1 )
        arc_Xe1_bri_list_filename = 'arc_Xe1_bri.lst'
        marc_Xe1_bri_filename = 'marc_Xe1_bri.fit'
        narc = len( arc_Xe1_bri_framen )
        arc_Xe1_bri_frames = []
        for i in range( narc ):
            arc_Xe1_bri_frames += [ 'r{0:.0f}.fit'.format( arc_Xe1_bri_framen[i] ) ]

        # Xenon arcs end of night:
        arc_Xe2_dim_framen = range( 1907388, 1907389+1 )
        arc_Xe2_dim_list_filename = 'arc_Xe2_dim.lst'
        marc_Xe2_dim_filename = 'marc_Xe2_dim.fit'
        narc = len( arc_Xe2_dim_framen )
        arc_Xe2_dim_frames = []
        for i in range( narc ):
            arc_Xe2_dim_frames += [ 'r{0:.0f}.fit'.format( arc_Xe2_dim_framen[i] ) ]
        arc_Xe2_bri_framen = range( 1907390, 1907391+1 )
        arc_Xe2_bri_list_filename = 'arc_Xe2_bri.lst'
        marc_Xe2_bri_filename = 'marc_Xe2_bri.fit'
        narc = len( arc_Xe2_bri_framen )
        arc_Xe2_bri_frames = []
        for i in range( narc ):
            arc_Xe2_bri_frames += [ 'r{0:.0f}.fit'.format( arc_Xe2_bri_framen[i] ) ]

    elif night=='':
        pdb.set_trace()#todo
    elif night=='':
        pdb.set_trace()#todo
    else:
        pdb.set_trace() # not recognised

    # Create master dark:
    dark_list_filepath = os.path.join( adir_full, dark_list_filename )
    np.savetxt( dark_list_filepath, np.array( dark_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( dark_list_filepath, raw_ddir_full, ddir_dark, \
                                     mdark_filename, frame_type='dark', n_exts=N_EXTS )

    # Create master dim flat with 1 arcsec slit:
    flat_1arcsec_dim_list_filepath = os.path.join( adir_full, flat_1arcsec_dim_list_filename )
    np.savetxt( flat_1arcsec_dim_list_filepath, np.array( flat_1arcsec_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( flat_1arcsec_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     mflat_1arcsec_dim_filename, frame_type='dflatdim', n_exts=N_EXTS )

    # Create master bright flat with 1 arcsec slit:
    flat_1arcsec_bri_list_filepath = os.path.join( adir_full, flat_1arcsec_bri_list_filename )
    np.savetxt( flat_1arcsec_bri_list_filepath, np.array( flat_1arcsec_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( flat_1arcsec_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     mflat_1arcsec_bri_filename, frame_type='dflatbri', n_exts=N_EXTS )

    # Create master dim flat with 10 arcsec slit:
    flat_10arcsec_dim_list_filepath = os.path.join( adir_full, flat_10arcsec_dim_list_filename )
    np.savetxt( flat_10arcsec_dim_list_filepath, np.array( flat_10arcsec_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( flat_10arcsec_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     mflat_10arcsec_dim_filename, frame_type='dflatdim', n_exts=N_EXTS )

    # Create master bright flat with 10 arcsec slit:
    flat_10arcsec_bri_list_filepath = os.path.join( adir_full, flat_10arcsec_bri_list_filename )
    np.savetxt( flat_10arcsec_bri_list_filepath, np.array( flat_10arcsec_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( flat_10arcsec_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     mflat_10arcsec_bri_filename, frame_type='dflatbri', n_exts=N_EXTS )

    # Create master dim Ar arc from start of night:
    arc_Ar1_dim_list_filepath = os.path.join( adir_full, arc_Ar1_dim_list_filename )
    np.savetxt( arc_Ar1_dim_list_filepath, np.array( arc_Ar1_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Ar1_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Ar1_dim_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master bright Ar arc from start of night:
    arc_Ar1_bri_list_filepath = os.path.join( adir_full, arc_Ar1_bri_list_filename )
    np.savetxt( arc_Ar1_bri_list_filepath, np.array( arc_Ar1_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Ar1_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Ar1_bri_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master dim Ar arc from end of night:
    arc_Ar2_dim_list_filepath = os.path.join( adir_full, arc_Ar2_dim_list_filename )
    np.savetxt( arc_Ar2_dim_list_filepath, np.array( arc_Ar2_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Ar2_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Ar2_dim_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master bright Ar arc from end of night:
    arc_Ar2_bri_list_filepath = os.path.join( adir_full, arc_Ar2_bri_list_filename )
    np.savetxt( arc_Ar2_bri_list_filepath, np.array( arc_Ar2_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Ar2_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Ar2_bri_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master dim Xe arc from start of night:
    arc_Xe1_dim_list_filepath = os.path.join( adir_full, arc_Xe1_dim_list_filename )
    np.savetxt( arc_Xe1_dim_list_filepath, np.array( arc_Xe1_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Xe1_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Xe1_dim_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master bright Xe arc from start of night:
    arc_Xe1_bri_list_filepath = os.path.join( adir_full, arc_Xe1_bri_list_filename )
    np.savetxt( arc_Xe1_bri_list_filepath, np.array( arc_Xe1_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Xe1_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Xe1_bri_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master dim Xe arc from end of night:
    arc_Xe2_dim_list_filepath = os.path.join( adir_full, arc_Xe2_dim_list_filename )
    np.savetxt( arc_Xe2_dim_list_filepath, np.array( arc_Xe2_dim_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Xe2_dim_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Xe2_dim_filename, frame_type='arc_ar', n_exts=N_EXTS )

    # Create master bright Xe arc from end of night:
    arc_Xe2_bri_list_filepath = os.path.join( adir_full, arc_Xe2_bri_list_filename )
    np.savetxt( arc_Xe2_bri_list_filepath, np.array( arc_Xe2_bri_frames, dtype=str ), fmt='%s' )
    reduction.median_combine_frames( arc_Xe2_bri_list_filepath, raw_ddir_full, ddir_flat, \
                                     marc_Xe2_bri_filename, frame_type='arc_ar', n_exts=N_EXTS )


    # Now create the final master files by subtracting the dims from the brights:

    dim_ifilepath = os.path.join( ddir_flat, mflat_10arcsec_dim_filename )
    bri_ifilepath = os.path.join( ddir_flat, mflat_10arcsec_bri_filename )
    ofilepath = os.path.join( ddir_flat, 'mdomeflat_10arcsec_final.fit' )
    reduction.subtract_dim_from_bright( dim_ifilepath, bri_ifilepath, ofilepath, frame_type='dflat', n_exts=N_EXTS )

    dim_ifilepath = os.path.join( ddir_flat, mflat_1arcsec_dim_filename )
    bri_ifilepath = os.path.join( ddir_flat, mflat_1arcsec_bri_filename )
    ofilepath = os.path.join( ddir_flat, 'mdomeflat_1arcsec_final.fit' )
    reduction.subtract_dim_from_bright( dim_ifilepath, bri_ifilepath, ofilepath, frame_type='dflat', n_exts=N_EXTS )

    dim_ifilepath = os.path.join( ddir_flat, marc_Ar1_dim_filename )
    bri_ifilepath = os.path.join( ddir_flat, marc_Ar1_bri_filename )
    ofilepath = os.path.join( ddir_flat, 'marc_Ar1_final.fit' )
    reduction.subtract_dim_from_bright( dim_ifilepath, bri_ifilepath, ofilepath, frame_type='arc_ar', n_exts=N_EXTS )

    dim_ifilepath = os.path.join( ddir_flat, marc_Ar2_dim_filename )
    bri_ifilepath = os.path.join( ddir_flat, marc_Ar2_bri_filename )
    ofilepath = os.path.join( ddir_flat, 'marc_Ar2_final.fit' )
    reduction.subtract_dim_from_bright( dim_ifilepath, bri_ifilepath, ofilepath, frame_type='arc_ar', n_exts=N_EXTS )

    return None


def calibrate_raw_science( night='' ):

    adir_full = os.path.join( ADIR, night )
    raw_ddir_full = os.path.join( RAW_DDIR, '{0}/raw'.format( night ) )
    cal_ddir_full = os.path.join( CAL_DDIR, night )
    ddir_dark = os.path.join( cal_ddir_full, 'dark' )

    # Create the list of raw science frames:
    if night=='20130517':
        framen = range( 1906913, 1907342+1 )
    elif night=='20130216':
        pdb.set_trace() #todo
    elif night=='20130701':
        pdb.set_trace() #todo
    nframes = len( framen )
    raw_science_frames = []
    for i in range( nframes ):
        raw_science_frames += [ 'r{0:.0f}.fit'.format( framen[i] ) ]
    raw_science_list_filename = 'raw_science.lst'
    raw_science_list_filepath = os.path.join( adir_full, raw_science_list_filename )
    np.savetxt( raw_science_list_filepath, np.array( raw_science_frames, dtype=str ), fmt='%s' )

    # Call the calibration routine:
    mdark_filepath = os.path.join( ddir_dark, 'mdark.fit' )
    mflat_filepath = None
    reduction.calibrate_raw_science( n_exts=N_EXTS, raw_ddir=raw_ddir_full, cal_ddir=cal_ddir_full, \
                                     mbias_filepath=mdark_filepath, mflat_filepath=mflat_filepath, \
                                     raw_science_list_filepath=raw_science_list_filepath )

    return None


def prep_stellar_obj( night='' ):
    #todo + include call to combine_spectra() in reduction.py
    adir_full = os.path.join( ADIR, night )
    #adir = '/home/tevans/analysis/gtc/hatp19'
    science_images_full_list_filename = 'cal_science_noflatfield.lst'
    badpix_maps_full_list_filename = 'badpix_full.lst'
    science_images_list_filename = 'science_images.lst' # should apply to all stars...
    badpix_maps_list_filename = 'badpix_maps.lst' # should apply to all stars...
    science_traces_list_filename = [ 'science_traces_reference.lst', 'science_traces_hatp19.lst' ]
    science_spectra_list_filename = [ 'science_spectra_reference.lst', 'science_spectra_hatp19.lst' ]
    ddir_science = os.path.join( ddir, 'object' )
    ddir_arc = os.path.join( ddir, 'arc' )
    star_names = [ 'reference',  'hatp19' ]
    disp_axis = 0
    crossdisp_axis = 0
    crossdisp_bounds = [ [ 150, 350 ], [ 510, 710 ] ]
    disp_bounds = [ [ 400, 2050 ], [ 400, 2050 ] ]
    stellar = reduction.prep_stellar_obj( adir=adir, \
                                          science_images_full_list_filename=science_images_full_list_filename, \
                                          badpix_maps_full_list_filename=badpix_maps_full_list_filename, \
                                          science_images_list_filename=science_images_list_filename, \
                                          badpix_maps_list_filename=badpix_maps_list_filename, \
                                          science_traces_list_filename=science_traces_list_filename, \
                                          science_spectra_list_filename=science_spectra_list_filename, \
                                          ddir_science=ddir_science, ddir_arc=ddir_arc, n_exts=N_EXTS, \
                                          star_names=star_names, \
                                          disp_axis=disp_axis, crossdisp_axis=crossdisp_axis, \
                                          crossdisp_bounds=crossdisp_bounds, disp_bounds=disp_bounds )
    return stellar


def run_all():
    #identify_bad_pixels()
    fit_traces()
    extract_spectra( spectral_ap_radius=15, sky_inner_radius=25, sky_band_width=5 )
    return None

def identify_bad_pixels():
    stellar = prep_stellar_obj()
    stellar.identify_bad_pixels()
    return None

def fit_traces():
    stellar = prep_stellar_obj()
    stellar.fit_traces( make_plots=True )
    return None

def extract_spectra( spectral_ap_radius=30, sky_inner_radius=45, sky_band_width=5 ):
    stellar = prep_stellar_obj()
    stellar.spectral_ap_radius = spectral_ap_radius
    stellar.sky_inner_radius = sky_inner_radius
    stellar.sky_band_width = sky_band_width
    stellar.extract_spectra()
    return None

def combine_spectra():
    # todo = add something to this routines that extracts
    # auxiliary variables, including opening each image and
    # reading stuff out of the headers like time stamp.
    stellar = prep_stellar_obj()
    image_list_filepath = os.path.join( stellar.adir, stellar.science_images_list )
    image_list = np.loadtxt( image_list_filepath, dtype=str )
    nimages = len( image_list )
    hatp19_spectra_list_filepath = os.path.join( stellar.adir, stellar.science_spectra_list[0] )
    hatp19_spectra_list = np.loadtxt( hatp19_spectra_list_filepath, dtype=str )
    if len( hatp19_spectra_list )!=nimages:
        pdb.set_trace()
    ref_spectra_list_filepath = os.path.join( stellar.adir, stellar.science_spectra_list[1] )
    ref_spectra_list = np.loadtxt( ref_spectra_list_filepath, dtype=str )    
    if len( ref_spectra_list )!=nimages:
        pdb.set_trace()
    mjds = []
    hatp19_spectra = []
    hatp19_disp_pixs = []
    hatp19_skyppix = []
    hatp19_nappixs = []
    hatp19_fwhm = []
    ref_spectra = []
    ref_disp_pixs = []
    ref_skyppix = []
    ref_nappixs = []
    ref_fwhm = []    
    for i in range( nimages ):
        print '... reading spectrum {0} of {1}'.format( i+1, nimages )
        image_filepath = os.path.join( stellar.ddir, image_list[i] )
        hdu = fitsio.FITS( image_filepath )
        h0 = hdu[0].read_header()
        mjds += [ h0['MJD-OBS'] ]
        hatp19_spectrum_filepath = os.path.join( stellar.adir, hatp19_spectra_list[i] )
        hatp19_spectrum = fitsio.FITS( hatp19_spectrum_filepath )
        hatp19_spectra += [ hatp19_spectrum[1].read_column( 'apflux' ) ]
        hatp19_disp_pixs += [ hatp19_spectrum[1].read_column( 'disp_pixs' ) ]
        hatp19_skyppix += [ hatp19_spectrum[1].read_column( 'skyppix' ) ]
        hatp19_nappixs += [ hatp19_spectrum[1].read_column( 'nappixs' ) ]
        hatp19_fwhm += [ hatp19_spectrum[1].read_header()['FWHM'] ]
        hatp19_spectrum.close()
        ref_spectrum_filepath = os.path.join( stellar.adir, ref_spectra_list[i] )
        ref_spectrum = fitsio.FITS( ref_spectrum_filepath )
        ref_spectra += [ ref_spectrum[1].read_column( 'apflux' ) ]
        ref_disp_pixs += [ ref_spectrum[1].read_column( 'disp_pixs' ) ]
        ref_skyppix += [ ref_spectrum[1].read_column( 'skyppix' ) ]
        ref_nappixs += [ ref_spectrum[1].read_column( 'nappixs' ) ]
        ref_fwhm += [ ref_spectrum[1].read_header()['FWHM'] ]
        ref_spectrum.close()
        #print 'aaa'
        #plt.figure()
        #plt.plot(hatp19_spectra[-1],'-b')
        #plt.plot(ref_spectra[-1],'-r')
        #pdb.set_trace()
        #print 'bbb'
    mjds = np.array( mjds )
    jds = mjds + 2400000.5
    tmins = ( jds-jds.min() )*24.*60.
    hatp19_spectra = np.row_stack( hatp19_spectra )
    hatp19_disp_pixs = np.row_stack( hatp19_disp_pixs )
    hatp19_skyppix = np.row_stack( hatp19_skyppix )
    hatp19_nappixs = np.row_stack( hatp19_nappixs )
    hatp19_fwhm = np.array( hatp19_fwhm )
    
    ref_spectra = np.row_stack( ref_spectra )
    ref_disp_pixs = np.row_stack( ref_disp_pixs )
    ref_skyppix = np.row_stack( ref_skyppix )
    ref_nappixs = np.row_stack( ref_nappixs )
    ref_fwhm = np.array( ref_fwhm )
    
    y1 = np.sum( hatp19_spectra, axis=1 )
    y2 = np.sum( ref_spectra, axis=1 )
    plt.figure()
    plt.subplot( 311 )
    ax1 = plt.gca()
    plt.plot( tmins, y1, '.r' )
    plt.plot( tmins, y2, '.b' )
    plt.subplot( 312, sharex=ax1 )
    plt.plot( tmins, y1/y2, '.k' )
    plt.subplot( 313, sharex=ax1 )
    plt.plot( tmins, hatp19_fwhm, '-r' )
    plt.plot( ref_fwhm, '-b' )
    plt.figure()
    plt.plot( tmins, y2/y1, '.k' )
    plt.title('HAT-P-19')
    plt.ylabel('Relative Flux')
    plt.xlabel('Time (minutes)')
    pdb.set_trace()
    return None
    
    
    
