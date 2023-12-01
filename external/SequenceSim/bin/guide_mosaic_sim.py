# Program to build guide image array mosaics for DES


# Python imports
import os
import time
import random
import threading

# External imports
import numpy
import pyfits

class guide_mosaic_sim:
    
    def __init__(self):
        print "Initializing simulation..."
        self.config = {
        'out_dir'   :   '/Users/santi/Desktop/raw_guider',
        'seq_len'   :   100,
        'basename'  :   'guider_sim',
        'ccd_size'  :   [2048,2048],
        'scale'     :   0.27,
        'turbulence_sigma'  :   0.2,
        'image_bias':   200
        }
        self.image_paths = [None, None, None, None]
        self.true_offset = [0.0, 0.0]
        
        if not os.path.exists(self.config['out_dir'] + '/tmp/'):
            os.mkdir(self.config['out_dir'] + '/tmp/')
    
    def simulate_guide_sequence(self):
        # Initialize Offsets file
        offsets_file = open(os.path.join(self.config['out_dir'], 'offsets.dat'), 'w')
        offsets_file.write('#seq\tXoff\tYoff\n')
        
        for self.guide_iteration in range(self.config['seq_len']):
            print "Simulating guide mosaic sequence %d..." %self.guide_iteration
            image_name = self.config['basename'] + '_seq_%03d'%self.guide_iteration + '.fits'
            self.add_telescope_error()
            self.simulate_guide_array(image_name)
            offsets_file.write("%d\t%.3f\t%.3f\n" %(self.guide_iteration, self.true_offset[0], self.true_offset[1]))
            
            # Assume perfect guiding correction
            self.true_offset = [0.0, 0.0]
            print "...mosaic complete."
            
        offsets_file.close()
        print "Simulation complete!"
    
    
    def simulate_guide_array(self, image_name):
        # Launch simulation threads
        th_array = []
        for iCCD in range(4):
            ccd_th = threading.Thread(target=self.simulate_guide_image, args=(iCCD, ))
            ccd_th.start()
            th_array.append(ccd_th)
        
        # Wait for CCD threads completion
        for ccd_th in th_array:
            ccd_th.join()
        
        # Build the Fits multi extension
        # Initialize the fits array
        mosaic_path = os.path.join(self.config['out_dir'], image_name)
        if os.path.exists(mosaic_path):
            os.remove(mosaic_path)
        pyfits.writeto(mosaic_path,
                       data = numpy.array([], dtype = 'uint8'),
                       header = pyfits.Header(),
                       clobber = 'True')
        # Append Guide CCDs
        for iCCD in range(4):
            ccd_hdu = pyfits.open(os.path.join(self.config['out_dir'], 'tmp', self.image_paths[iCCD]))
            
            # AMP A
            ccd_data_A = numpy.zeros([2048, 1080], dtype = 'uint16') + self.config['image_bias']
            ccd_data_A[:, 6:1030] =  (ccd_hdu[0].data[:, :1024] + self.config['image_bias']).astype('uint16')
            pyfits.append(mosaic_path, ccd_data_A, ccd_hdu[0].header)
            
            # AMP B
            ccd_data_B = numpy.zeros([2048, 1080], dtype = 'uint16') + self.config['image_bias']
            ccd_data_B[:, 50:1074] =  (ccd_hdu[0].data[:, 1024:] + self.config['image_bias']).astype('uint16')
            pyfits.append(mosaic_path, ccd_data_B, ccd_hdu[0].header)
            
    
    
    def simulate_guide_image(self,iCCD):
        # Single CCD Image Simulation
        #print "[CCD %d] Image simulation threaded."%iCCD
        try:
            # Full images 
            roi_radius = 0
            roi_center = [0,0]
            sim_cmd = ( "./simulate_c " + "sim.params " +
                        " %d %d %d %d %d %d %d %f %f %s" %(self.config['ccd_size'][0],
                                                           self.config['ccd_size'][1],
                                                           self.guide_iteration,
                                                           roi_radius,
                                                           roi_center[0],
                                                           roi_center[1],
                                                           iCCD,
                                                           self.true_offset[0],
                                                           self.true_offset[1],
                                                           self.config['out_dir'] ))
            os.system(sim_cmd)
        except:
            # Stop guiding & exit
            print "[CCD %d] Guide Image Simulation error." %iCCD
            return -1
        
        
        #print "[CCD %d] ...guide image complete!" %iCCD
        
        # Image path+file name
        self.image_paths[iCCD] = "decam-guide_ccd%d.fits"%iCCD
        
    
    
    def add_telescope_error(self):
        # Simulates telescope gaussian error (for Simulation mode only)
        if self.guide_iteration == 0:
            # No error for the first iteration
            self.true_offset = [0.0, 0.0]
        else:
            # Error is added to previous correction
            self.true_offset = [random.gauss(self.true_offset[0],self.config['turbulence_sigma']/self.config['scale']),
                                random.gauss(self.true_offset[1],self.config['turbulence_sigma']/self.config['scale'])]
            print 'New simulated telescope offset is [%.5f, %.5f] (pixels)' %(self.true_offset[0], self.true_offset[1])
            
    def thread_os_cmd(self, in_cmd):
        os.system(in_cmd)
        

gms = guide_mosaic_sim()
gms.simulate_guide_sequence()