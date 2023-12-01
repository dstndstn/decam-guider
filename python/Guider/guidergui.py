#!/usr/bin/env python
from SISPIlib.application import Application
import SISPIlib.discovery as discovery

# GUI imports
from Tkinter import *
import tkMessageBox
import pyfits


# PML & SVE imports
import PML.core
#from sve.pythonclient import *

# Standard imports
import os
import time
import math


class GUI(Application):
    # The default role name
    role = 'GUIDERGUI'
    component = 'GUIDERGUI'
    commands = ['configure', 'connect_guider', 'update']
    # Set of defaults, can be accessed through self.config
    defaults = {'window_width' : 1000,                   # Window width size (pix)
                'window_height' : 680,                   # Window height size (pix)
                'screen_size' : 512,                     # Image Screen size
                'ini_pos_h' : 50,                        # Window horizontal initial position
                'ini_pos_v' : 50,                        # Window vertical initial position
                'guide_ccds' : ['GS1','GS2','GN1','GN2'],    # Guide CCD names [0,1,2,3] respectively
                'gamma_contrast': 4                      # Gamma value for contrast mode
                }
    
    def init(self):
        # Initialize paths
        # Use GUIDER's working directory
        self.working_directory = self.application_working_directory().replace('GUIDERGUI','GUIDER')
        # Get the working directories
        self.tmp_directory = os.path.join(self.working_directory,"gui")
        # Check & build directories structure
        if not os.path.exists(self.tmp_directory):
            os.makedirs(self.tmp_directory)
        # Go to bin path
        self.product_path = os.getenv('GUIDER_DIR')
        os.chdir(self.product_path+"")
        
        # Init font types
        self.text_normal = "Arial 9"
        self.text_highlight = "Arial 9 bold"
        self.text_small = "Arial 8"
        self.text_small_highlight = "Arial 8 bold"
        
        # Internal Positions and sizes
        self.centroids_evo_width = 425      # Centroids evolution plot width (pix)
        self.centroids_evo_height = 180     # Centroids evolution plot height (pix)
        self.x_plot_start = 40              # Initial x axis position at the plot (in pixels)
        self.y_max_value = 2.0              # Y axis max value for plotting (in arcsec)
        self.x_axis_step = 5                # Number of pixels per iteration at the plot
        self.star_box_size = 10             # Size of star boxes (pix)
        
        # GUI specific inits
        self.enable_manual_selection = False
        self.manual_x_pos = -1
        self.manual_y_pos = -1
        self.ccd_image_size = None
        
        # Initialize root Frame
        self.root = Tk()
        self.root.title("DES Guider Tk v0.02")
        self.root.geometry("%dx%d%+d%+d" % (self.config['window_width'], 
                                       self.config['window_height'],
                                       self.config['ini_pos_h'],
                                       self.config['ini_pos_v']))
        self.root.resizable(width = FALSE, height = FALSE)
        self.pad = 10
        self.frame_width = self.config['window_width'] - (512 + 4 * self.pad)
        self.frame_height = self.config['window_height'] - 4 * self.pad
        
        # Construct root frame
        self.construct_root_frame()
        # make the Guider GUI discoverable
        discovery.discoverable(role = self.role, tag = 'GUIDERGUI', component=self.component)
        
    def configure(self, param = ''):
        self.info('Configuring...')
        self.connect_guider()
        self.info('Configured')
        return self.SUCCESS
    
    def update(self, param = ''):
        """
        Update display
        """
        self.load_current_config()
        return self.SUCCESS
        
    def connect_guider(self, param=''):
        self.debug("User asked to connect.")
        self.connected = False
        
        # Check if guider already connected
        try:
            g_status = self.shared_guiderstatus._value
            self.info("Guider already connected. Guider status is: %s" %g_status)
        except:
            self.info("Guider not connected yet. Configuring PML and SVE...")
            # Configure PML connection with Guider
            self.guider_pml = PML.core.PML_Connection("GUIDER", "GUIDER")
            # Subscribe to SVE Shared Variables
            """ Guider Status """
            self.shared_guiderstatus = self.shared_variable(name = "GUIDERSTATUS", group = "GUIDER")
            self.shared_guiderstatus.subscribe(callback = self.event_guiderstatus)
            """ CCDs (distances in pixels) """
            self.shared_ccds = []
            for i in range( len( self.config['guide_ccds'] ) ):
                self.shared_ccds.append( self.shared_variable(name = "CCD%d"%i, group = "GUIDER") )
                self.shared_ccds[i].subscribe(callback = self.event_ccd)
            """ Output Centroid (in arcsecs) """
            self.shared_centroid = self.shared_variable(name = "CENTROID", group = "GUIDER")
            self.shared_centroid.subscribe(callback = self.event_centroid_out)
        
        # Update to current status
        current_status = self.shared_guiderstatus._value
        self.info("Current GUIDER status is %s" %current_status)
        self.label_guiderstatus.config(text = "Guider status: " + current_status)
        if current_status == "READY":
#            self.btn_connect_guider.config(text = "Connected", state = DISABLED)
            self.connected = True
            # Enable all configuration
            self.subframe_ccds_1N_enable.config(state = NORMAL)
            self.subframe_ccds_2N_enable.config(state = NORMAL)
            self.subframe_ccds_1S_enable.config(state = NORMAL)
            self.subframe_ccds_2S_enable.config(state = NORMAL)
            self.subframe_ccds_1N_auto.config(state = NORMAL)
            self.subframe_ccds_2N_auto.config(state = NORMAL)
            self.subframe_ccds_1S_auto.config(state = NORMAL)
            self.subframe_ccds_2S_auto.config(state = NORMAL)
            self.subframe_ccds_1N_man.config(state = NORMAL)
            self.subframe_ccds_2N_man.config(state = NORMAL)
            self.subframe_ccds_1S_man.config(state = NORMAL)
            self.subframe_ccds_2S_man.config(state = NORMAL)
            self.subframe_mode_auto_radio.config(state = NORMAL)
            self.subframe_mode_self_radio.config(state = NORMAL)
            self.subframe_mode_user_radio.config(state = NORMAL)
            #self.subframe_tcs_enable.config(state = NORMAL) # Currently configured from OCS
            self.subframe_tcs_feedback_label.config(state = NORMAL)
            self.subframe_tcs_feedback_entry.config(state = NORMAL)
            self.subframe_tcs_min_latency_time_label.config(state = NORMAL)
            self.subframe_tcs_min_latency_time_entry.config(state = NORMAL)
            
            # Update with current status
            self.event_guiderstatus(event = None)
            self.load_current_config()
            self.debug("Guider GUI successfully connected to Guider.")
    
    # Frame constructors & withdraws
    def construct_root_frame(self):
        # Labels
        self.label_guiderstatus = Label(self.root, text = "Guider status: ? ", font = self.text_highlight )
        self.label_obs_displayccd = Label(self.root, text = "Display CCD:", font = self.text_small_highlight)
        self.label_obs_displaymode = Label(self.root, text = "Display mode:", font = self.text_small_highlight)
        
        # Buttons
        self.btn_connect_guider = Button(self.root, text = "Connect Guider", width = 12, command = self.connect_guider)
        
        # Radio Buttons
        """ Views """
        self.var_fdisplay = IntVar()
        self.radio_fdisplay_obs = Radiobutton(self.root,
                                               text = "Observer View",
                                               variable = self.var_fdisplay,
                                               value = 0,
                                               font = self.text_normal,
                                               command = self.cmd_frame_display)
        self.radio_fdisplay_stars = Radiobutton(self.root,
                                               text = "Guide Stars View",
                                               variable = self.var_fdisplay,
                                               value = 1,
                                               font = self.text_normal,
                                               command = self.cmd_frame_display)
        self.radio_fdisplay_ccds = Radiobutton(self.root,
                                               text = "Guide CCDs View",
                                               variable = self.var_fdisplay,
                                               value = 2,
                                               font = self.text_normal,
                                               command = self.cmd_frame_display)
        self.radio_fdisplay_conf = Radiobutton(self.root,
                                                text = "Configuration View",
                                                variable = self.var_fdisplay,
                                                value = 3,
                                                font = self.text_normal,
                                                command = self.cmd_frame_display)
        self.radio_fdisplay_obs.select()
        
        """ Single/Multi Display """
        self.var_multidisplay = BooleanVar()
        self.radio_obs_display_single = Radiobutton(self.root,
                                               text = "Single",
                                               variable = self.var_multidisplay,
                                               value = False,
                                               font = self.text_small,
                                               command = self.cmd_config_multidisplay)
                                               
        self.radio_obs_display_multi = Radiobutton(self.root,
                                               text = "Multi",
                                               variable = self.var_multidisplay,
                                               value = True,
                                               font = self.text_small,
                                               command = self.cmd_config_multidisplay)
        
        """ CCD Display (Single only) """
        self.var_displayccd = IntVar()
        self.radio_obs_display_ccd1S = Radiobutton(self.root,
                                               text = "CCD GS1",
                                               variable = self.var_displayccd,
                                               value = 0,
                                               font = self.text_small)
                                               
        self.radio_obs_display_ccd2S = Radiobutton(self.root,
                                               text = "CCD GS2",
                                               variable = self.var_displayccd,
                                               value = 1,
                                               font = self.text_small)
        self.radio_obs_display_ccd1N = Radiobutton(self.root,
                                               text = "CCD GN1",
                                               variable = self.var_displayccd,
                                               value = 2,
                                               font = self.text_small)
                                               
        self.radio_obs_display_ccd2N = Radiobutton(self.root,
                                               text = "CCD GN2",
                                               variable = self.var_displayccd,
                                               value = 3,
                                               font = self.text_small)
        
        # Check Buttons
        self.var_obs_gamma = BooleanVar()
        self.check_obs_gamma = Checkbutton (self.root,
                                           text = "Gamma Contrast",
                                           variable = self.var_obs_gamma, 
                                           font = self.text_small, 
                                           state = NORMAL)
        
        # Canvas
        """ Main Screen """
        self.main_screen = Canvas(self.root ,bg = "grey", width = 512 + self.pad, height = 512 + self.pad)
        """ >> Full Image """
        self.full_screen = PhotoImage( file = "external/media/welcome.gif" ) # Load Welcome Image
        self.main_screen.create_image( (self.config['screen_size'] + self.pad)/2 + 1,
                                       (self.config['screen_size'] + self.pad)/2 + 1,
                                       image = self.full_screen,
                                       tag = "full_screen",
                                       state = NORMAL) # Add to Canvas
        """ >> Multi image """
        multi_view_initial_path = "external/media/welcome_256.gif"
        self.multi_screen_1S = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
        self.multi_screen_2S = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
        self.multi_screen_1N = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
        self.multi_screen_2N = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
        self.main_screen.create_image( (self.config['screen_size'] + self.pad)/4 + 2,
                                      (self.config['screen_size'] + self.pad)/4 + 2,
                                      image = self.multi_screen_1S,
                                      tag = "multi_screen_1S",
                                      state = HIDDEN) # Add to Canvas
        self.main_screen.create_image( 3* (self.config['screen_size'] + self.pad)/4,
                                       (self.config['screen_size'] + self.pad)/4 + 2,
                                       tag = "multi_screen_2S",
                                       image = self.multi_screen_2S,
                                       state = HIDDEN) # Add to Canvas
        self.main_screen.create_image( (self.config['screen_size'] + self.pad)/4 + 2,
                                       3* (self.config['screen_size'] + self.pad)/4 ,
                                       tag = "multi_screen_1N",
                                       image = self.multi_screen_1N,
                                       state = HIDDEN) # Add to Canvas
        self.main_screen.create_image( 3* (self.config['screen_size'] + self.pad)/4,
                                       3*(self.config['screen_size'] + self.pad)/4,
                                       tag = "multi_screen_2N",
                                       image = self.multi_screen_2N,
                                       state = HIDDEN,) # Add to Canvas
        
        self.main_screen.create_text( (self.config['screen_size'] + self.pad)/4 + 2,
                                      (self.config['screen_size'] + self.pad)/4 + 2 - 12 * self.pad,
                                      tag = "multi_screen_1S_label",
                                      text = "CCD GS1",
                                      fill = "#0A0",
                                      font = self.text_small_highlight,
                                      state = HIDDEN) # Add to Canvas
        self.main_screen.create_text( 3* (self.config['screen_size'] + self.pad)/4,
                                      (self.config['screen_size'] + self.pad)/4 + 2 - 12 * self.pad,
                                      tag = "multi_screen_2S_label",
                                      text = "CCD GS2",
                                      fill = "#0A0",
                                      font = self.text_small_highlight,
                                      state = HIDDEN) # Add to Canvas
        self.main_screen.create_text( (self.config['screen_size'] + self.pad)/4 + 2,
                                      3* (self.config['screen_size'] + self.pad)/4  - 12 * self.pad,
                                      tag = "multi_screen_1N_label",
                                      text = "CCD GN1",
                                      fill = "#0A0",
                                      font = self.text_small_highlight,
                                      state = HIDDEN) # Add to Canvas
        self.main_screen.create_text( 3* (self.config['screen_size'] + self.pad)/4,
                                      3*(self.config['screen_size'] + self.pad)/4 - 12 * self.pad,
                                      tag = "multi_screen_2N_label",
                                      text = "CCD GN2",
                                      fill = "#0A0",
                                      font = self.text_small_highlight,
                                      state = HIDDEN,) # Add to Canvas
                                      
        self.main_screen.bind( "<ButtonRelease-1>",  self.event_manual_selection )
        
        # Frames
        self.construct_obs_frame()
        self.construct_stars_frame()
        self.construct_ccds_frame()
        self.construct_conf_frame()
        
        # Packing & Positioning
        self.btn_connect_guider.place(x = self.pad ,
                                      y = self.config['window_height'] - self.pad,
                                      anchor = SW )
        self.radio_fdisplay_obs.place(x = 1 * self.config['window_width'] / 5,
                                      y = self.config['window_height'] - 3 *self.pad,
                                      anchor = SW)
        self.radio_fdisplay_ccds.place(x = 2 * self.config['window_width'] / 6 + self.pad,
                                      y = self.config['window_height'] - self.pad,
                                      anchor = SW)  
        self.radio_fdisplay_stars.place(x = 2 * self.config['window_width'] / 6 + self.pad,
                                      y = self.config['window_height'] - 3 * self.pad,
                                      anchor = SW)  
        self.radio_fdisplay_conf.place(x = 1 * self.config['window_width'] / 5,
                                       y = self.config['window_height'] - self.pad,
                                       anchor = SW)
        self.label_guiderstatus.place(x = self.pad,
                                      y = self.config['window_height'] - 5 * self.pad,
                                      anchor = SW)
        self.obs_frame.place(x = self.config['window_width'] - self.pad,
                             y = 2 * self.pad,
                             width = self.frame_width,
                             height = self.frame_height,
                             anchor = NE)
        self.main_screen.place(x = self.pad, y = 7 * self.pad, anchor = NW)
        self.label_obs_displaymode.place(x = 28 * self.pad, y = 2 * self.pad, anchor = NW)
        self.radio_obs_display_single.place(x = 36 * self.pad, y = 2 * self.pad, anchor = NW)
        self.radio_obs_display_multi.place(x = 42 * self.pad, y = 2 * self.pad, anchor = NW)
        self.check_obs_gamma.place(x = 36 * self.pad, y = 4 * self.pad, anchor = NW)
        self.label_obs_displayccd.place(x = self.pad, y = 2 * self.pad, anchor = NW)
        self.radio_obs_display_ccd1N.place(x = 9 * self.pad, y = 2 * self.pad, anchor = NW)
        self.radio_obs_display_ccd2N.place(x = 16 * self.pad, y = 2 * self.pad, anchor = NW)
        self.radio_obs_display_ccd1S.place(x = 9 * self.pad, y = 4 * self.pad, anchor = NW)
        self.radio_obs_display_ccd2S.place(x = 16 * self.pad, y = 4 * self.pad, anchor = NW)
        
    
    def construct_obs_frame(self):
        # Declare frame
        self.obs_frame = Frame(self.root, relief = "groove", bd = 2, takefocus=TRUE)
        
        # Labels
        self.label_obs_title = Label(self.obs_frame, text = "Observer Guider View",
                                     font = self.text_highlight, foreground = "#666")
        self.label_centr_out_x = Label(self.obs_frame,
                                       text="   -        ",
                                       font = self.text_small,
                                       relief = "groove",
                                       bd = 2,
                                       width = 9,
                                       anchor = E,
                                       foreground = "#00A")
        self.label_centr_out_y = Label(self.obs_frame,
                                       text="   -        ",
                                       font = self.text_small,
                                       relief = "groove",
                                       bd = 2,
                                       width = 9,
                                       anchor = E,
                                       foreground = "#A00")
        # X is DEC
        self.label_centr_out_x_title = Label(self.obs_frame,
                                       text="DEC Centroid",
                                       font = self.text_highlight,
                                       foreground = "#00A")
        # Y is RA
        self.label_centr_out_y_title = Label(self.obs_frame,
                                       text="RA Centroid",
                                       font = self.text_highlight,
                                       foreground = "#A00")
        
        
        # Canvas
        """ Centroid evolution """
        self.plot_centroid_evolution = Canvas(self.obs_frame,
                                         width = self.centroids_evo_width,
                                         height = self.centroids_evo_height,
                                         relief="flat",
                                         bd = 0)
        self.plot_centroid_evolution.create_rectangle(self.x_plot_start, 2,
                                                      self.centroids_evo_width,
                                                      self.centroids_evo_height - 15,
                                                      width = 1,
                                                      fill = "white",
                                                      tags = "base") #limits rectangle
        self.logo_bg = PhotoImage( file = "external/media/DES_white.gif" )
        self.plot_centroid_evolution.create_image((self.centroids_evo_width + self.x_plot_start + 2*self.pad ) / 2,
                                                  (self.centroids_evo_height - 15) / 2,
                                                  image = self.logo_bg)
        self.plot_centroid_evolution.create_line(self.x_plot_start,
                                                (self.centroids_evo_height - 15) / 2,
                                                self.centroids_evo_width,
                                                (self.centroids_evo_height - 15) / 2,
                                                width=1,
                                                fill="#BBBBBB",
                                                tags="base") #center line
        # Plot labels
        self.plot_centroid_evolution.create_text((self.centroids_evo_width + self.x_plot_start) / 2,
                                                 self.centroids_evo_height-5,
                                                 text = "Image number",
                                                 font = self.text_small,
                                                 tags = "base")
        self.plot_centroid_evolution.create_text(20, (self.centroids_evo_height - 15 ) / 2,
                                                 text = "arcsec", font = self.text_small, tags = "base")
        # Y axis max min values
        self.plot_centroid_evolution.create_text(20, 10, text = str(self.y_max_value), font = self.text_small, tags = "base")
        self.plot_centroid_evolution.create_text(20, self.centroids_evo_height - 15,
                                                 text = str(-self.y_max_value), font = self.text_small, tags = "base")
        # X axis labels
        self.plot_centroid_evolution.create_text(self.x_plot_start, self.centroids_evo_height - 5,
                                                 text = str(0), font = self.text_small, anchor = W,
                                                 tags = "x_label_min")
        self.plot_centroid_evolution.create_text(self.centroids_evo_width, self.centroids_evo_height - 5,
                                                 text = str(1 + (self.centroids_evo_width - self.x_plot_start) / self.x_axis_step),
                                                 font = self.text_small,
                                                 anchor = E,
                                                 tags = "x_label_max")
        # Title
        self.plot_centroid_evolution.create_text(48, 15,
                                                 text = "Centroids Evolution",
                                                 font = "Arial 10 bold",
                                                 fill = "#AAAAAA",
                                                 tags = "base",
                                                 anchor = W)
        self.plot_centroid_evolution.create_text(130, 145,
                                                 text = "Mean:\nDev:",
                                                 font = self.text_small,
                                                 fill = "#888888",
                                                 tags = "mean_dev", 
                                                 state = HIDDEN)
        
        #self.plot_centroid_evolution.create_text(130,145,text="Mean:\nDev:",font="Arial 8",fill="#888888",tags="mean_dev", state=HIDDEN)
        
        # Packing & Positioning
        self.label_obs_title.place(x = self.pad, y = self.pad, anchor = NW)
        self.plot_centroid_evolution.place(x = self.frame_width - self.pad, y = self.frame_height - 6 * self.pad, anchor = SE)
        self.label_centr_out_y_title.place(x = self.frame_width - 10*self.pad, y = self.frame_height - 4*self.pad, anchor = E)
        self.label_centr_out_x_title.place(x = self.frame_width - 2*self.pad, y = self.frame_height - 4*self.pad, anchor = E)
        self.label_centr_out_y.place(x = self.frame_width - 10*self.pad -3, y = self.frame_height - 2*self.pad, anchor = E)
        self.label_centr_out_x.place(x = self.frame_width - 2*self.pad -3, y = self.frame_height - 2*self.pad, anchor = E)
    
    def construct_stars_frame(self):
        # Declare frame
        self.stars_frame = Frame(self.root, relief = "groove", bd = 2, takefocus = TRUE)
        
        # Labels
        self.label_stars_title = Label(self.stars_frame, text = "Guide Stars View",
                                     font = self.text_highlight, foreground = "#666")
        
        # Buttons
        
        
        # Stars list
        
        " CCDGS1 "
        self.stars_list_frame_1S = Frame(self.stars_frame)
        self.stars_list_1S = Text (self.stars_list_frame_1S, bg = "white", width = 40, height = 9)
        self.stars_label_top_1S = Label (self.stars_list_frame_1S,text="Star       S/N            " + 
                                                                 "Flux      Mag       FWHM        ")
        self.stars_label_side_1S = Label (self.stars_list_frame_1S,text="CCDGS1 ")
        self.scrollbar_stars_1S = Scrollbar(self.stars_list_frame_1S)
        self.stars_list_1S.config(yscrollcommand = self.scrollbar_stars_1S.set)
        self.scrollbar_stars_1S.config(command = self.stars_list_1S.yview)
        
        self.stars_list_1S.tag_config("reference", font = "Arial 10 bold", foreground = "#00DF00", lmargin1 = 8)
        self.stars_list_1S.tag_config("normal", font = "Arial 10 bold", foreground = "#FCD116", lmargin1 = 8)
        self.stars_list_1S.tag_config("corrupted", font = "Arial 10 bold", foreground = "#DF0000", lmargin1 = 8)
        self.stars_list_1S.tag_config("text_bold", font = "Arial 10 bold")
        self.stars_list_1S.tag_config("back", font = "Arial 30 bold", foreground = "#F1F1F1")
        self.stars_list_1S.tag_config("text", font = "Arial 10", lmargin1 = 8)
        self.stars_list_1S.insert(END,"\n\n\n", "text")
        self.stars_list_1S.insert(END,"     Stars List", "back")
        self.stars_list_1S.config(state = DISABLED)
        # SubPacking
        self.stars_label_top_1S.pack(side=TOP,anchor=E)
        self.stars_label_side_1S.pack(side=LEFT,anchor=W)
        self.stars_list_1S.pack(side=LEFT)
        self.scrollbar_stars_1S.pack(side=LEFT,fill=Y)
        
        " CCDGS2 "
        self.stars_list_frame_2S = Frame(self.stars_frame)
        self.stars_list_2S = Text (self.stars_list_frame_2S, bg = "white", width = 40, height = 9)
        self.stars_label_top_2S = Label (self.stars_list_frame_2S,text="Star       S/N            " + 
                                                                 "Flux      Mag       FWHM        ")
        self.stars_label_side_2S = Label (self.stars_list_frame_2S,text="CCDGS2 ")
        self.scrollbar_stars_2S = Scrollbar(self.stars_list_frame_2S)
        self.stars_list_2S.config(yscrollcommand = self.scrollbar_stars_2S.set)
        self.scrollbar_stars_2S.config(command = self.stars_list_2S.yview)
        
        self.stars_list_2S.tag_config("reference", font = "Arial 10 bold", foreground = "#00DF00", lmargin1 = 8)
        self.stars_list_2S.tag_config("normal", font = "Arial 10 bold", foreground = "#FCD116", lmargin1 = 8)
        self.stars_list_2S.tag_config("corrupted", font = "Arial 10 bold", foreground = "#DF0000", lmargin1 = 8)
        self.stars_list_2S.tag_config("text_bold", font = "Arial 10 bold")
        self.stars_list_2S.tag_config("back", font = "Arial 30 bold", foreground = "#F1F1F1")
        self.stars_list_2S.tag_config("text", font = "Arial 10", lmargin1 = 8)
        self.stars_list_2S.insert(END,"\n\n\n", "text")
        self.stars_list_2S.insert(END,"     Stars List", "back")
        self.stars_list_2S.config(state = DISABLED)
        # SubPacking
        self.stars_label_top_2S.pack(side=TOP,anchor=E)
        self.stars_label_side_2S.pack(side=LEFT,anchor=W)
        self.stars_list_2S.pack(side=LEFT)
        self.scrollbar_stars_2S.pack(side=LEFT,fill=Y)
        
        " CCDGN1 "
        self.stars_list_frame_1N = Frame(self.stars_frame)
        self.stars_list_1N = Text (self.stars_list_frame_1N, bg = "white", width = 40, height = 9)
        self.stars_label_top_1N = Label (self.stars_list_frame_1N,text="Star       S/N            " + 
                                                                 "Flux      Mag       FWHM        ")
        self.stars_label_side_1N = Label (self.stars_list_frame_1N,text="CCDGN1 ")
        self.scrollbar_stars_1N = Scrollbar(self.stars_list_frame_1N)
        self.stars_list_1N.config(yscrollcommand = self.scrollbar_stars_1N.set)
        self.scrollbar_stars_1N.config(command = self.stars_list_1N.yview)
        
        self.stars_list_1N.tag_config("reference", font = "Arial 10 bold", foreground = "#00DF00", lmargin1 = 8)
        self.stars_list_1N.tag_config("normal", font = "Arial 10 bold", foreground = "#FCD116", lmargin1 = 8)
        self.stars_list_1N.tag_config("corrupted", font = "Arial 10 bold", foreground = "#DF0000", lmargin1 = 8)
        self.stars_list_1N.tag_config("text_bold", font = "Arial 10 bold")
        self.stars_list_1N.tag_config("back", font = "Arial 30 bold", foreground = "#F1F1F1")
        self.stars_list_1N.tag_config("text", font = "Arial 10", lmargin1 = 8)
        self.stars_list_1N.insert(END,"\n\n\n", "text")
        self.stars_list_1N.insert(END,"     Stars List", "back")
        self.stars_list_1N.config(state = DISABLED)
        # SubPacking
        self.stars_label_top_1N.pack(side=TOP,anchor=E)
        self.stars_label_side_1N.pack(side=LEFT,anchor=W)
        self.stars_list_1N.pack(side=LEFT)
        self.scrollbar_stars_1N.pack(side=LEFT,fill=Y)
        
        " CCDGN2 "
        self.stars_list_frame_2N = Frame(self.stars_frame)
        self.stars_list_2N = Text (self.stars_list_frame_2N, bg = "white", width = 40, height = 9)
        self.stars_label_top_2N = Label (self.stars_list_frame_2N,text="Star       S/N            " + 
                                                                 "Flux      Mag       FWHM        ")
        self.stars_label_side_2N = Label (self.stars_list_frame_2N,text="CCDGN2 ")
        self.scrollbar_stars_2N = Scrollbar(self.stars_list_frame_2N)
        self.stars_list_2N.config(yscrollcommand = self.scrollbar_stars_2N.set)
        self.scrollbar_stars_2N.config(command = self.stars_list_2N.yview)
        
        self.stars_list_2N.tag_config("reference", font = "Arial 10 bold", foreground = "#00DF00", lmargin1 = 8)
        self.stars_list_2N.tag_config("normal", font = "Arial 10 bold", foreground = "#FCD116", lmargin1 = 8)
        self.stars_list_2N.tag_config("corrupted", font = "Arial 10 bold", foreground = "#DF0000", lmargin1 = 8)
        self.stars_list_2N.tag_config("text_bold", font = "Arial 10 bold")
        self.stars_list_2N.tag_config("back", font = "Arial 30 bold", foreground = "#F1F1F1")
        self.stars_list_2N.tag_config("text", font = "Arial 10", lmargin1 = 8)
        self.stars_list_2N.insert(END,"\n\n\n", "text")
        self.stars_list_2N.insert(END,"     Stars List", "back")
        self.stars_list_2N.config(state = DISABLED)
        # SubPacking
        self.stars_label_top_2N.pack(side=TOP,anchor=E)
        self.stars_label_side_2N.pack(side=LEFT,anchor=W)
        self.stars_list_2N.pack(side=LEFT)
        self.scrollbar_stars_2N.pack(side=LEFT,fill=Y)
        
        
        # Packing & Positioning
        self.label_stars_title.place(x = self.pad, y = self.pad, anchor = NW)
        self.stars_list_frame_1S.place(x = self.frame_width - self.pad, y = 3 * self.pad, anchor = NE)
        self.stars_list_frame_2S.place(x = self.frame_width - self.pad, y = 18 * self.pad, anchor = NE)
        self.stars_list_frame_1N.place(x = self.frame_width - self.pad, y = 33 * self.pad, anchor = NE)
        self.stars_list_frame_2N.place(x = self.frame_width - self.pad, y = 48 * self.pad, anchor = NE)
    
    def construct_ccds_frame(self):
        # Declare frame
        self.ccds_frame = Frame(self.root, relief = "groove", bd = 2, takefocus = TRUE)
        
        # Labels
        self.label_ccds_title = Label(self.ccds_frame, text = "Guide CCDs View",
                                     font = self.text_highlight, foreground = "#666")
        
        # Buttons
        
        # Packing & Positioning
        self.label_ccds_title.place(x = self.pad, y = self.pad, anchor = NW)
    
    def construct_conf_frame(self):
        # Declare frame
        self.conf_frame = Frame(self.root, relief = "groove", bd = 2, takefocus = TRUE)
        
        # Labels
        self.label_conf_title = Label(self.conf_frame,
                                      text = "Configuration Guider View",
                                      font = self.text_highlight,
                                      foreground = "#666")
        
        # Buttons
        
        # SubFrames
        # >> CCD configuration <<    
        self.subframe_ccds = Frame(self.conf_frame, relief = "groove", bd = 2, padx = 5, pady = 5)
        self.subframe_ccds_modelabel = Label(self.subframe_ccds,
                                             text = "Guide star selection",
                                             font = self.text_small_highlight)
        """ GS1 """
        self.subframe_ccds_1S_title = Label(self.subframe_ccds, text = "CCDGS1", font = self.text_small_highlight)
        self.subframe_ccds_1S_enable_var = BooleanVar()
        self.subframe_ccds_1S_enable = Checkbutton(self.subframe_ccds,
                                                   text = "Enabled",
                                                   font = self.text_small,
                                                   variable = self.subframe_ccds_1S_enable_var,
                                                   command = (lambda ccd_id = 0: self.cmd_config_ccd_enable(ccd_id) ),
                                                   state = DISABLED )
        self.subframe_ccds_1S_mode_var = IntVar()
        self.subframe_ccds_1S_auto = Radiobutton(self.subframe_ccds,
                                                 text = "Auto",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_1S_mode_var,
                                                 value = 0,
                                                 command = (lambda ccd_id = 0: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        self.subframe_ccds_1S_man = Radiobutton(self.subframe_ccds,
                                                 text = "Manual",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_1S_mode_var,
                                                 value = 1,
                                                 command = (lambda ccd_id = 0: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        """ GS2 """
        self.subframe_ccds_2S_title = Label(self.subframe_ccds, text = "CCDGS2", font = self.text_small_highlight)
        self.subframe_ccds_2S_enable_var = BooleanVar()
        self.subframe_ccds_2S_enable = Checkbutton(self.subframe_ccds,
                                                   text = "Enabled",
                                                   font = self.text_small,
                                                   variable = self.subframe_ccds_2S_enable_var,
                                                   command = (lambda ccd_id = 1: self.cmd_config_ccd_enable(ccd_id) ),
                                                   state = DISABLED )
        self.subframe_ccds_2S_mode_var = IntVar()
        self.subframe_ccds_2S_auto = Radiobutton(self.subframe_ccds,
                                                 text = "Auto",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_2S_mode_var,
                                                 value = 0,
                                                 command = (lambda ccd_id = 1: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        self.subframe_ccds_2S_man = Radiobutton(self.subframe_ccds,
                                                 text = "Manual",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_2S_mode_var,
                                                 value = 1,
                                                 command = (lambda ccd_id = 1: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        """ GN1 """
        self.subframe_ccds_1N_title = Label(self.subframe_ccds, text = "CCDGN1", font = self.text_small_highlight)
        self.subframe_ccds_1N_enable_var = BooleanVar()
        self.subframe_ccds_1N_enable = Checkbutton(self.subframe_ccds,
                                                   text = "Enabled",
                                                   font = self.text_small,
                                                   variable = self.subframe_ccds_1N_enable_var,
                                                   command = (lambda ccd_id = 2: self.cmd_config_ccd_enable(ccd_id) ),
                                                   state = DISABLED )
        self.subframe_ccds_1N_mode_var = IntVar()
        self.subframe_ccds_1N_auto = Radiobutton(self.subframe_ccds,
                                                 text = "Auto",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_1N_mode_var,
                                                 value = 0,
                                                 command = (lambda ccd_id = 2: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        self.subframe_ccds_1N_man = Radiobutton(self.subframe_ccds,
                                                 text = "Manual",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_1N_mode_var,
                                                 value = 1,
                                                 command = (lambda ccd_id = 2: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        """ GN2 """
        self.subframe_ccds_2N_title = Label(self.subframe_ccds, text = "CCDGN2", font = self.text_small_highlight)
        self.subframe_ccds_2N_enable_var = BooleanVar()
        self.subframe_ccds_2N_enable = Checkbutton(self.subframe_ccds,
                                                   text = "Enabled",
                                                   font = self.text_small,
                                                   variable = self.subframe_ccds_2N_enable_var,
                                                   command = (lambda ccd_id = 3: self.cmd_config_ccd_enable(ccd_id) ),
                                                   state = DISABLED )
        self.subframe_ccds_2N_mode_var = IntVar()
        self.subframe_ccds_2N_auto = Radiobutton(self.subframe_ccds,
                                                 text = "Auto",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_2N_mode_var,
                                                 value = 0,
                                                 command = (lambda ccd_id = 3: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        self.subframe_ccds_2N_man = Radiobutton(self.subframe_ccds,
                                                 text = "Manual",
                                                 font = self.text_small,
                                                 variable = self.subframe_ccds_2N_mode_var,
                                                 value = 1,
                                                 command = (lambda ccd_id = 3: self.cmd_config_ccd_mode(ccd_id) ),
                                                 state = DISABLED )
        
        # SubPacking
        self.subframe_ccds_modelabel.place( x = 125, y = 0  )
        """ GS1 """
        self.subframe_ccds_1S_title.place(  x = 5  , y = 25 )
        self.subframe_ccds_1S_enable.place( x = 55 , y = 25 )
        self.subframe_ccds_1S_auto.place(   x = 130, y = 25 )
        self.subframe_ccds_1S_man.place(    x = 180, y = 25 )
        """ GS2 """
        self.subframe_ccds_2S_title.place(  x = 5  , y = 50 )
        self.subframe_ccds_2S_enable.place( x = 55 , y = 50 )
        self.subframe_ccds_2S_auto.place(   x = 130, y = 50 )
        self.subframe_ccds_2S_man.place(    x = 180, y = 50 )
        """ GN1 """
        self.subframe_ccds_1N_title.place(  x = 5  , y = 75 )
        self.subframe_ccds_1N_enable.place( x = 55 , y = 75 )
        self.subframe_ccds_1N_auto.place(   x = 130, y = 75 )
        self.subframe_ccds_1N_man.place(    x = 180, y = 75 )
        """ GN2 """
        self.subframe_ccds_2N_title.place(  x = 5  , y = 100)
        self.subframe_ccds_2N_enable.place( x = 55 , y = 100)
        self.subframe_ccds_2N_auto.place(   x = 130, y = 100)
        self.subframe_ccds_2N_man.place(    x = 180, y = 100)
        
        
        # >> Guide mode << 
        self.subframe_mode = Frame(self.conf_frame, relief = "groove", bd = 2, padx = 5, pady = 5, takefocus = TRUE)
        
        self.subframe_mode_title = Label(self.subframe_mode,
                                        text = "Operation Mode",
                                        font = self.text_small_highlight)
        self.subframe_mode_var = IntVar()
        self.subframe_mode_auto_radio =  Radiobutton(self.subframe_mode,
                                                    text = "Auto - Loop over ROI",
                                                    font = self.text_small,
                                                    variable = self.subframe_mode_var,
                                                    value = 0,
                                                    command = self.cmd_config_guider_mode,
                                                    state = DISABLED )
        self.subframe_mode_self_radio =  Radiobutton(self.subframe_mode,
                                                    text = "Self - Guider sets guide star & ROI over full image",
                                                    font = self.text_small,
                                                    variable = self.subframe_mode_var,
                                                    value = 1,
                                                    command = self.cmd_config_guider_mode,
                                                    state = DISABLED )
        self.subframe_mode_user_radio =  Radiobutton(self.subframe_mode,
                                                    text = "User - User sets guide star & ROI over full image",
                                                    font = self.text_small,
                                                    variable = self.subframe_mode_var,
                                                    value = 2,
                                                    command = self.cmd_config_guider_mode,
                                                    state = DISABLED )
        
        
        # SubPacking
        self.subframe_mode_title.place(x = 0, y = 0)
        self.subframe_mode_auto_radio.place(x = 0, y = 20)
        self.subframe_mode_self_radio.place(x = 0, y = 40)
        self.subframe_mode_user_radio.place(x = 0, y = 60)


        # >> TCS Correction Signal <<
        self.subframe_tcs = Frame(self.conf_frame, relief = "groove", bd = 2, padx = 5, pady = 5, takefocus = TRUE)
        
        self.subframe_tcs_title = Label(self.subframe_tcs,
                                        text = "TCS Correction Signal",
                                        font = self.text_small_highlight)
                                        
        
        self.subframe_tcs_enable_var = BooleanVar()
        self.subframe_tcs_enable = Checkbutton(self.subframe_tcs,
                                                   text = "Send corrections",
                                                   font = self.text_small,
                                                   variable = self.subframe_tcs_enable_var,
                                                   command = self.cmd_config_tcs,
                                                   state = DISABLED )
        
        self.subframe_tcs_feedback_label = Label(self.subframe_tcs,
                                        text = "Signal feedback:             %",
                                        font = self.text_small,
                                        state = DISABLED)
        self.subframe_tcs_feedback_entry = Entry(self.subframe_tcs, width = 3, bg = "white", state = DISABLED)
        self.subframe_tcs_feedback_entry.bind( "<Return>",  self.cmd_config_tcs_feedback )
        
        self.subframe_tcs_min_latency_time_label = Label(self.subframe_tcs,
                                        text = "Minimum latency:              s",
                                        font = self.text_small,
                                        state = DISABLED)
        self.subframe_tcs_min_latency_time_entry = Entry(self.subframe_tcs, width = 3, bg = "white", state = DISABLED)
        self.subframe_tcs_min_latency_time_entry.bind( "<Return>",  self.cmd_config_tcs_min_latency_time )
        
        # SubPacking
        self.subframe_tcs_title.place(x = 0, y = 0)
        self.subframe_tcs_enable.place(x = 0, y = 25)
        self.subframe_tcs_feedback_label.place(x = 0, y = 50)
        self.subframe_tcs_feedback_entry.place(x = 80, y = 50)
        self.subframe_tcs_min_latency_time_label.place(x = 0, y = 75)
        self.subframe_tcs_min_latency_time_entry.place(x = 80, y = 75)
        
        
        # Packing & Positioning
        self.label_conf_title.place(x = self.pad, y = self.pad, anchor = NW)
        self.subframe_mode.place(x = self.frame_width / 2, y = 4 * self.pad, width = 270, heigh = 100, anchor = N)
        self.subframe_ccds.place(x = self.frame_width / 2, y = 15 * self.pad, width = 270, heigh = 150, anchor = N)
        self.subframe_tcs.place(x = self.frame_width - self.pad - 5, y = 31 * self.pad, width = 210, heigh = 110, anchor = NE)
    
    def load_current_config(self):
        # Load Guider mode
        guide_mode = self.guider_pml('get', 'default_mode')
        if guide_mode.lower() == 'auto':
            self.subframe_mode_var.set(0)
            # Set all CCDs to auto and disable manual modes
            self.subframe_ccds_1S_auto.select()
            self.subframe_ccds_2S_auto.select()
            self.subframe_ccds_1N_auto.select()
            self.subframe_ccds_2N_auto.select()
            self.subframe_ccds_1S_man.config(state = DISABLED)
            self.subframe_ccds_2S_man.config(state = DISABLED)
            self.subframe_ccds_1N_man.config(state = DISABLED)
            self.subframe_ccds_2N_man.config(state = DISABLED)
        elif guide_mode.lower() == 'self':
            self.subframe_mode_var.set(1)
            # Set all CCDs to auto and disable manual modes
            self.subframe_ccds_1S_auto.select()
            self.subframe_ccds_2S_auto.select()
            self.subframe_ccds_1N_auto.select()
            self.subframe_ccds_2N_auto.select()
            self.subframe_ccds_1S_man.config(state = DISABLED)
            self.subframe_ccds_2S_man.config(state = DISABLED)
            self.subframe_ccds_1N_man.config(state = DISABLED)
            self.subframe_ccds_2N_man.config(state = DISABLED)
        elif guide_mode.lower() == 'user':
            self.subframe_mode_var.set(2)
            # Enable manual CCD configuration
            self.subframe_ccds_1S_man.config(state = NORMAL)
            self.subframe_ccds_2S_man.config(state = NORMAL)
            self.subframe_ccds_1N_man.config(state = NORMAL)
            self.subframe_ccds_2N_man.config(state = NORMAL)
            self.subframe_ccds_1S_man.select()
            self.subframe_ccds_2S_man.select()
            self.subframe_ccds_1N_man.select()
            self.subframe_ccds_2N_man.select()
            
        else:
            self.error("Invalid mode name passed from Guider.")
            
        
        # Load CCD configuration
        """ GS1 """
        if self.shared_ccds[0]._value['active'] == True:
            self.subframe_ccds_1S_enable.select()
        else:
            self.subframe_ccds_1S_enable.deselect()
            
        if self.shared_ccds[0]._value['select_mode'] == 'auto':
            self.subframe_ccds_1S_mode_var.set(0)
        else:
            self.subframe_ccds_1S_mode_var.set(1)
        """ GS2 """
        if self.shared_ccds[1]._value['active'] == True:
            self.subframe_ccds_2S_enable.select()
        else:
            self.subframe_ccds_2S_enable.deselect()
        
        if self.shared_ccds[1]._value['select_mode'] == 'auto':
            self.subframe_ccds_2S_mode_var.set(0)
        else:
            self.subframe_ccds_2S_mode_var.set(1)
        """ GN1 """
        if self.shared_ccds[2]._value['active'] == True:
            self.subframe_ccds_1N_enable.select()
        else:
            self.subframe_ccds_1N_enable.deselect()
        
        if self.shared_ccds[2]._value['select_mode'] == 'auto':
            self.subframe_ccds_1N_mode_var.set(0)
        else:
            self.subframe_ccds_1N_mode_var.set(1)
        """ GN2 """
        if self.shared_ccds[3]._value['active'] == True:
            self.subframe_ccds_2N_enable.select()
        else:
            self.subframe_ccds_2N_enable.deselect()
        
        if self.shared_ccds[3]._value['select_mode'] == 'auto':
            self.subframe_ccds_2N_mode_var.set(0)
        else:
            self.subframe_ccds_2N_mode_var.set(1)
        
        self.cmd_config_multidisplay()
        
        
        # Load TCS configuration
        enablet_value = self.guider_pml('get', 'default_send_corrections')
        self.subframe_tcs_enable_var.set(enablet_value)
        if enablet_value == True:
            feedback_percentage = self.guider_pml('get', 'tcs_feedback')
            self.subframe_tcs_feedback_entry.config(state = NORMAL)
            self.subframe_tcs_feedback_label.config(state = NORMAL)
            self.subframe_tcs_feedback_entry.delete(0, END)
            self.subframe_tcs_feedback_entry.insert(0,"%d"%int(feedback_percentage))
            self.subframe_tcs_feedback_label.config(text = "Signal feedback:             %% [Current is %d]" %feedback_percentage)
        else:
            self.subframe_tcs_feedback_entry.delete(0, END)
            self.subframe_tcs_feedback_entry.insert(0,"0")
            self.subframe_tcs_feedback_entry.config(state = DISABLED)
            self.subframe_tcs_feedback_label.config(state = DISABLED)
            self.subframe_tcs_feedback_label.config(text = "Signal feedback:             % [Current is 0]")
        
        # Min latency time
        min_latency_time = self.guider_pml('get', 'min_latency_time')
        self.subframe_tcs_min_latency_time_entry.delete(0, END)
        self.subframe_tcs_min_latency_time_entry.insert(0,"%.1f"%float(min_latency_time))
        self.subframe_tcs_min_latency_time_label.config(text = "Minimum latency:              s [Current is %.1fs]" %min_latency_time)
    
    
    # Callback commands
    """def cmd_startstop(self):
        # Start / Stop guiding (manual)
        current_status = self.shared_guiderstatus._value
        if current_status == "READY":
            self.guider_pml('start_guide','')
        elif current_status == "GUIDING":
            self.guider_pml('stop_guide','')
        # Disable buttons
        self.btn_obs_startstop.config( state = DISABLED )
        self.btn_exit.config( state = DISABLED )
    
    def cmd_exit(self):
        # Exit Guider & GUI
        current_status = self.shared_guiderstatus._value
        if current_status in ('READY', 'INITIALIZED'):
            self.guider_pml('exit_guider','')
            sys.exit()
        else:
            tkMessageBox.showwarning(title = "Warning!",
                                     message = "Guider not in standby status. Cannot exit.",
                                     icon = tkMessageBox.WARNING)
    
    """
    def cmd_frame_display (self):
        # Set current view (Frame)
        selected_view = self.var_fdisplay.get()
        if selected_view == 0: # Observer View
            self.stars_frame.place_forget()
            self.ccds_frame.place_forget()
            self.conf_frame.place_forget()
            self.obs_frame.place(x = self.config['window_width'] - self.pad,
                                 y = 2 * self.pad,
                                 width = self.frame_width,
                                 height = self.frame_height,
                                 anchor = NE)
        elif selected_view == 1: # Stars View
            self.obs_frame.place_forget()
            self.ccds_frame.place_forget()
            self.conf_frame.place_forget()
            self.stars_frame.place(x = self.config['window_width'] - self.pad,
                                 y = 2 * self.pad,
                                 width = self.frame_width,
                                 height = self.frame_height,
                                 anchor = NE)
        elif selected_view == 2: # CCDs View
            self.obs_frame.place_forget()
            self.stars_frame.place_forget()
            self.conf_frame.place_forget()
            self.ccds_frame.place(x = self.config['window_width'] - self.pad,
                                 y = 2 * self.pad,
                                 width = self.frame_width,
                                 height = self.frame_height,
                                 anchor = NE)
        elif selected_view == 3: # Configuration View
            self.obs_frame.place_forget()
            self.stars_frame.place_forget()
            self.ccds_frame.place_forget()
            self.conf_frame.place(x = self.config['window_width'] - self.pad,
                                  y = 2 * self.pad,
                                  width = self.frame_width,
                                  height = self.frame_height,
                                  anchor = NE)
    
    def cmd_config_multidisplay(self):
        if self.var_multidisplay.get() == False: # Single display
            self.main_screen.itemconfig("multi_screen_1S", state = HIDDEN)
            self.main_screen.itemconfig("multi_screen_2S", state = HIDDEN)
            self.main_screen.itemconfig("multi_screen_1N", state = HIDDEN)
            self.main_screen.itemconfig("multi_screen_2N", state = HIDDEN)
            self.main_screen.itemconfig("full_screen", state = NORMAL)
            if self.shared_ccds[0]._value['active'] == False:
                state_multi = DISABLED
                # Move selection to an active ccd
                if self.var_displayccd.get() == 0:
                    self.var_displayccd.set(1)
            else:
                state_multi = NORMAL
            self.radio_obs_display_ccd1S.config(state = state_multi)
            if self.shared_ccds[1]._value['active'] == False:
                state_multi = DISABLED
                # Move selection to an active ccd
                if self.var_displayccd.get() == 1:
                    self.var_displayccd.set(2)
            else:
                state_multi = NORMAL
            self.radio_obs_display_ccd2S.config(state = state_multi)
            if self.shared_ccds[2]._value['active'] == False:
                state_multi = DISABLED
                # Move selection to an active ccd
                if self.var_displayccd.get() == 2:
                    self.var_displayccd.set(3)
            else:
                state_multi = NORMAL
            self.radio_obs_display_ccd1N.config(state = state_multi)
            if self.shared_ccds[3]._value['active'] == False:
                state_multi = DISABLED
                # Move selection to an active ccd
                if self.var_displayccd.get() == 3:
                    self.var_displayccd.set(0)
            else:
                state_multi = NORMAL
            self.radio_obs_display_ccd2N.config(state = state_multi)
            self.label_obs_displayccd.config(state = NORMAL)
        else: # Multi display
            # Clean Star boxes
            self.main_screen.delete("starbox")
            
            # Update display states
            if self.shared_ccds[0]._value['active'] == False:
                state_multi = HIDDEN
            else:
                state_multi = NORMAL
            self.main_screen.itemconfig("multi_screen_1S", state = state_multi)
            if self.shared_ccds[1]._value['active'] == False:
                state_multi = HIDDEN
            else:
                state_multi = NORMAL            
            self.main_screen.itemconfig("multi_screen_2S", state = state_multi)
            if self.shared_ccds[2]._value['active'] == False:
                state_multi = HIDDEN
            else:
                state_multi = NORMAL
            self.main_screen.itemconfig("multi_screen_1N", state = state_multi)
            if self.shared_ccds[3]._value['active'] == False:
                state_multi = HIDDEN
            else:
                state_multi = NORMAL
            self.main_screen.itemconfig("multi_screen_2N", state = state_multi)
            self.main_screen.itemconfig("full_screen", state = HIDDEN)
            self.radio_obs_display_ccd1S.config(state = DISABLED)
            self.radio_obs_display_ccd2S.config(state = DISABLED)
            self.radio_obs_display_ccd1N.config(state = DISABLED)
            self.radio_obs_display_ccd2N.config(state = DISABLED)
            self.label_obs_displayccd.config(state = DISABLED)
    
    def cmd_config_ccd_enable(self, ccd_id):
        # Read enable var
        if ccd_id == 0:
            enable_value = self.subframe_ccds_1S_enable_var.get()
        elif ccd_id == 1:
            enable_value = self.subframe_ccds_2S_enable_var.get()
        elif ccd_id == 2:
            enable_value = self.subframe_ccds_1N_enable_var.get()
        elif ccd_id == 3:
            enable_value = self.subframe_ccds_2N_enable_var.get()    
        else:
            return -1
        
        # Send PML configuration command to guider
        out_string = "ccd_config='enable,%s,%s'" %(enable_value, ccd_id)
        if self.guider_pml('set', out_string) == self.SUCCESS:
            # Reload remote configuration
            self.load_current_config()
            return self.SUCCESS
            
        else: # Undo button if failed
            if ccd_id == 0:
                self.subframe_ccds_1S_enable_var.set(not enable_value)
            elif ccd_id == 1:
                self.subframe_ccds_2S_enable_var.set(not enable_value)
            elif ccd_id == 2:
                self.subframe_ccds_1N_enable_var.set(not enable_value)
            elif ccd_id == 3:
                self.subframe_ccds_2N_enable_var.set(not enable_value)
            else:
                return -1
        
    
    def cmd_config_ccd_mode(self, ccd_id):
        # Read mode var
        if ccd_id == 0:
            mode_value = self.subframe_ccds_1S_mode_var.get()
        elif ccd_id == 1:
            mode_value = self.subframe_ccds_2S_mode_var.get()
        elif ccd_id == 2:
            mode_value = self.subframe_ccds_1N_mode_var.get()
        elif ccd_id == 3:
            mode_value = self.subframe_ccds_2N_mode_var.get()
        else:
            return -1
        
        # Convert value to mode
        if mode_value == 0:
            mode_value = 'auto'
        elif mode_value == 1:
            mode_value = 'manual'
        else:
            return -1
        
        # Send PML configuration command to guider
        out_string = "ccd_config='select_mode,%s,%s'" %(mode_value, ccd_id)
        if self.guider_pml('set', out_string) == self.SUCCESS:
            self.load_current_config()
            return self.SUCCESS
        else: # Undo button if failed
            if ccd_id == 0:
                self.subframe_ccds_1S_mode_var.set(-1)
            elif ccd_id == 1:
                self.subframe_ccds_2S_mode_var.set(-1)
            elif ccd_id == 2:
                self.subframe_ccds_1N_mode_var.set(-1)
            elif ccd_id == 3:
                self.subframe_ccds_2N_mode_var.set(-1)
            else:
                return -1
                
    
    def cmd_config_tcs(self):
        # Get tkinter values
        enable_value = self.subframe_tcs_enable_var.get()
                
        # Pass to Guider
        out_string = "send_corrections=%s" % enable_value
        result = self.guider_pml('set', out_string)
        if result == self.SUCCESS:
            self.info("TCS signal correctly configured.")
        
        # Load guider settings into GUI
        self.load_current_config()
        return result
    
    def cmd_config_tcs_feedback(self, event):
        # Get tkinter values
        tcs_feedback = int(self.subframe_tcs_feedback_entry.get()) 
        out_string = "tcs_feedback=%d" % tcs_feedback
        self.subframe_tcs.focus_set() # Quit entry focus
        result = self.guider_pml('set', out_string)
        if result == self.SUCCESS:
            self.info("TCS feedback correctly configured to %d%%." %tcs_feedback)
            self.subframe_tcs_feedback_label.config(text = "Signal feedback:             %% [Current is %d]" %tcs_feedback)
        return result
    
    def cmd_config_tcs_min_latency_time(self, event):
        # Get tkinter values
        min_latency_time = float(self.subframe_tcs_min_latency_time_entry.get()) 
        out_string = "min_latency_time=%.1f" % min_latency_time
        self.subframe_tcs.focus_set() # Quit entry focus
        result = self.guider_pml('set', out_string)
        if result == self.SUCCESS:
            self.info("Minimum latency time correctly configured to %f sec." %min_latency_time)
            self.subframe_tcs_min_latency_time_label.config(text = "Minimum latency:              s [Current is %.1f s]" %min_latency_time)
        return result
    
    def cmd_config_guider_mode(self):
        # Get tkinter values
        guider_mode_id = self.subframe_mode_var.get()
                
        # Pass to Guider
        if guider_mode_id == 0: # Auto
            mode = 'auto'
        elif guider_mode_id == 1: # Self
            mode = 'self'
        elif guider_mode_id == 2: # User
            mode = 'user'
        else:
            self.error("Invalid guide mode id")
        out_string = "guimode=%s" % mode
        result = self.guider_pml('set', out_string)
        if result == self.SUCCESS:
            self.info("Guider mode correctly configured.")
        
        # Load guider settings into GUI
        self.load_current_config()
        return result
    
    
    # Event functions
    def event_guiderstatus(self, event):
        if self.connected == False:
            self.debug("Not connected yet. Nothing to do in guider status event.")
            return
        current_status = self.shared_guiderstatus._value
        self.debug("Guider changed status to %s" %current_status)
        lbl = "Guider status: " + current_status
        self.label_guiderstatus.config(text = lbl)
        # Update & enable button
        if current_status == "READY":
            #self.label_guiderstatus.config(text = "Guider status: " + current_status)
            pass
        elif current_status == "GUIDING":
            #self.label_guiderstatus.config(text = "Guider status: " + current_status)
            # Initialize plots
            self.iseq = 0
            self.ax_pix_i = (self.centroids_evo_height - 15) / 2  # Initialize ax initial value
            self.ay_pix_i = (self.centroids_evo_height - 15) / 2  # Initialize ay initial value
            self.ax_pix_i_tcs = (self.centroids_evo_height - 15) / 2  # Initialize ax initial value
            self.ay_pix_i_tcs = (self.centroids_evo_height - 15) / 2  # Initialize ay initial value
            self.x_axis_count = self.x_plot_start
            self.plot_centroid_evolution.delete("xline")
            self.plot_centroid_evolution.delete("yline")
            self.plot_centroid_evolution.itemconfig("x_label_max",
                text = str( 1 + (self.centroids_evo_width - self.x_plot_start) / self.x_axis_step))
            self.plot_centroid_evolution.itemconfig("x_label_min", text = str(0))
            self.plot_centroid_evolution.itemconfig("mean_dev",state=HIDDEN)
            self.label_centr_out_x.config(text = "  -        ")
            self.label_centr_out_y.config(text = "  -        ")
            # Clean Star boxes
            self.main_screen.delete("starbox")
            # Clean Star Lists
            self.stars_list_1S.config(state=NORMAL)
            self.stars_list_1S.delete(1.0, END)
            self.stars_list_1S.config(state=DISABLED)
            self.stars_list_2S.config(state=NORMAL)
            self.stars_list_2S.delete(1.0, END)
            self.stars_list_2S.config(state=DISABLED)
            self.stars_list_1N.config(state=NORMAL)
            self.stars_list_1N.delete(1.0, END)
            self.stars_list_1N.config(state=DISABLED)
            self.stars_list_2N.config(state=NORMAL)
            self.stars_list_2N.delete(1.0, END)
            self.stars_list_2N.config(state=DISABLED)
            # Restore image displays
            multi_view_initial_path = "external/media/welcome_256.gif"
            self.full_screen = PhotoImage( file = "external/media/welcome.gif" ) # Load Welcome Image
            self.multi_screen_1S = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
            self.multi_screen_2S = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
            self.multi_screen_1N = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
            self.multi_screen_2N = PhotoImage( file = multi_view_initial_path ) # Load Welcome Image
            self.main_screen.itemconfig("full_screen", image = self.full_screen )
            self.main_screen.itemconfig("multi_screen_1S", image = self.multi_screen_1S )
            self.main_screen.itemconfig("multi_screen_2S", image = self.multi_screen_2S )
            self.main_screen.itemconfig("multi_screen_1N", image = self.multi_screen_1N )
            self.main_screen.itemconfig("multi_screen_2N", image = self.multi_screen_2N )
             
    
    def event_ccd(self,event):
        # Do nothing if not connected yet
        if self.connected == False:
            return
        # Avoid events out of guiding
        current_status = self.shared_guiderstatus._value
        if current_status != "GUIDING":
            return
        
        ccd_id = int( event.name[-1] ) # CCDX
        self.debug("New CCD %d event received" %ccd_id)
        
        current_CCD = self.shared_ccds[ccd_id]._value
        
        # Refresh Stars Lists
        if current_CCD['original_reference_star'] != None: # Run only after reading guide data
            if ccd_id == 0: # CCDGS1
                stars_list_txt = []
                # Create star list to order by SN (txt)
                self.stars_list_1S.config(state=NORMAL)
                self.stars_list_1S.delete(1.0, END)
                istar = 0
                
                stars_list_sorted = sorted(current_CCD['stars_list'],
                                           key=lambda stars: stars['sn'], reverse=True)
                
                if current_CCD['original_reference_star'] == 'waiting': 
                    # Not ready yet in manual mode
                    reference_id = -1
                else:
                    reference_id = current_CCD['last_reference_star']['number']
                
                for star in stars_list_sorted:
                    if star['number'] == reference_id:
                        text_type = "reference"
                    elif star['suitable'] == True:
                        text_type = "text"
                    else:
                        text_type = "corrupted"
                    self.stars_list_1S.insert(END,"   %d\t%.1f\t%.1e\t%.2f       %.2f\n"
                                                  %(istar+1,
                                                  star['sn'],
                                                  star['flux_auto'],
                                                  star['mag_ref'],
                                                  star['fwhm_image']),
                                                  text_type)
                    istar += 1
                self.stars_list_1S.config(state=DISABLED)
            elif ccd_id == 1: # CCDGS2
                stars_list_txt = []
                # Create star list to order by SN (txt)
                self.stars_list_2S.config(state=NORMAL)
                self.stars_list_2S.delete(1.0, END)
                istar = 0
                
                stars_list_sorted = sorted(current_CCD['stars_list'],
                                           key=lambda stars: stars['sn'], reverse = True)
                                           
                if current_CCD['original_reference_star'] == 'waiting': 
                    # Not ready yet in manual mode
                    reference_id = -1
                else:
                    reference_id = current_CCD['last_reference_star']['number']
                    
                for star in stars_list_sorted:
                    if star['number'] == reference_id:
                        text_type = "reference"
                    elif star['suitable'] == True:
                        text_type = "text"
                    else:
                        text_type = "corrupted"
                    self.stars_list_2S.insert(END,"   %d\t%.1f\t%.1e\t%.2f       %.2f\n"
                                                  %(istar+1,
                                                  star['sn'],
                                                  star['flux_auto'],
                                                  star['mag_ref'],
                                                  star['fwhm_image']),
                                                  text_type)
                    istar += 1
                self.stars_list_2S.config(state=DISABLED)
                
            elif ccd_id == 2: # CCDGN1
                stars_list_txt = []
                # Create star list to order by SN (txt)
                self.stars_list_1N.config(state=NORMAL)
                self.stars_list_1N.delete(1.0, END)
                istar = 0
                
                stars_list_sorted = sorted(current_CCD['stars_list'],
                                           key=lambda stars: stars['sn'], reverse = True)
                
                if current_CCD['original_reference_star'] == 'waiting': 
                    # Not ready yet in manual mode
                    reference_id = -1
                else:
                    reference_id = current_CCD['last_reference_star']['number']
                
                for star in stars_list_sorted:
                    if star['number'] == reference_id:
                        text_type = "reference"
                    elif star['suitable'] == True:
                        text_type = "text"
                    else:
                        text_type = "corrupted"
                    self.stars_list_1N.insert(END,"   %d\t%.1f\t%.1e\t%.2f       %.2f\n"
                                                  %(istar+1,
                                                  star['sn'],
                                                  star['flux_auto'],
                                                  star['mag_ref'],
                                                  star['fwhm_image']),
                                                  text_type)
                    istar += 1
                self.stars_list_1N.config(state=DISABLED)
                
            elif ccd_id == 3: # CCDGN2
                stars_list_txt = []
                # Create star list to order by SN (txt)
                self.stars_list_2N.config(state=NORMAL)
                self.stars_list_2N.delete(1.0, END)
                istar = 0
                
                stars_list_sorted = sorted(current_CCD['stars_list'],
                                           key=lambda stars: stars['sn'], reverse=True)
                
                if current_CCD['original_reference_star'] == 'waiting': 
                    # Not ready yet in manual mode
                    reference_id = -1
                else:
                    reference_id = current_CCD['last_reference_star']['number']
                
                for star in stars_list_sorted:
                    if star['number'] == reference_id:
                        text_type = "reference"
                    elif star['suitable'] == True:
                        text_type = "text"
                    else:
                        text_type = "corrupted"
                    self.stars_list_2N.insert(END,"   %d\t%.1f\t%.1e\t%.2f       %.2f\n"
                                                  %(istar+1,
                                                  star['sn'],
                                                  star['flux_auto'],
                                                  star['mag_ref'],
                                                  star['fwhm_image']),
                                                  text_type)
                    istar += 1
                self.stars_list_2N.config(state=DISABLED)
            
            
            # Refresh Single/Multi CCD display
            
            # Wait if other CCD is in manual selection (semaphore?)
            #while self.enable_manual_selection == True:
            
            # Change View to Single in Manual and display 
            if current_CCD['original_reference_star'] == 'waiting':
                self.var_multidisplay.set(False)
                self.var_displayccd.set(ccd_id)
                self.cmd_config_multidisplay()
            
            # >> Single CCD display
            if self.var_multidisplay.get() == False:
                                
                # If this ccd is display ccd, update main_screen
                if ccd_id == self.var_displayccd.get():
                    self.debug("Updating Single display for CCD %d" %ccd_id)
                    
                    # Separate CCD from extension
                    mosaic_image_path = current_CCD['image_path']
                    single_image_path = os.path.join( self.tmp_directory, "converted_ccd%d.fits" %ccd_id )
                    if os.path.exists(single_image_path):
                        os.remove(single_image_path)
                    ccd_hdu = pyfits.open(mosaic_image_path)
                    self.ccd_image_size = list(ccd_hdu[ccd_id + 1].data.shape)
                    pyfits.writeto(single_image_path,
                                   data = ccd_hdu[ccd_id + 1].data,
                                   header = ccd_hdu[ccd_id + 1].header)
                    ccd_hdu.close()
                    
                    # Convert new image to shape & format
                    in_image_path = single_image_path
                    out_image_path = single_image_path.replace(".fits", ".gif")
                    if self.var_obs_gamma.get() == True:
                        gamma_str = " -gamma %d " % self.config['gamma_contrast']
                    else:
                        gamma_str = " "
                    convert_str = "convert -filter point -resize 512 " + gamma_str + in_image_path + " " + out_image_path
                    os.system(convert_str)
                    
                    # Display
                    self.full_screen = PhotoImage( file = out_image_path ) # Load Image
                    self.main_screen.itemconfig("full_screen", image = self.full_screen )
                    
                    # Remove previous detection boxes
                    self.main_screen.delete("starbox")
                    
                    if current_CCD['original_reference_star'] == 'waiting': 
                        # Not ready yet in manual mode
                        ref_id = -1
                    else:
                        ref_id = current_CCD['last_reference_star']['number']
                    
                    stars_list_sorted = sorted(current_CCD['stars_list'],
                                               key=lambda stars: stars['sn'], reverse = True)
                    for istar, star in enumerate(stars_list_sorted):
                        if star['suitable'] == True:
                            if ref_id == star['number']:
                                boxcolor = "green"
                            else:
                                boxcolor = "yellow"
                        else:
                            boxcolor = "red"
                        # Full Frame (Self and User modes)
                        #if ( self.subframe_mode_var.get() == 1 or 
                        #   ( self.subframe_mode_var.get() == 2 and ref_id == -1) ) :
                        if self.ccd_image_size[0] > 512:
                            x_i = ((512. / (self.ccd_image_size[0])) * star['x_image']) - self.star_box_size +1
                            y_i = (512 - (512. / (self.ccd_image_size[1])) * star['y_image']) - self.star_box_size +1
                            y_f = (512 - (512. / (self.ccd_image_size[1])) * star['y_image']) + self.star_box_size +1
                            x_f = ((512. / (self.ccd_image_size[0])) * star['x_image']) + self.star_box_size +1
                            self.main_screen.create_rectangle(x_i + self.pad / 2,
                                                              y_i + self.pad / 2,
                                                              x_f + self.pad / 2,
                                                              y_f + self.pad / 2,
                                                              outline = boxcolor,
                                                              tag = "starbox",
                                                              width = 2)
                            # Star ID label positioning                                  
                            if  x_i + 25 < 490:
                                x_i_label = x_i + 30
                            else:
                                x_i_label = x_i - 8
                            if y_i > 25:
                                y_i_label = y_i + 4
                            else:
                                y_i_label = y_i + 16
                            self.main_screen.create_text(x_i_label + self.pad / 2,
                                                              y_i_label + self.pad / 2,
                                                              text = str(istar+1),
                                                              fill = boxcolor,
                                                              font="Arial 9",
                                                              tag = "starbox")
                            
                        # ROI image (Auto mode)
                        else:
                            x_i = ((512. / (self.ccd_image_size[0] + 1)) * star['x_image']) - 2 * self.star_box_size
                            y_i = (512 - (512. / (self.ccd_image_size[1] + 1)) * star['y_image']) - 2 * self.star_box_size
                            y_f = (512 - (512. / (self.ccd_image_size[1] + 1)) * star['y_image']) + 2 * self.star_box_size
                            x_f = ((512. / (self.ccd_image_size[0] + 1)) * star['x_image']) + 2 * self.star_box_size
                            self.main_screen.create_rectangle(x_i + self.pad / 2,
                                                              y_i + self.pad / 2,
                                                              x_f + self.pad / 2,
                                                              y_f + self.pad / 2,
                                                              outline = boxcolor,
                                                              tag = "starbox",
                                                              width = 3)
                                                              
                           # Star ID label positioning
                            if  x_i + 2*25 < 490:
                                x_i_label = x_i + 2*30
                            else:
                                x_i_label = x_i - 2*8
                            if y_i > 2*25:
                                y_i_label = y_i + 2*4
                            else:
                                y_i_label = y_i + 2*16
                            self.main_screen.create_text(x_i_label + self.pad / 2,
                                                              y_i_label + self.pad / 2,
                                                              text = str(istar+1),
                                                              fill = boxcolor,
                                                              font="Arial 14 bold",
                                                              tag = "starbox")
            
            # >> Multi CCD display
            else: 
                # Skip if current CCD is disabled
                if current_CCD['active'] == False:
                    self.main_screen.itemconfig("multi_screen_1N", state = HIDDEN)
                    return
                else:
                    self.main_screen.itemconfig("multi_screen_1N", state = NORMAL)
                    
                # Clean posible Star boxes remaining
                self.main_screen.delete("starbox")
                
                # Separate CCD from extension
                mosaic_image_path = current_CCD['image_path']
                single_image_path = os.path.join( self.tmp_directory, "converted_ccd%d.fits" %ccd_id )
                if os.path.exists(single_image_path):
                    os.remove(single_image_path)
                ccd_hdu = pyfits.open(mosaic_image_path)
                pyfits.writeto(single_image_path,
                               data = ccd_hdu[ccd_id + 1].data,
                               header = ccd_hdu[ccd_id + 1].header)
                ccd_hdu.close()
                
                # Convert new image to shape & format
                in_image_path = single_image_path
                out_image_path = single_image_path.replace(".fits", ".gif")
                if self.var_obs_gamma.get() == True:
                    gamma_str = " -gamma %d " % self.config['gamma_contrast']
                else:
                    gamma_str = " "
                convert_str = "convert -filter point -resize 256 " + gamma_str + in_image_path + " " + out_image_path
                os.system(convert_str)
                
                # Update Canvas
                if ccd_id == 0: 
                    self.multi_screen_1S = PhotoImage( file = out_image_path ) # Load Image
                    self.main_screen.itemconfig("multi_screen_1S", image = self.multi_screen_1S )
                elif ccd_id == 1:
                    self.multi_screen_2S = PhotoImage( file = out_image_path ) # Load Image
                    self.main_screen.itemconfig("multi_screen_2S", image = self.multi_screen_2S )
                elif ccd_id == 2:
                    self.multi_screen_1N = PhotoImage( file = out_image_path ) # Load Image
                    self.main_screen.itemconfig("multi_screen_1N", image = self.multi_screen_1N )
                elif ccd_id == 3:
                    self.multi_screen_2N = PhotoImage( file = out_image_path ) # Load Image
                    self.main_screen.itemconfig("multi_screen_2N", image = self.multi_screen_2N )
                    
            
            # Manual selection
            if current_CCD['original_reference_star'] == 'waiting':
                self.enable_manual_selection = True
                # Set ccd_id screen & Single Image Display
                # TBD
                self.main_screen.config(cursor = "target", bg = "red")
                self.main_screen.create_text(256, 25, 
                    text = "Please, select your guiding star in CCD %s" %self.config['guide_ccds'][ccd_id],
                    font = "Arial 10 bold",
                    fill = "white",
                    tag = "select_text")
                while self.enable_manual_selection == True:
                    time.sleep(0.1)
                    
                # Search star selected by user
                closest_dist_to_target = 10000
                selected_star = None
                for star in current_CCD['stars_list']:
                    
                    dist_to_target = math.sqrt((star['x_image'] - self.manual_x_pos) ** 2 +
                                               (star['y_image'] - self.manual_y_pos) ** 2 )
                    if dist_to_target < closest_dist_to_target:
                        closest_dist_to_target = dist_to_target
                        selected_star = star
                
                # Set manually selected reference star
                current_CCD['original_reference_star'] = selected_star  # CHANGE
                # Set reference manually
                self.guider_pml('set_reference_manual', '%d %d'
                                %(ccd_id, current_CCD['original_reference_star']['number']) )
    
    def event_centroid_out(self, event):
        self.debug("New CENTROID event received")
        # Do nothing if not connected yet
        if self.connected == False:
            return
        
        new_centroid = self.shared_centroid._value
        
        # Check if centroid information is available yet
        if new_centroid == None:
            return
        
        # Check if GUI is initialized
        try: self.x_axis_count
        except: return
        
        
        # Refresh CCD status
        self.load_current_config()
        
        # Update labels
        self.label_centr_out_x.config(text = '%.3f"   ' %new_centroid['measured'][0])
        self.label_centr_out_y.config(text = '%.3f"   ' %new_centroid['measured'][1])
        
        # Update Centroid evolution plot
        """ Convert from X offset (in CCD pixels) to plot pixels (in arcsec) """
        self.ax_pix_f = - (new_centroid['measured'][0] - self.y_max_value) * (self.centroids_evo_height - 15)/(2 * self.y_max_value)
        self.ay_pix_f = - (new_centroid['measured'][1] - self.y_max_value) * (self.centroids_evo_height - 15)/(2 * self.y_max_value)
        self.ax_pix_f_tcs = - (new_centroid['tcssignal'][0] - self.y_max_value) * (self.centroids_evo_height - 15)/(2 * self.y_max_value)
        self.ay_pix_f_tcs = - (new_centroid['tcssignal'][1] - self.y_max_value) * (self.centroids_evo_height - 15)/(2 * self.y_max_value)
        if self.x_axis_count < self.centroids_evo_width:
            # Measured centroid
            self.plot_centroid_evolution.create_line(self.x_axis_count,
                                                     self.ax_pix_i,
                                                     self.x_axis_count + self.x_axis_step,
                                                     self.ax_pix_f,
                                                     width = 2,
                                                     fill = "#CCF",
                                                     tags = "xline")
            self.plot_centroid_evolution.create_line(self.x_axis_count,
                                                     self.ay_pix_i,
                                                     self.x_axis_count + self.x_axis_step,
                                                     self.ay_pix_f,
                                                     width = 2,
                                                     fill = "#FCC",
                                                     tags = "yline")
            # TCS signal correction
            self.plot_centroid_evolution.create_line(self.x_axis_count,
                                                     self.ax_pix_i_tcs,
                                                     self.x_axis_count + self.x_axis_step,
                                                     self.ax_pix_f_tcs,
                                                     width = 2,
                                                     fill = "#22F",
                                                     tags = "xline")
            self.plot_centroid_evolution.create_line(self.x_axis_count,
                                                     self.ay_pix_i_tcs,
                                                     self.x_axis_count + self.x_axis_step,
                                                     self.ay_pix_f_tcs,
                                                     width = 2,
                                                     fill = "#F22",
                                                     tags = "yline")
            self.x_axis_count = self.x_axis_count + self.x_axis_step
        else:
            self.plot_centroid_evolution.move("xline", -self.x_axis_step, 0)
            self.plot_centroid_evolution.move("yline", -self.x_axis_step, 0)
            # Measured centroid
            self.plot_centroid_evolution.create_line(self.x_axis_count - self.x_axis_step, 
                                                     self.ax_pix_i, 
                                                     self.x_axis_count, 
                                                     self.ax_pix_f, 
                                                     width = 2,
                                                     fill = "#CCF",
                                                     tags = "xline")
            self.plot_centroid_evolution.create_line(self.x_axis_count - self.x_axis_step,
                                                     self.ay_pix_i,
                                                     self.x_axis_count,
                                                     self.ay_pix_f,
                                                     width = 2,
                                                     fill = "#FCC",
                                                     tags = "yline")
            # TCS signal correction
            self.plot_centroid_evolution.create_line(self.x_axis_count - self.x_axis_step, 
                                                     self.ax_pix_i_tcs, 
                                                     self.x_axis_count, 
                                                     self.ax_pix_f_tcs, 
                                                     width = 2,
                                                     fill = "#22F",
                                                     tags = "xline")
            self.plot_centroid_evolution.create_line(self.x_axis_count - self.x_axis_step,
                                                     self.ay_pix_i_tcs,
                                                     self.x_axis_count,
                                                     self.ay_pix_f_tcs,
                                                     width = 2,
                                                     fill = "#F22",
                                                     tags = "yline")
            self.plot_centroid_evolution.itemconfig("x_label_max",
                text = str(self.iseq + 1))
            self.plot_centroid_evolution.itemconfig("x_label_min", 
                text = str(self.iseq + 1 - (self.centroids_evo_width - self.x_plot_start) / self.x_axis_step))
            # Remove lines outside the plot
            # Find all X lines out of the plot
            out_plot = list(self.plot_centroid_evolution.find_overlapping(-200000, 0,
                                                                          self.x_plot_start - 1,
                                                                          self.centroids_evo_height))
            """ X """
            x_lines = list(self.plot_centroid_evolution.find_withtag("xline"))
            # DELETE all lines out of the plot
            for iline in x_lines:
                if out_plot.count(iline)==1: #this line is outside the plot area
                    self.plot_centroid_evolution.delete(iline)
            """ Y """
            y_lines = list(self.plot_centroid_evolution.find_withtag("yline"))
            # DELETE all lines out of the plot
            for iline in y_lines:
                if out_plot.count(iline)==1: #this line is outside the plot area
                    self.plot_centroid_evolution.delete(iline)
        
        self.ax_pix_i = self.ax_pix_f
        self.ay_pix_i = self.ay_pix_f
        self.ax_pix_i_tcs = self.ax_pix_f_tcs
        self.ay_pix_i_tcs = self.ay_pix_f_tcs
        self.iseq += 1
    
    
    def event_manual_selection(self, event):
        if self.enable_manual_selection == True:
            self.main_screen.config(cursor = "arrow", bg = "grey")
            self.manual_x_pos = (event.x - (self.pad/2)) * self.ccd_image_size[0] / 512.
            self.manual_y_pos = (512 - event.y + (self.pad/2)) * self.ccd_image_size[1] / 512.
            self.main_screen.delete("select_text")
            self.enable_manual_selection = False
    
    
    # Main
    def main(self):
        self.root.mainloop()
        self.info("GUI exited gracefully!")
    
    


if __name__ == "__main__":
    GUI().run()
