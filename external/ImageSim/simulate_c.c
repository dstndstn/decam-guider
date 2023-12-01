/*  Program to simulate guider images */
/*  the program runs with a set of arguments */
/*  simulate_c parameter_file nx ny seq_number box_width_half(0:do not use) x_star_pos y_star_pos CCD_number xoff yoff working_directory*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "proto.h"
#include "fitsio.h"

/* global variables */
/* to be read from parameter file */

float gain;
float readout_noise;
float scale;
float zeropoint;
float seeing;
float exptime;
float sky;
float stars_mag_bright;
float stars_mag_faint;
int stars_seed;
float turbulence_sigma;
int sample_freq;
FILE *fd;



int main(int argc, char **argv) {

  time_t start, stop;
  int elapsed;

  int i,j,k,s,iseq;
  char ParameterFile[100];

  float ad = -0.7;          /* parameter of stellar nmber counts */
  float alpha = 0.18;       /* parameter of stellar nmber counts */


  int star_max_pix_size;    /* Maximum pixels to print a star */
  int st_iterations;        /* Number of iterations to print each star */
  char buf[200],bufoff[200],str_tmp[200];
  float area_pix,area_sky;
  int nx,ny;                /*CCD pixel size*/
  float rnmax;
  int ir,nstar;
  int idummy;
  long ldummy;
  float fdummy;
  float t1,t2;
  float *starmag;
  float *xs,*ys,*fl;
  float sigma;
  float xoff,yoff,roff,random_angle;
  float xd,yd;
  float sv;
  float *v;
  FILE *fd1,*fd2,*fd3;
  int bitpix;
  long naxis;
  long *naxes;
  float *rand_atm_x;
  float *rand_atm_y;
  int xbox_star_pos_c;
  int ybox_star_pos_c;
  int xbox_star_pos_i;
  int ybox_star_pos_i;
  int xbox_star_pos_f;
  int ybox_star_pos_f;
  int box_width_half;
  int box_full_width;
  int iCCD;
  char working_directory[200];


  start = time(NULL);
  if (argc < 12) {
    printf("Missing parameters\n Format is: ./simulate_c parameter_file nx ny seq_number box_width_half(0:do not use) x_star_pos(-1: preselected ROI) y_star_pos CCD_number xoff yoff working_directory\n");
    error(0);
  } else {
    strcpy(ParameterFile, argv[1]);

    nx = atoi( argv[2] );
    ny = atoi( argv[3] );
    iseq = atoi( argv[4] );
    box_width_half = atoi( argv[5] );
    xbox_star_pos_c = atoi( argv[6] );
    ybox_star_pos_c = atoi( argv[7] );
    iCCD = atoi( argv[8] );
    xoff = atof( argv[9] );
    yoff = atof( argv[10] );
    strcpy(working_directory, argv[11]);

    /*Compute star box cut out*/
    if (box_width_half!=0) {
	xbox_star_pos_i = xbox_star_pos_c - box_width_half;
	xbox_star_pos_f = xbox_star_pos_c + box_width_half;
	if (xbox_star_pos_i<0) {
	    xbox_star_pos_i = 0;
	    xbox_star_pos_f = (box_width_half*2) +1;
	  }
	if (xbox_star_pos_f > nx) {
	    xbox_star_pos_i = nx-((box_width_half*2) +1) ;
	    xbox_star_pos_f = nx;
	  }

	ybox_star_pos_i = ybox_star_pos_c - box_width_half;
	ybox_star_pos_f = ybox_star_pos_c + box_width_half;
	if (ybox_star_pos_i<0) {
	    ybox_star_pos_i = 0;
	    ybox_star_pos_f = (box_width_half*2) +1;
	  }
	if (ybox_star_pos_f > ny) {
	    ybox_star_pos_i = ny-((box_width_half*2) +1) ;
	    ybox_star_pos_f = ny;
	  }
	
	/*Preselected ROI*/  
    if (xbox_star_pos_c == -1 ){
        xbox_star_pos_i = 0;
        xbox_star_pos_f = (box_width_half*2) +1; 
        ybox_star_pos_i = 0;
	    ybox_star_pos_f = (box_width_half*2) +1;
	    
      }
    }


    /*Get Parameters from parameter file*/
    read_parameter_file(ParameterFile,working_directory);

    /* Modify seed for each CCD*/
    stars_seed = stars_seed+iCCD;

    star_max_pix_size = seeing*5/scale;
    st_iterations = exptime*sample_freq; /*exptime*(X iterations per second)*/

    v=malloc(nx * ny * sizeof(float));

    /*Create .offsets file*/
    sprintf(bufoff,"%s/data/tmp/offsets.temp",working_directory);

    if (iseq==0) {
      if (!(fd2 = fopen(bufoff,"w"))) {  /*First iteration create*/
	printf ("ERROR: Failed to open/create file=%s\n",bufoff);
	error(0);
      }
    } else {
      if (!(fd2 = fopen(bufoff,"a"))) { /*Next iterations append*/
	printf ("ERROR: Failed to open file=%s\n",bufoff);
	error(0);
      }
    }

    area_pix=nx*ny;
    area_sky=area_pix*scale*scale;

    /*Number of Stars in CCD*/
    rnmax=pow(10.,(ad+stars_mag_faint*alpha));
    nstar=rnmax*area_sky/3600./3600. ;

    starmag=malloc(nstar * sizeof(float));
    xs=malloc(nstar * sizeof(float));
    ys=malloc(nstar * sizeof(float));
    fl=malloc(nstar * sizeof(float));

    /* initialize ran2 random number generator */
    /* Position & magnitude seed working together*/
    ldummy = (long) (-stars_seed);
    //fdummy=gasdev(&ldummy);


    /* Get list of star MAGNITUDES */
    ir=0;
    while (ir<nstar) {
      t1=stars_mag_bright+ran2(&ldummy)*(stars_mag_faint-stars_mag_bright);
      t2=ran2(&ldummy)*rnmax;
      if (t2 <= pow(10.,(ad+t1*alpha))) {
        starmag[ir]=t1;
        ir++;
      }
    }

    /* Get list of stars POSITION and FLUX */
    for (k=0; k < nstar; k++) {
      xs[k]=ran2(&ldummy)*nx ;
      ys[k]=ran2(&ldummy)*ny ;
      fl[k]=pow(10.,(-.4*(starmag[k]-zeropoint)))*exptime ;
    }
    
    /* Ensure central bright star (15-18 mag) in preselected ROI*/
    if (xbox_star_pos_c == -1 ){
        xs[0] = box_width_half;
        ys[0] = box_width_half;
        starmag[0] = 15 + ran2(&ldummy)*3;
        // printf ("Central star mag: %f\n",starmag[0]) ;
        fl[0]=pow(10.,(-.4*( starmag[0] - zeropoint)))*exptime ;
        }

    /* generate image values */
    sv=pow(10.,(-.4*(sky-zeropoint))*scale*scale*exptime); /* sky value e- per pixel */

    /* take ldummy from previous iterations */
    if (iseq != 0){
      sprintf(str_tmp,"%s/data/tmp/ldummy.temp",working_directory);
      if (!(fd3 = fopen(str_tmp,"r"))) {
	printf ("Failed to open ldummy file");
	error(0);
      } else {
	fscanf( fd3,"%ld",&ldummy );
	ldummy=-ldummy;
	fclose(fd3);
      }
    }

    /* Create Image for current sequence */

    sprintf(buf,"%s/data/tmp/decam-guide_ccd%d.fits",working_directory,iCCD);

    rand_atm_x= malloc(nstar *st_iterations* sizeof(float));
    rand_atm_y= malloc(nstar *st_iterations* sizeof(float));


    /*Random sequence for non-gaussian exposition*/
    sigma=seeing/2.3548/scale;
    for (k=0;k<nstar;k++){
      for (s=0;s<st_iterations;s++){
	rand_atm_x[s+k*st_iterations]= gasdev(&ldummy)*sigma;
	rand_atm_y[s+k*st_iterations]= gasdev(&ldummy)*sigma;
      }
    }

    if (box_width_half == 0){

	/*Print pixel per pixel*/
	for (i=0;i<nx;i++) {
	  for (j=0;j<ny;j++) {
	    /* Sky */
	    v[i+j*nx]=sv+sqrt(sv)*gasdev(&ldummy) ;

	    /* Read-out noise */
	    v[i+j*nx]=v[i+j*nx]+readout_noise*gasdev(&ldummy) ;

	    /* Stars */
	    for (k=0;k<nstar;k++) {
	      if ( abs(xs[k]-i)<star_max_pix_size & abs(ys[k]-j)<star_max_pix_size) {
		for (s=0;s<st_iterations;s++){
		  xd=((float) i )-(xs[k]+xoff+ rand_atm_x[s+k*st_iterations] -1) ;
		  yd=((float) j )-(ys[k]+yoff+ rand_atm_y[s+k*st_iterations] -1) ;
		  if ((xd<0.5) & (xd>-0.5) & (yd<0.5) & (yd>-0.5) ){
		    v[i+j*nx]=v[i+j*nx]+fl[k]/st_iterations;
		  }
		}
	      }
	    }

	    v[i+j*nx]=v[i+j*nx]/gain ;      /* convert to ADU */

	  }

	}


	/*WRITE fits image*/
	bitpix = FLOAT_IMG;   /* image of float type */
	//bitpix = LONG_IMG;
	naxis = 2;
	naxes=malloc(naxis * sizeof(long));
	naxes[0]= nx;
	naxes[1]= ny;
	write_image(buf, bitpix, naxis, naxes, &v[0]);


      }

    else {

	box_full_width=(box_width_half*2) +1;

	/*Print pixel per pixel*/
	for (i=0;i<box_full_width;i++) {
	  for (j=0;j<box_full_width;j++) {
	    /* Sky */
	    v[i+j*box_full_width]=sv+sqrt(sv)*gasdev(&ldummy) ;

	    /* Read-out noise */
	    v[i+j*box_full_width]=v[i+j*box_full_width]+readout_noise*gasdev(&ldummy) ;

	    /* Stars */
	    for (k=0;k<nstar;k++) {
	      if ( abs(xs[k]-i-xbox_star_pos_i)<star_max_pix_size & abs(ys[k]-j-ybox_star_pos_i)<star_max_pix_size) {
		for (s=0;s<st_iterations;s++){
		  xd=((float) i +xbox_star_pos_i )-(xs[k]+xoff+ rand_atm_x[s+k*st_iterations] -1) ;
		  yd=((float) j +ybox_star_pos_i )-(ys[k]+yoff+ rand_atm_y[s+k*st_iterations] -1) ;
		  if ((xd<0.5) & (xd>-0.5) & (yd<0.5) & (yd>-0.5) ){
		    v[i+j*box_full_width]=v[i+j*box_full_width]+fl[k]/st_iterations;
		  }
		}
	      }
	    }

	    v[i+j*box_full_width]=v[i+j*box_full_width]/gain ;      /* convert to ADU */

	  }

	}

	/*WRITE fits image*/
	bitpix = FLOAT_IMG;   /* image of float type */
	//bitpix = LONG_IMG;
	naxis = 2;
	naxes=malloc(naxis * sizeof(long));
	naxes[0]= box_full_width;
	naxes[1]= box_full_width;
	write_image(buf, bitpix, naxis, naxes, &v[0]);


    }


    /* write stars to file */
    /*for (k=0;k<nstar;k++) {
      fprintf(fd2,"%4d %4d %8.3f %8.3f %8.3f %8.3f %8.2f %12.3e\n",iseq,k,xs[k],ys[k],xoff,yoff,starmag[k],fl[k]);
    }*/


    /*Free memory*/
    free(rand_atm_x);
    free(rand_atm_y);
    free(naxes);
    free(v);
    free(starmag);
    free(xs);
    free(ys);
    free(fl);
  }


  /* Save ldummy for next iteration */
  sprintf(str_tmp,"%s/data/tmp/ldummy.temp",working_directory);
  if (!(fd3 = fopen(str_tmp,"w"))) {
    printf ("Failed to create ldummy file");
    error(0);
  } else {
    fprintf(fd3,"%ld",ldummy);
    fclose(fd3);
  }

  /* Make .DONE file */
  sprintf(buf,"%s/data/tmp/decam-guide_ccd%d.DONE",working_directory,iCCD);
  fd = fopen(buf, "w");
  fclose(fd);

  /*Compute & Print time*/
  stop = time(NULL);
  elapsed = difftime(stop, start);

  //printf("Simulation time: %dm %ds\n", elapsed/60,elapsed%60);



}


void read_parameter_file(char *fname,char *working_directory) {
#define DOUBLE 1
#define STRING 2
#define INT 3
#define FLOAT 4
#define MAXTAGS 300

  FILE *fd, *fdout;
  char buf[200], buf1[200], buf2[200], buf3[400];
  int i, j, nt;
  int id[MAXTAGS];
  void *addr[MAXTAGS];
  char tag[MAXTAGS][50];
  int pnum, errorFlag = 0;

  nt = 0;

  strcpy(tag[nt], "gain");
  addr[nt] = &gain;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "readout_noise");
  addr[nt] = &readout_noise;
  id[nt++] = FLOAT;;

  strcpy(tag[nt], "scale");
  addr[nt] = &scale;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "zeropoint");
  addr[nt] = &zeropoint;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "seeing");
  addr[nt] = &seeing;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "exptime");
  addr[nt] = &exptime;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "sky");
  addr[nt] = &sky;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "stars_mag_bright");
  addr[nt] = &stars_mag_bright;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "stars_mag_faint");
  addr[nt] = &stars_mag_faint;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "stars_seed");
  addr[nt] = &stars_seed;
  id[nt++] = INT;

  strcpy(tag[nt], "turbulence_sigma");
  addr[nt] = &turbulence_sigma;
  id[nt++] = FLOAT;

  strcpy(tag[nt], "sample_freq");
  addr[nt] = &sample_freq;
  id[nt++] = INT;

  

  if (fd = fopen(fname, "r")) {
    sprintf(buf, "%s/data/tmp/%s",working_directory, "sim-usedvalues.params");
    if (!(fdout = fopen(buf, "w"))) {
      printf("error opening file '%s' \n", buf);
      errorFlag = 1;
    } else {
      while(!feof(fd)) {
        *buf = 0;
        fgets(buf, 200, fd);
        if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
	  continue;

        if(buf1[0] == '%')
	  continue;

        for(i = 0, j = -1; i < nt; i++) {
	  if(strcmp(buf1, tag[i]) == 0) {
	    j = i;
	    tag[i][0] = 0;
	    break;
	  }
        }

        if (j >= 0) {
	  switch (id[j]) {
	    case DOUBLE:
	      *((double *) addr[j]) = atof(buf2);
	      fprintf(fdout, "%-35s%g\n", buf1, *((double *) addr[j]));
	      break;
	    case STRING:
	      strcpy(addr[j], buf2);
	      fprintf(fdout, "%-35s%s\n", buf1, buf2);
	      break;
	    case INT:
	      *((int *) addr[j]) = atoi(buf2);
	      fprintf(fdout, "%-35s%d\n", buf1, *((int *) addr[j]));
	      break;
	    case FLOAT:
	      *((float *) addr[j]) = (float) atof(buf2);
	      fprintf(fdout, "%-35s%f\n", buf1, *((float *) addr[j]));
	      break;
	  }
        } else {
	  fprintf(stdout, "Error in file %s:   Tag '%s' not allowed or multiple defined.\n",fname, buf1);
	  errorFlag = 1;
        }
      }
      fclose(fd);
      fclose(fdout);


    }
  } else {
    printf("Parameter file %s not found.\n", fname);
    errorFlag = 1;
  }
  

  for(i = 0; i < nt; i++) {
    if(*tag[i]) {
      printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], fname);
      errorFlag = 1;
    }
  }
  

  if(errorFlag) {
      exit(0);
  }
#undef DOUBLE
#undef STRING
#undef INT
#undef MAXTAGS
}

int error(int v){
  printf("%d",v);
}

