#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "fitsio.h"

void printerror( int status);

void write_image(char *filename, int bitpix, long naxis, long *naxes, float *array) {


  fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
  int status;
  long  fpixel, nelements, exposure;
  int i;

  /*printf ("Starting write_image\n");*/

  remove(filename) ;    /* Delete old file if it already exists */

  status = 0;         /* initialize status before calling fitsio routines */

  if (fits_create_file(&fptr, filename, &status)) /* create new FITS file */
    printerror( status );           /* call printerror if error occurs */


  if (fits_create_img(fptr,  bitpix, naxis, naxes, &status) )
    printerror( status );          

  
  fpixel = 1;                               /* first pixel to write      */
  nelements = 1.;
  for (i=0; i<naxis; i++) {
    nelements *= naxes[i] ;          /* number of pixels to write */
  }

  if (fits_write_img(fptr, TFLOAT, fpixel, nelements, &array[0], &status) )
    printerror( status );

  if ( fits_close_file(fptr, &status) )                /* close the file */
    printerror( status );           

  /* return;*/

}


void printerror( int status)
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/


    if (status)
    {
       fits_report_error(stderr, status); /* print error report */

       exit( status );    /* terminate the program, returning error status */
    }
    return;
}
