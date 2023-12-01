CC = gcc
OPTIMIZE =-g
EXEC = simulate_c
OBJS = simulate_c.o write_image.o ran2.o gasdev.o ran1.o
INCL = proto.h Makefile
CFITSIO_LIBS = -L/opt/local/lib/
CFITSIO_INC = -I/opt/local/include/
LIBS = $(CFITSIO_LIBS) -lcfitsio -lm
#LIBS = $(CFITSIO_LIBS) ./libcfitsio.a -lm


all: $(OBJS) $(EXEC)


$(EXEC): $(OBJS) 
	$(CC) $(OPTIMIZE) $(OBJS) $(LIBS) $(CFITSIO_INC) -o $(EXEC) 

simulate_c.o:
	$(CC) -c -o simulate_c.o simulate_c.c $(CFITSIO_LIBS) $(CFITSIO_INC) 

write_image.o:
	$(CC) -c -o write_image.o write_image.c $(CFITSIO_LIBS) $(CFITSIO_INC)


clean:
	rm -f $(OBJS)
