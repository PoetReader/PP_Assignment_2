#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <math.h>
#include <string.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) < (b) ? (b) : (a))
#define GP(X,Y) GetPixel(img, width, height,(X),(Y))


/**
 * @brief Convert an image to grayscale
 *
 * @param img
 * @param width
 * @param height
 */
void ImageToGrayscale(RGB *img, const int width, const int height){
    for(int i=0;i<width*height;i++){
        char grayscale=img[i].red*0.3+img[i].green*0.59+img[i].blue*0.11;
        img[i].red=grayscale;
        img[i].green=grayscale;
        img[i].blue=grayscale;
    }
}

/**
 * @brief Return a pixel no matter the coordinate
 *
 * @param img
 * @param width
 * @param height
 * @param x
 * @param y
 * @return RGB
 */
RGB GetPixel(RGB *img, const int width, const int height, const int x, const int y){
	if(x<0 || y<0 || x>=width || y >=height){
		int approxX=MIN(MAX(x,0),width);
		int approxY=MIN(MAX(y,0),height);
		return(img[approxX+approxY*width]);
	}
	return(img[x+y*width]);
}

void ApplySobel(RGB *img, const int width, const int height){}

void ApplyEmboss(RGB *img, const int width, const int height){}


int main(){
    const char *marguerite="marguerite.bmp";
    const char *outsobel="sobel.bmp";
    const char *outemboss="emboss.bmp";

    // Get image dimensions
    int width,height;
    GetSize(marguerite,&width, &height);

    // Init memory
    RGB *sobel=malloc(sizeof(RGB)*width*height);
    RGB *emboss=malloc(sizeof(RGB)*width*height);

    // Load images
    LoadRegion(marguerite,0,0, width,height,sobel);
    LoadRegion(marguerite,0,0, width,height,emboss);

    // Apply filters
    ApplySobel(sobel,width,height);
    ApplyEmboss(emboss,width,height);

    // Save images
    CreateBMP(outsobel,width,height);
    WriteRegion(outsobel,0,0,width, height,sobel);
    CreateBMP(outemboss,width,height);
    WriteRegion(outemboss,0,0,width, height,emboss);

    // Free memory
    free(sobel);
    free(emboss);

    return(0);
}
