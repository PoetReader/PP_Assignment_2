#include "bmp.h"

void GetSize(const char* filepath,int* width, int *height){
	bmp_img img;
	bmp_img_read (&img, filepath);
    *width=img.img_header.biWidth;
    *height=img.img_header.biHeight;
    bmp_img_free (&img);
}



void LoadRegion(const char* filepath,const int x,const int y,const int width,const int height,RGB* region){
    bmp_img img;
	bmp_img_read (&img, filepath);
    // Load region
    for(int curY=y;curY<y+height;curY++){
        for(int curX=x;curX<x+width;curX++){
            region[(curY-y)*width+(curX-x)].red=img.img_pixels[curY][curX].red;
            region[(curY-y)*width+(curX-x)].green=img.img_pixels[curY][curX].green;
            region[(curY-y)*width+(curX-x)].blue=img.img_pixels[curY][curX].blue;
        }
    }
    bmp_img_free (&img);
}

void WriteRegion(const char* filepath,const int x,const int y,const int width,const int height,RGB* region){
    bmp_img img;
	bmp_img_read (&img, filepath);
    // Load region
    for(int curY=y;curY<y+height;curY++){
        for(int curX=x;curX<x+width;curX++){
            img.img_pixels[curY][curX].red=region[(curY-y)*width+(curX-x)].red;
            img.img_pixels[curY][curX].green=region[(curY-y)*width+(curX-x)].green;
            img.img_pixels[curY][curX].blue=region[(curY-y)*width+(curX-x)].blue;
        }
    }
    bmp_img_write (&img, filepath);
    bmp_img_free (&img);
}

void CreateBMP(const char* filepath,const int width,const int height){
    bmp_img img;
    bmp_img_init_df (&img, width,height);
    bmp_img_write (&img, filepath);
    bmp_img_free (&img);
}