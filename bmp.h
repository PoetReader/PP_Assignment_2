#pragma once

#include <stdlib.h>
#include <stdio.h>
#include "libbmp/libbmp.h"


typedef struct RGB {
  unsigned char red;
  unsigned char green;
  unsigned char blue;
}RGB;

void GetSize(const char* filepath,int* width, int *height);
void LoadRegion(const char* filepath,const int x,const int y,const int width,const int height,RGB* region);
void WriteRegion(const char* filepath,const int x,const int y,const int width,const int height,RGB* region);
void CreateBMP(const char* filepath,const int width,const int height);
