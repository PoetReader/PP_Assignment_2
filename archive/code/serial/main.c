#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <math.h>
#include <string.h>
#include <time.h>


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define GP(X, Y) GetPixel(img, width, height, (X), (Y))

/**
 * @brief Convert an image to grayscale
 *
 * @param img
 * @param width
 * @param height
 */
void ImageToGrayscale(RGB *img, const int width, const int height)
{
    for (int i = 0; i < width * height; i++)
    {
        char grayscale = img[i].red * 0.3 + img[i].green * 0.59 + img[i].blue * 0.11;
        img[i].red = grayscale;
        img[i].green = grayscale;
        img[i].blue = grayscale;
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
RGB GetPixel(RGB *img, const int width, const int height, const int x, const int y)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        int approxX = MIN(MAX(x, 0), width); // out of bounds?
        int approxY = MIN(MAX(y, 0), height);
        return (img[approxX + approxY * width]);
    }
    return (img[x + y * width]);
}

/**
 * @brief Apply Sobel filter to image
 */
void ApplySobel(RGB *img, const int width, const int height)
{
    // First use ImageToGrayScale on image
    ImageToGrayscale(img, width, height);

    // Next two 3 by 3 matrices (or kernels) should be convoluted with the grayscale version of the image A
    // Sobel Matrix Kernel
    const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    const int Gy[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}};

    // temp buffer for calc
    RGB *temp = (RGB *)malloc(sizeof(RGB) * width * height);
    // convolusion over each pixel
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            // get pixel
            float pixel_x = 0.0f;
            float pixel_y = 0.0f;

            // Convolusion iterating over neighbouring pixels as well
            for (int con_x = -1; con_x <= 1; con_x++)
            {
                for (int con_y = -1; con_y <= 1; con_y++)
                {
                    // Get the pixels out of the 3x3 neighbourhood
                    RGB pixel = GetPixel(img, width, height, x + con_x, y + con_y);
                    // +1 to get the correct index in the Gx or Gy matrix
                    // Calc to receive A * Gx or A * Gy
                    pixel_x += pixel.red * Gx[con_y + 1][con_x + 1];
                    pixel_y += pixel.red * Gy[con_y + 1][con_x + 1];
                }
            }
            // G = sqrt(Gx^2 + Gy^2)
            float res = sqrtf(pixel_x * pixel_x + pixel_y * pixel_y);

            // bring value into correct range
            unsigned char final_res = (unsigned char)MIN(MAX(res, 0.0f), 255.0f);

            // get correct index of array
            int index = x + y * width;
            temp[index].red = final_res;
            temp[index].green = final_res;
            temp[index].blue = final_res;
        }
    }
    // copy temp image to img
    memcpy(img, temp, sizeof(RGB) * width * height);
    //free buffer
    free(temp);
}

void ApplyEmboss(RGB *img, const int width, const int height)
{
    /*1. Ein 3x3 Emboss Kernel K
    2. Faltung durchführen, Bias addieren ?
    Clamping auf [0,255] anwenden.
    */
}

int main(int argc, char *argv[])
{
    // check for correct parameters
    if (argc < 2)
    {
        fprintf(stderr, "Useage: %s <input.bmp>\n", argv[0]);
        return 0;
    }

    const char *marguerite = argv[1];
    const char *outsobel = "sobel.bmp";
    // const char *outemboss = "emboss.bmp";

    // Get image dimensions
    int width, height;
    GetSize(marguerite, &width, &height);

    // Init memory
    RGB *sobel = malloc(sizeof(RGB) * width * height);
    // RGB *emboss = malloc(sizeof(RGB) * width * height);

    // Load images
    LoadRegion(marguerite, 0, 0, width, height, sobel);
    // LoadRegion(marguerite, 0, 0, width, height, emboss);

    // Apply filters
    // start measuring the time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    ApplySobel(sobel, width, height);
    clock_gettime(CLOCK_MONOTONIC, &end); // finish measuring the time
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time: %.3f seconds\n", elapsed);
    // ApplyEmboss(emboss, width, height);

    // Save images
    CreateBMP(outsobel, width, height);
    WriteRegion(outsobel, 0, 0, width, height, sobel);
    // CreateBMP(outemboss, width, height);
    // WriteRegion(outemboss, 0, 0, width, height, emboss);

    // Free memory
    free(sobel);
    // free(emboss);

    return (0);
}
