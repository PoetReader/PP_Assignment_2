#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <mpi.h>
#include <math.h>
#include <string.h>

#include <omp.h>

#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))

/**
 * @brief Partition the image into tiles
 * @param rank
 * @param size
 * @param height
 * @param start
 * @param rows
 * @param upper_halo
 * @param lower_halo
 */
static void GetTiles(int rank, int size, int height, int *start, int *rows, int *upper_halo, int *lower_halo)
{
    int tile = (height + size - 1) / size;                // get tile size
    *start = MIN(rank * tile, height);                    // get start row of rank
    *rows = MIN(tile, height - *start);                   // get number of rows for the rank (if last rank then rest)
    *upper_halo = (*rows > 0 && *start > 0);              // set upper halo to true depending if rank has border tile or not
    *lower_halo = (*rows > 0 && *start < height - *rows); // set lower halo to true depending if rank has border tile or not
}

/**
 * @brief Build the scatter and gather table to use MPI_Scatterv() and MPI_Gatherv()
 * @param size
 * @param height
 * @param total_rows
 * @param start_rows
 * @param true_rows
 * @param true_start
 */
static void BuildScatterGather(int size, int height, int *total_rows, int *start_rows, int *true_rows, int *true_start)
{
    for (int rank = 0; rank < size; rank++)
    {
        int start, rows, upper_halo, lower_halo;
        GetTiles(rank, size, height, &start, &rows, &upper_halo, &lower_halo);
        total_rows[rank] = rows + lower_halo + upper_halo; // get rows including halo rows
        start_rows[rank] = start - upper_halo;             // start including halo rows to send to ranks
        true_rows[rank] = rows;                            // send rows without halo back
        true_start[rank] = start;                          // get start of a tile without halo
    }
}

/**
 * @brief Convert an image to grayscale (parallel)
 * @param tile
 * @param total_pixels
 */
void ImageToGrayscaleParallel(RGB *tile, const long total_pixels)
{
#pragma omp parallel for schedule(static) // pixel spread to threads
    for (long i = 0; i < total_pixels; i++)
    {

        unsigned char grayscale = tile[i].red * 0.3 + tile[i].green * 0.59 + tile[i].blue * 0.11;
        tile[i].red = grayscale;
        tile[i].green = grayscale;
        tile[i].blue = grayscale;
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
        int approxX = MIN(MAX(x, 0), width - 1);
        int approxY = MIN(MAX(y, 0), height - 1);
        return (img[approxX + approxY * width]);
    }
    return (img[x + y * width]);
}

/**
 * @brief Apply Sobel filter to image parallel
 * @param rank
 * @param tile_img
 * @param width
 * @param upper_halo
 * @param lower_halo
 * @param rows
 * @param tile_output
 */
void ApplySobelParallel(int rank, RGB *tile_img, const int width, const int upper_halo, const int lower_halo, const int rows, RGB *tile_output)
{
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

    int local_height = upper_halo + lower_halo + rows; // calc local height basically used_rows

    // print the threads working during parallelisation
#pragma omp parallel
    {
#pragma omp master
        {
            printf("Sobel Filter => Rank %d: %d threads working\n", rank, omp_get_num_threads());
        }
#pragma omp parallel for collapse(2) schedule(static) // parallellise
        for (int y = upper_halo; y < (upper_halo + rows); y++)
        {
            for (int x = 0; x < width; x++)
            {

                float pixel_x = 0.0f;
                float pixel_y = 0.0f;

                for (int con_y = -1; con_y <= 1; con_y++)
                {
                    for (int con_x = -1; con_x <= 1; con_x++)
                    {
                        RGB pixel = GetPixel(tile_img, width, local_height, x + con_x, y + con_y);
                        pixel_x += pixel.red * Gx[con_y + 1][con_x + 1];
                        pixel_y += pixel.red * Gy[con_y + 1][con_x + 1];
                    }
                }
                float res = sqrtf(pixel_x * pixel_x + pixel_y * pixel_y);
                unsigned char final_res = (unsigned char)MIN(MAX(res, 0.0f), 255.0f);

                size_t index = (size_t)(y - upper_halo) * width + x; // remember the halo row
                tile_output[index].red = final_res;
                tile_output[index].green = final_res;
                tile_output[index].blue = final_res;
            }
        }
    }
}

/**
 * @brief Apply Emboss filter to image parallel
 * @param rank
 * @param tile_img
 * @param width
 * @param upper_halo
 * @param lower_halo
 * @param rows
 * @param tile_output
 */
void ApplyEmbossParallel(int rank, RGB *tile_img, const int width, const int upper_halo, const int lower_halo, const int rows, RGB *tile_output)
{
    // Sobel Matrix Kernel
    const int Kernel[3][3] = {
        {-2, -1, 0},
        {-1, 1, 1},
        {0, 1, 2}};

    int local_height = upper_halo + lower_halo + rows; // calc local height basically used_rows

#pragma omp parallel
    {
#pragma omp master
        {
            printf("Emboss Filter => Rank %d: %d threads working\n", rank, omp_get_num_threads());
        }
#pragma omp parallel for collapse(2) schedule(static) // parallellise
        for (int y = upper_halo; y < (upper_halo + rows); y++)
        {
            for (int x = 0; x < width; x++)
            {
                int sum = 0;
                for (int con_y = -1; con_y <= 1; con_y++)
                {
                    for (int con_x = -1; con_x <= 1; con_x++)
                    {
                        RGB pixel = GetPixel(tile_img, width, local_height, x + con_x, y + con_y);
                        sum += pixel.red * Kernel[con_y + 1][con_x + 1];
                    }
                }
                unsigned char res = (unsigned char)MIN(MAX(sum, 0), 255);
                size_t index = (size_t)(y - upper_halo) * width + x; // remember the halo row
                tile_output[index].red = res;
                tile_output[index].green = res;
                tile_output[index].blue = res;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Multi Threading MPI setup
    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) // check for correct number of parameters
    {
        if (rank == 0)
        { // only rank 0 prints
            fprintf(stderr, "Useage: %s <input.bmp>\n", argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    const char *marguerite = argv[1];
    const char *outsobel = "sobelparallel.bmp";
    const char *outemboss = "embossparallel.bmp";

    int width, height; // Get image dimensions
    RGB *img;
    RGB *sobel_full;
    RGB *emboss_full;

    if (rank == 0) // Rank 0 gets image dimension
    {
        GetSize(marguerite, &width, &height);
        img = malloc(sizeof(RGB) * width * height);
        LoadRegion(marguerite, 0, 0, width, height, img); // Load entire image
    }

    // Image size broadcasted to other MPI processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);


    //alloc sobel_full and emboss_full
    if (rank == 0)
    {
        sobel_full = malloc(sizeof(RGB) * (size_t)width * height);
        emboss_full = malloc(sizeof(RGB) * (size_t)width * height);
    }

    // Tiling
    int start, rows, upper_halo, lower_halo;
    GetTiles(rank, size, height, &start, &rows, &upper_halo, &lower_halo);
    const int used_rows = rows + upper_halo + lower_halo; // get row number with halo or without depending if boarder tile

    MPI_Datatype rowtype;                                              // MPI Type for an entire row
    MPI_Type_contiguous(width * (int)sizeof(RGB), MPI_BYTE, &rowtype); // one row = width *sizeof(RGB) Bytes
    MPI_Type_commit(&rowtype);                                         // activate type

    int *total_rows; // scatter: array holding rows including all ranks rows + halo rows
    int *start_rows; // scatter: array holding start row of each rank
    int *true_rows;  // gather: array holding the true rows (without halo rows) for eah rank
    int *true_start; // gather: array holding start row of true rows for each rank

    if (rank == 0)
    {
        total_rows = (int *)malloc(size * sizeof(int)); // Alloc mem
        start_rows = (int *)malloc(size * sizeof(int));
        true_rows = (int *)malloc(size * sizeof(int));
        true_start = (int *)malloc(size * sizeof(int));
        BuildScatterGather(size, height, total_rows, start_rows, true_rows, true_start); // Fill the arrays for scatter and gather tables
    }

    // Init mem for tile with halo rows, and tile output without halo rows
    RGB *tile_img = (RGB *)malloc((size_t)used_rows * width * sizeof(RGB));
    RGB *tile_output_sobel = (RGB *)malloc((size_t)rows * width * sizeof(RGB));
    RGB *tile_output_emboss = (RGB *)malloc((size_t)rows * width * sizeof(RGB));

    // rank 0 sends tiles (with halo rows) and each rank receives its tile to work upon
    MPI_Scatterv(img, total_rows, start_rows, rowtype, tile_img, used_rows, rowtype, 0, MPI_COMM_WORLD);

    long total_pixels = (long)used_rows * width;

    // Start timer
    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec start_timer, end;
    clock_gettime(CLOCK_MONOTONIC, &start_timer);

    ImageToGrayscaleParallel(tile_img, total_pixels);

    ApplySobelParallel(rank, tile_img, width, upper_halo, lower_halo, rows, tile_output_sobel);
    ApplyEmbossParallel(rank, tile_img, width, upper_halo, lower_halo, rows, tile_output_emboss);

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &end); // finish measuring the time
    double local_elapsed = (end.tv_sec - start_timer.tv_sec) + (end.tv_nsec - start_timer.tv_nsec) / 1e9;

    // Start timer to measure gathering time
    MPI_Barrier(MPI_COMM_WORLD);
    struct timespec start_timer_run, end_run;
    clock_gettime(CLOCK_MONOTONIC, &start_timer_run);

    // each rank sends their tile without halo rows and rank 0 gathers the tiles
    MPI_Gatherv(tile_output_sobel, rows, rowtype, sobel_full, true_rows, true_start, rowtype, 0, MPI_COMM_WORLD);
    MPI_Gatherv(tile_output_emboss, rows, rowtype, emboss_full, true_rows, true_start, rowtype, 0, MPI_COMM_WORLD);

    // End timer to measure gathering time
    clock_gettime(CLOCK_MONOTONIC, &end_run); // finish measuring the time
    double local_elapsed_run = (end_run.tv_sec - start_timer_run.tv_sec) + (end_run.tv_nsec - start_timer_run.tv_nsec) / 1e9;
    double max_elapsed_run;
    MPI_Reduce(&local_elapsed_run, &max_elapsed_run, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // get slowest rank
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Init memory
    // RGB *sobel = malloc(sizeof(RGB) * width * height);
    // RGB *emboss = malloc(sizeof(RGB) * width * height);

    // Load images
    // LoadRegion(marguerite, 0, 0, width, height, sobel);
    // LoadRegion(marguerite, 0, 0, width, height, emboss);

    // Apply filters
    // start measuring the time

    // ApplyEmboss(emboss, width, height);

    if (rank == 0)
    {
        CreateBMP(outsobel, width, height);
        WriteRegion(outsobel, 0, 0, width, height, sobel_full);
        CreateBMP(outemboss, width, height);
        WriteRegion(outemboss, 0, 0, width, height, emboss_full);
        printf("Time: %.9f seconds\nTime to put everything together: %.9f\n", max_elapsed, max_elapsed_run);
    }

    // Free memory
    free(tile_img);
    free(tile_output_sobel);
    free(tile_output_emboss);

    if (rank == 0)
    {
        free(img);
        free(sobel_full);
        free(emboss_full);
        free(total_rows);
        free(start_rows);
        free(true_rows);
        free(true_start);
    }
    MPI_Type_free(&rowtype); // free MPI type

    MPI_Finalize();

    return (0);
}
