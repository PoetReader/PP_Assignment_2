#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <mpi.h>
#include <math.h>
#include <string.h>

#include <omp.h>


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
 * @brief Calculate the convolution of a pixel for Sobel filtering
 * @param up
 * @param mid
 * @param down
 * @param x_left
 * @param x
 * @param x_right
 * @param out
 * @param idx
 */
void SobelPixel(const RGB *up, const RGB *mid, const RGB *down, int xl, int x, int xr, RGB *out, int idx)
{
    // leave out index where multiplicator is 0 and already use factor out the minus
    int gx = (up[xr].red + 2 * mid[xr].red + down[xr].red) - (up[xl].red + 2 * mid[xl].red + down[xl].red);
    int gy = (up[xl].red + 2 * up[x].red + up[xr].red) - (down[xl].red + 2 * down[x].red + down[xr].red);

    float res = sqrtf((float)(gx * gx + gy * gy));
    unsigned char final_res = (unsigned char)MIN(res, 255.0f);

    out[idx].red = final_res;
    out[idx].green = final_res;
    out[idx].blue = final_res;
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
void ApplySobelParallel(int rank, RGB *tile_img, const int width, const int upper_halo,
                        const int lower_halo, const int rows, RGB *tile_output)
{
    const int local_height = upper_halo + lower_halo + rows;

#pragma omp parallel for schedule(static)
    for (int y = upper_halo; y < upper_halo + rows; y++) // iterate through the rows
    {
        // part 9x9 neighbourhood into rows array move pointer to start location of up, mid or down
        RGB *up = tile_img + (size_t)MAX(y - 1, 0) * width;
        RGB *mid = tile_img + (size_t)y * width;
        RGB *down = tile_img + (size_t)MIN(y + 1, local_height - 1) * width;
        RGB *out = tile_output + (size_t)(y - upper_halo) * width;

        // Left pixel of the row
        SobelPixel(up, mid, down, 0, 0, MIN(1, width - 1), out, 0);

        // inner pixel iterating through the pixels in a row
        for (int x = 1; x < width - 1; x++)
        {
            SobelPixel(up, mid, down, x - 1, x, x + 1, out, x);
        }

        // right pixel of the row
        SobelPixel(up, mid, down, MAX(width - 2, 0), width - 1, width - 1, out, width - 1);
    }
}

/**
 * @brief Calculate the convolution of a pixel for Emboss filtering
 * @param up
 * @param mid
 * @param down
 * @param x_left
 * @param x
 * @param x_right
 * @param out
 * @param idx
 */
void EmbossPixel(const RGB *up, const RGB *mid, const RGB *down, int xl, int x, int xr, RGB *out, int idx)
{
    // leave out index where multiplicator is 0 and already use factor out the minus
    int sum = -2 * up[xl].red - 1 * up[x].red - mid[xl].red + mid[x].red + mid[xr].red + down[x].red + 2 * down[xr].red;

    unsigned char res = (unsigned char)MIN(MAX(sum, 0), 255);

    out[idx].red = res;
    out[idx].green = res;
    out[idx].blue = res;
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
void ApplyEmbossParallel(int rank, RGB *tile_img, const int width, const int upper_halo,
                        const int lower_halo, const int rows, RGB *tile_output)
{
     const int local_height = upper_halo + lower_halo + rows;

#pragma omp parallel for schedule(static)
    for (int y = upper_halo; y < upper_halo + rows; y++) // iterate through the rows
    {
        // part 9x9 neighbourhood into rows array move pointer to start location of up, mid or down
        RGB *up = tile_img + (size_t)MAX(y - 1, 0) * width;
        RGB *mid = tile_img + (size_t)y * width;
        RGB *down = tile_img + (size_t)MIN(y + 1, local_height - 1) * width;
        RGB *out = tile_output + (size_t)(y - upper_halo) * width;

        // Left pixel of the row
        EmbossPixel(up, mid, down, 0, 0, MIN(1, width - 1), out, 0);

        // inner pixel iterating through the pixels in a row
        for (int x = 1; x < width - 1; x++)
        {
            EmbossPixel(up, mid, down, x - 1, x, x + 1, out, x);
        }

        // right pixel of the row
        EmbossPixel(up, mid, down, MAX(width - 2, 0), width - 1, width - 1, out, width - 1);
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
    const char *outemboss = "emboss.bmp";

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
    double start_timer = MPI_Wtime();

    ImageToGrayscaleParallel(tile_img, total_pixels);

    ApplySobelParallel(rank, tile_img, width, upper_halo, lower_halo, rows, tile_output_sobel);
    ApplyEmbossParallel(rank, tile_img, width, upper_halo, lower_halo, rows, tile_output_emboss);

    // End timer
    double local_elapsed = MPI_Wtime() - start_timer;
    // each rank sends their tile without halo rows and rank 0 gathers the tiles

    // Start timer to measure gathering time
    MPI_Barrier(MPI_COMM_WORLD);
    double start_timer_run = MPI_Wtime();

    // each rank sends their tile without halo rows and rank 0 gathers the tiles
    MPI_Gatherv(tile_output_sobel, rows, rowtype, img, true_rows, true_start, rowtype, 0, MPI_COMM_WORLD);
    MPI_Gatherv(tile_output_emboss, rows, rowtype, emboss_full, true_rows, true_start, rowtype, 0, MPI_COMM_WORLD);

    // End timer to measure gathering time
    double local_elapsed_run = MPI_Wtime() - start_timer_run;
    double max_elapsed_run;
    MPI_Reduce(&local_elapsed_run, &max_elapsed_run, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // get slowest rank
    double max_elapsed;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


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
