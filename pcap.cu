#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <cuda_runtime.h>

// CUDA kernel to convert image to grayscale
__global__ void grayscale(unsigned char *rgb_image, unsigned char *gray_image, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int index = (row * width + col) * 3;
        int gray = 0.2126 * rgb_image[index] + 0.7152 * rgb_image[index + 1] + 0.0722 * rgb_image[index + 2];
        gray_image[row * width + col] = (unsigned char) gray;
    }
}

// CUDA kernel to convert image to sepia
__global__ void sepia(unsigned char *rgb_image, unsigned char *sepia_image, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int index = (row * width + col) * 3;
        int r = rgb_image[index];
        int g = rgb_image[index + 1];
        int b = rgb_image[index + 2];
        int sr = (int)(0.393 * r + 0.769 * g + 0.189 * b);
        int sg = (int)(0.349 * r + 0.686 * g + 0.168 * b);
        int sb = (int)(0.272 * r + 0.534 * g + 0.131 * b);
        sepia_image[index] = (unsigned char) (sr > 255 ? 255 : sr);
        sepia_image[index + 1] = (unsigned char) (sg > 255 ? 255 : sg);
        sepia_image[index + 2] = (unsigned char) (sb > 255 ? 255 : sb);
    }
}

// CUDA kernel to convert image to negative
__global__ void negative(unsigned char *rgb_image, unsigned char *neg_image, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int index = (row * width + col) * 3;
        neg_image[index] = 255 - rgb_image[index];
        neg_image[index + 1] = 255 - rgb_image[index + 1];
        neg_image[index + 2] = 255 - rgb_image[index + 2];
    }
}

// Function to read JPEG image file
int read_jpeg(const char *filename, unsigned char **image, int *width, int *height) {
    FILE *input_file = fopen(filename, "rb");

    if (!input_file) {
        fprintf(stderr, "Error: Could not open JPEG file %s\n", filename);
        return -1;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, input_file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    int num_components = cinfo.num_components;

    *image = (unsigned char *) malloc(*width * *height * num_components * sizeof(unsigned char));
    unsigned char *rowptr;
    while (cinfo.output_scanline < cinfo.output_height) {
        rowptr = *image + cinfo.output_scanline * *width * num_components;
        jpeg_read_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(input_file);

    return 0;
}

void write_jpeg_grayscale(unsigned char *image, int width, int height, char *output_path) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_pointer[1];
    int row_stride;

    FILE *outfile = fopen(output_path, "wb");
    if (!outfile) {
        fprintf(stderr, "Error opening output jpeg file %s\n!", output_path);
        return;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1; // grayscale
    cinfo.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width;

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
}
// Function to write JPEG image file
int write_jpeg(const char *filename, unsigned char *image, int width, int height, int quality) {
    FILE *output_file = fopen(filename, "wb");

    if (!output_file) {
        fprintf(stderr, "Error: Could not open JPEG file %s\n", filename);
        return -1;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, output_file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    unsigned char *rowptr;
    while (cinfo.next_scanline < cinfo.image_height) {
        rowptr = image + cinfo.next_scanline * width * 3;
        jpeg_write_scanlines(&cinfo, &rowptr, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(output_file);

    return 0;
}

int main() {
    const char *input_filename = "input.jpg";
    char *gray_filename = "gray.jpg";
    const char *sepia_filename = "sepia.jpg";
    const char *neg_filename = "neg.jpg";

    unsigned char *rgb_image, *gray_image, *sepia_image, *neg_image;
    int width, height;

    // Read input JPEG image
    if (read_jpeg(input_filename, &rgb_image, &width, &height) < 0) {
        return -1;
    }

    // Allocate memory for output images
    gray_image = (unsigned char *) malloc(width * height * sizeof(unsigned char));
    sepia_image = (unsigned char *) malloc(width * height * 3 * sizeof(unsigned char));
    neg_image = (unsigned char *) malloc(width * height * 3 * sizeof(unsigned char));

    // Set CUDA device and allocate memory on GPU
    cudaSetDevice(0);
    unsigned char *d_rgb_image, *d_gray_image, *d_sepia_image, *d_neg_image;
    cudaMalloc(&d_rgb_image, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_gray_image, width * height * sizeof(unsigned char));
    cudaMalloc(&d_sepia_image, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_neg_image, width * height * 3 * sizeof(unsigned char));

    // Copy input image to GPU memory
    cudaMemcpy(d_rgb_image, rgb_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set grid and block sizes for CUDA kernels
    int flag=0;
    dim3 block(32, 32, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    int choice = 1;
    while (true){
        printf("What do you want to perform:\n1) Convert to Grey\n2) Convert to Sepia\n3) Convert to Negative\n4)Exit\nEnter your choice: ");
        scanf("%d", &choice);

        switch(choice){
            case 1:
                // Convert RGB image to grayscale using CUDA kernel
                grayscale<<<grid, block>>>(d_rgb_image, d_gray_image, width, height);

                // Copy grayscale image from GPU to host memory
                cudaMemcpy(gray_image, d_gray_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

                write_jpeg_grayscale(gray_image, width, height,gray_filename);
                break;
            case 2:
              // Convert RGB image to sepia using CUDA kernel
              sepia<<<grid, block>>>(d_rgb_image, d_sepia_image, width, height);  

              // Copy sepia image from GPU to host memory
              cudaMemcpy(sepia_image, d_sepia_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

              write_jpeg(sepia_filename, sepia_image, width, height, 100);
              break;
            case 3:
                // Convert RGB image to negative using CUDA kernel
                negative<<<grid, block>>>(d_rgb_image, d_neg_image, width, height);

                // Copy negative image from GPU to host memory
                cudaMemcpy(neg_image, d_neg_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

                write_jpeg(neg_filename, neg_image, width, height, 100);
                break;
            case 4:
                flag=1;
                break;
            default:
                flag=1;
                break;
        }
        if(flag==1)
        {
            break;
        }
    }

    
    
    
    

    // Free GPU memory
    cudaFree(d_rgb_image);
    cudaFree(d_gray_image);
    cudaFree(d_sepia_image);
    cudaFree(d_neg_image);

    // Free host memory
    free(rgb_image);
    free(gray_image);
    free(sepia_image);
    free(neg_image);

    return 0;
}