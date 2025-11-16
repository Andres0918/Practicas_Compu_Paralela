#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;
using namespace std::chrono;

#define CHECK_CUDA(call) { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); } }

//////////////////////////
// Generar kernel Emboss de tamaño K
void generate_emboss_kernel(int K, vector<float>& kernel) {
    kernel.resize(K * K);
    int r = K / 2;

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            int y = i - r;
            int x = j - r;

            // Emboss diagonal (de esquina superior izquierda a inferior derecha)
            // La idea es: valores negativos en la parte superior-izquierda,
            // cero en el centro, positivos en la parte inferior-derecha

            float diagonal_dist = (x + y) / (float)(2 * r);

            if (diagonal_dist < -0.3f) {
                // Zona superior-izquierda: valores negativos
                float weight = 1.0f / (1.0f + abs(x) + abs(y));
                kernel[i * K + j] = -2.0f * weight;
            } else if (diagonal_dist > 0.3f) {
                // Zona inferior-derecha: valores positivos
                float weight = 1.0f / (1.0f + abs(x) + abs(y));
                kernel[i * K + j] = 2.0f * weight;
            } else {
                // Zona central: transición suave
                kernel[i * K + j] = diagonal_dist * 3.0f;
            }
        }
    }

    // Normalizar para que la suma sea cercana a 0 (importante para emboss)
    float sum = 0.0f;
    for (int i = 0; i < K * K; i++) {
        sum += kernel[i];
    }
    for (int i = 0; i < K * K; i++) {
        kernel[i] -= sum / (K * K);
    }
}

//////////////////////////
// CPU Emboss
void emboss_cpu(const float* in, float* out, int W, int H,
                const float* kernel, int K) {
    int r = K/2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = -r; ky <= r; ++ky) {
                int yy = y + ky;
                if (yy < 0 || yy >= H) continue;
                for (int kx = -r; kx <= r; ++kx) {
                    int xx = x + kx;
                    if (xx < 0 || xx >= W) continue;
                    float val = in[yy * W + xx];
                    sum += val * kernel[(ky + r)*K + (kx + r)];
                }
            }
            // Agregar offset de 128 para centrar en gris medio
            out[y * W + x] = sum + 128.0f;
        }
    }
}

//////////////////////////
// GPU kernel
__global__ void emboss_kernel_gpu(const float* in, float* out,
                                  const float* kernel, int W, int H, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int r = K / 2;
    float sum = 0.0f;

    for (int ky = -r; ky <= r; ++ky) {
        int yy = y + ky;
        if (yy < 0 || yy >= H) continue;
        for (int kx = -r; kx <= r; ++kx) {
            int xx = x + kx;
            if (xx < 0 || xx >= W) continue;
            float val = in[yy * W + xx];
            sum += val * kernel[(ky + r) * K + (kx + r)];
        }
    }

    // Agregar offset de 128 para centrar en gris medio
    out[y * W + x] = sum + 128.0f;
}

//////////////////////////
// Save float image (0..255) with clamping
bool save_image_u8(const string &path, const float* imgf, int W, int H) {
    vector<unsigned char> buf(W*H);
    for (int i = 0; i < W*H; ++i) {
        float v = imgf[i];
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        buf[i] = (unsigned char)(v + 0.5f);
    }
    int ok = stbi_write_png(path.c_str(), W, H, 1, buf.data(), W);
    return ok != 0;
}

//////////////////////////
// MAIN
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s imagen_entrada.(jpg/png)\n", argv[0]);
        return 1;
    }

    string infile = argv[1];
    int W, H, n;
    unsigned char* data = stbi_load(infile.c_str(), &W, &H, &n, 1);
    if (!data) {
        fprintf(stderr, "Error al cargar %s\n", infile.c_str());
        return 1;
    }
    printf("Imagen cargada: %s (%dx%d)\n", infile.c_str(), W, H);

    vector<float> imgf(W*H);
    for (int i = 0; i < W*H; ++i) imgf[i] = float(data[i]);
    stbi_image_free(data);

    // Tamaños de kernel a probar
    vector<int> kernel_sizes = {9, 21, 65};

    // Reservar memoria GPU para imagen
    float *d_in, *d_out;
    size_t imgsz = W * H * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_in, imgsz));
    CHECK_CUDA(cudaMalloc(&d_out, imgsz));
    CHECK_CUDA(cudaMemcpy(d_in, imgf.data(), imgsz, cudaMemcpyHostToDevice));

    printf("\n=================================================\n");
    printf("FILTRO EMBOSS - COMPARACIÓN DE TAMAÑOS DE KERNEL\n");
    printf("=================================================\n\n");

    for (int K : kernel_sizes) {
        printf("--- KERNEL SIZE: %dx%d ---\n", K, K);

        // Generar kernel Emboss
        vector<float> emboss_kernel;
        generate_emboss_kernel(K, emboss_kernel);

        vector<float> out_cpu(W*H, 0.0f);
        vector<float> out_gpu(W*H, 0.0f);

        // CPU
        auto t0 = high_resolution_clock::now();
        emboss_cpu(imgf.data(), out_cpu.data(), W, H, emboss_kernel.data(), K);
        auto t1 = high_resolution_clock::now();
        double cpu_ms = duration_cast<milliseconds>(t1 - t0).count();
        printf("CPU time: %.3f ms\n", cpu_ms);

        // GPU
        float *d_kernel;
        size_t ksz = K * K * sizeof(float);
        CHECK_CUDA(cudaMalloc(&d_kernel, ksz));
        CHECK_CUDA(cudaMemcpy(d_kernel, emboss_kernel.data(), ksz, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));

        emboss_kernel_gpu<<<grid, block>>>(d_in, d_out, d_kernel, W, H, K);

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float gpu_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_ms, start, stop));

        CHECK_CUDA(cudaMemcpy(out_gpu.data(), d_out, imgsz, cudaMemcpyDeviceToHost));

        printf("GPU time: %.3f ms\n", gpu_ms);
        printf("Speedup: %.2fx\n", cpu_ms / gpu_ms);

        // Guardar resultados
        char fname_cpu[100], fname_gpu[100];
        sprintf(fname_cpu, "emboss_cpu_%dx%d.png", K, K);
        sprintf(fname_gpu, "emboss_gpu_%dx%d.png", K, K);

        save_image_u8(fname_cpu, out_cpu.data(), W, H);
        save_image_u8(fname_gpu, out_gpu.data(), W, H);
        printf("Guardado: %s y %s\n\n", fname_cpu, fname_gpu);

        // Liberar kernel
        CHECK_CUDA(cudaFree(d_kernel));
    }

    // Liberar memoria GPU
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    printf("=================================================\n");
    printf("Procesamiento completado.\n");
    printf("=================================================\n");

    return 0;
}