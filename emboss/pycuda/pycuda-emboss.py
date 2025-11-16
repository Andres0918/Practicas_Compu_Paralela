import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from PIL import Image
import time

print("PyCUDA listo. Memoria libre:", cuda.mem_get_info())

# Kernel CUDA para emboss
mod = SourceModule("""
__global__ void emboss_kernel(float *input, float *output, float *kernel, 
                              int width, int height, int kernel_size)
{
    // Calcular posición del pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int r = kernel_size / 2;
    float sum = 0.0f;

    // Aplicar convolución
    for (int ky = -r; ky <= r; ky++) {
        int yy = y + ky;
        if (yy < 0 || yy >= height) continue;

        for (int kx = -r; kx <= r; kx++) {
            int xx = x + kx;
            if (xx < 0 || xx >= width) continue;

            float pixel_val = input[yy * width + xx];
            float kernel_val = kernel[(ky + r) * kernel_size + (kx + r)];
            sum += pixel_val * kernel_val;
        }
    }

    // Agregar offset de 128 y guardar resultado
    output[y * width + x] = sum + 128.0f;
}
""")


def generate_emboss_kernel(K):
    """Genera kernel emboss de tamaño KxK"""
    kernel = np.zeros((K, K), dtype=np.float32)
    r = K // 2

    for i in range(K):
        for j in range(K):
            y = i - r
            x = j - r

            # Distancia diagonal
            diagonal_dist = (x + y) / (2.0 * r) if r > 0 else 0

            if diagonal_dist < -0.3:
                # Zona superior-izquierda: valores negativos
                weight = 1.0 / (1.0 + abs(x) + abs(y))
                kernel[i, j] = -2.0 * weight
            elif diagonal_dist > 0.3:
                # Zona inferior-derecha: valores positivos
                weight = 1.0 / (1.0 + abs(x) + abs(y))
                kernel[i, j] = 2.0 * weight
            else:
                # Zona central: transición suave
                kernel[i, j] = diagonal_dist * 3.0

    # Normalizar para que suma ≈ 0
    kernel_sum = np.sum(kernel)
    kernel -= kernel_sum / (K * K)

    return kernel


def load_image_grayscale(path):
    """Carga imagen y la convierte a escala de grises"""
    img = Image.open(path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    return img_array


def save_image(path, img_array):
    """Guarda imagen desde array float32 (0-255)"""
    img_array = np.clip(img_array, 0, 255)
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)
    print(f"Guardado: {path}")


def emboss_cpu(image, kernel):
    """Implementación CPU del filtro emboss - OPTIMIZADA con NumPy"""
    from scipy.ndimage import convolve

    # Usar convolución optimizada de scipy (implementada en C)
    output = convolve(image, kernel, mode='constant', cval=0.0)

    # Agregar offset de 128
    output += 128.0

    return output


def emboss_gpu(image, kernel, block_config, grid_config):
    """Implementación GPU del filtro emboss con configuración específica"""
    H, W = image.shape
    K = kernel.shape[0]

    # Obtener función del kernel
    emboss_func = mod.get_function("emboss_kernel")

    # Preparar datos
    input_gpu = cuda.mem_alloc(image.nbytes)
    output_gpu = cuda.mem_alloc(image.nbytes)
    kernel_gpu = cuda.mem_alloc(kernel.nbytes)

    # Copiar a GPU
    cuda.memcpy_htod(input_gpu, image)
    cuda.memcpy_htod(kernel_gpu, kernel.flatten())

    # Medir tiempo
    start = cuda.Event()
    end = cuda.Event()

    start.record()

    # Ejecutar kernel
    emboss_func(
        input_gpu, output_gpu, kernel_gpu,
        np.int32(W), np.int32(H), np.int32(K),
        block=block_config, grid=grid_config
    )

    end.record()
    end.synchronize()

    # Tiempo en milisegundos
    gpu_time = start.time_till(end)

    # Copiar resultado de vuelta
    output = np.empty_like(image)
    cuda.memcpy_dtoh(output, output_gpu)

    # Liberar memoria
    input_gpu.free()
    output_gpu.free()
    kernel_gpu.free()

    return output, gpu_time


def main():
    # Cargar imagen
    import sys
    if len(sys.argv) < 2:
        print("Uso: python emboss_pycuda.py imagen.jpg")
        return

    img_path = sys.argv[1]
    image = load_image_grayscale(img_path)
    H, W = image.shape
    print(f"\nImagen cargada: {img_path} ({W}x{H})")

    # Tamaños de kernel a probar
    kernel_sizes = [9, 21, 65]

    print("\n" + "=" * 70)
    print("FILTRO EMBOSS - COMPARACIÓN PyCUDA")
    print("=" * 70)

    for K in kernel_sizes:
        print(f"\n--- KERNEL SIZE: {K}x{K} ---")

        # Generar kernel
        kernel = generate_emboss_kernel(K)

        # CPU
        print("\n🖥️  CPU:")
        start_cpu = time.time()
        output_cpu = emboss_cpu(image, kernel)
        end_cpu = time.time()
        cpu_time = (end_cpu - start_cpu) * 1000  # convertir a ms
        print(f"   Tiempo: {cpu_time:.3f} ms")

        # Guardar resultado CPU
        save_image(f"emboss_cpu_{K}x{K}.png", output_cpu)

        # Diferentes configuraciones de grid/block
        configs = [
            # (block, grid, nombre)
            # ((16, 16, 1), ((W + 15) // 16, (H + 15) // 16, 1), "16x16 threads/block"),
            # ((32, 32, 1), ((W + 31) // 32, (H + 31) // 32, 1), "32x32 threads/block"),
            # ((8, 8, 1), ((W + 7) // 8, (H + 7) // 8, 1), "8x8 threads/block"),
            # ((64, 1, 1), ((W + 63) // 64, H, 1), "64x1 threads/block"),
            # ((1, 64, 1), (W, (H + 63) // 64, 1), "1x64 threads/block"),
            # 12 hilos en 1 bloque - necesitas calcular cuántos bloques para cubrir la imagen
            ((12, 1, 1), ((W + 11) // 12, H, 1), "12x1 threads/block (12 hilos)"),
        ]

        print("\n GPU - Diferentes configuraciones:")
        best_time = float('inf')
        best_config = None

        for block, grid, name in configs:
            try:
                output_gpu, gpu_time = emboss_gpu(image, kernel, block, grid)
                speedup = cpu_time / gpu_time
                print(f"   {name:25s} | Tiempo: {gpu_time:7.3f} ms | Speedup: {speedup:.2f}x")

                if gpu_time < best_time:
                    best_time = gpu_time
                    best_config = name
                    best_output = output_gpu
            except cuda.Error as e:
                print(f"   {name:25s} | ❌ Error: {e}")

        print(f"\n   ⭐ Mejor configuración: {best_config} ({best_time:.3f} ms)")

        # Guardar mejor resultado GPU
        save_image(f"emboss_gpu_{K}x{K}.png", best_output)
        print()

    print("=" * 70)
    print("Procesamiento completado")
    print("=" * 70)


if __name__ == "__main__":
    main()