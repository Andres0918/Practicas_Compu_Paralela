import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.concurrent.*;

public class Main {

    public static void main(String[] args) throws Exception {
        // -------------------- Parámetros editables --------------------
        String inputPath = "C:\\Users\\s3_xc\\OneDrive\\Desktop\\entrada.jpg";
        String outputName = "salida_emboss651.png";
        int kernelSize = 65;       // usa 9, 21 o 65
        int numThreads = 8;
        // --------------------------------------------------------------

        long t0Total = System.nanoTime();

        // 1) Cargar imagen
        BufferedImage input = ImageIO.read(new File(inputPath));
        if (input == null) {
            System.out.println("No se pudo leer la imagen.");
            return;
        }

        final int width = input.getWidth();
        final int height = input.getHeight();
        System.out.println("Imagen cargada: " + inputPath);
        System.out.println("Resolución: " + width + "x" + height);

        // 2) Crear kernel EMBOSS variable
        float[] kernel = generateEmbossKernel(kernelSize);
        int kCenter = kernelSize / 2;

        // 3) Convertir imagen a arreglo
        int[] src = new int[width * height];
        input.getRGB(0, 0, width, height, src, 0, width);
        int[] dst = new int[width * height];

        // 4) Procesamiento paralelo
        long t0 = System.nanoTime();
        ExecutorService pool = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();

        int rowsPerBlock = (height + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; t++) {
            int startY = t * rowsPerBlock;
            int endY = Math.min(height, startY + rowsPerBlock);
            if (startY >= endY) break;

            final int fStart = startY;
            final int fEnd = endY;

            futures.add(pool.submit(() ->
                    convolveBlock(src, dst, width, height, kernel, kernelSize, kCenter, fStart, fEnd)
            ));
        }

        for (Future<?> f : futures) f.get();
        pool.shutdown();
        long t1 = System.nanoTime();

        // 5) Guardar resultado
        BufferedImage output = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        output.setRGB(0, 0, width, height, dst, 0, width);
        ImageIO.write(output, "png", new File(outputName));

        long t1Total = System.nanoTime();

        System.out.println("Filtro: EMBOSS " + kernelSize + "x" + kernelSize);
        System.out.println("Hilos: " + numThreads);
        System.out.printf("Tiempo convolución: %.3f ms%n", (t1 - t0) / 1e6);
        System.out.printf("Tiempo total: %.3f ms%n", (t1Total - t0Total) / 1e6);
        System.out.println("Salida guardada en: " + outputName);
    }

    // --------------------------------------------------------------
    // Emboss genérico N×N (gradiente diagonal)
    // --------------------------------------------------------------
    private static float[] generateEmbossKernel(int size) {
        if (size % 2 == 0) throw new IllegalArgumentException("Kernel debe ser impar");

        float[] kernel = new float[size * size];

        int center = size / 2;
        int index = 0;

        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {

                // gradiente diagonal
                float value = (x + y) - (2 * center);

                kernel[index++] = value;
            }
        }

        return kernel;
    }

    // --------------------------------------------------------------
    // Convolución por bloques
    // --------------------------------------------------------------
    private static void convolveBlock(
            int[] src, int[] dst,
            int width, int height,
            float[] kernel, int kSize, int kCenter,
            int startY, int endY
    ) {
        for (int y = startY; y < endY; y++) {
            for (int x = 0; x < width; x++) {

                double rAcc = 0, gAcc = 0, bAcc = 0;
                int kIdx = 0;

                for (int ky = 0; ky < kSize; ky++) {
                    int sy = clamp(y + (ky - kCenter), 0, height - 1);
                    int base = sy * width;

                    for (int kx = 0; kx < kSize; kx++, kIdx++) {
                        int sx = clamp(x + (kx - kCenter), 0, width - 1);

                        int argb = src[base + sx];
                        int r = (argb >> 16) & 0xFF;
                        int g = (argb >> 8) & 0xFF;
                        int b = argb & 0xFF;

                        float kv = kernel[kIdx];

                        rAcc += r * kv;
                        gAcc += g * kv;
                        bAcc += b * kv;
                    }
                }

                // Emboss: centramos suma en 128
                int rOut = clamp((int)(rAcc + 128), 0, 255);
                int gOut = clamp((int)(gAcc + 128), 0, 255);
                int bOut = clamp((int)(bAcc + 128), 0, 255);

                int a = (src[y * width + x] >> 24) & 0xFF;
                dst[y * width + x] = (a << 24) | (rOut << 16) | (gOut << 8) | bOut;
            }
        }
    }

    // --------------------------------------------------------------
    // Auxiliar
    // --------------------------------------------------------------
    private static int clamp(int v, int lo, int hi) {
        return (v < lo) ? lo : Math.min(v, hi);
    }
}
