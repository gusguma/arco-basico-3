///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	BASICO 3 : Hilos y Bloques
/// Autor:		Gustavo Gutierrez Martin
/// Fecha:		Octubre 2022
///////////////////////////////////////////////////////////////////////////

/// dependencias ///
#include <cstdio>
#include <cstdlib>
#include <ctime>

/// constantes ///
#define MB (1<<20) /// MiB = 2^20
#define MAX_THREADS_PER_BLOCK 10
#define MAX_INT_NUMBER 2147483647

/// muestra por consola que no se ha encontrado un dispositivo CUDA
int getErrorDevice();
/// muestra los datos de los dispositivos CUDA encontrados
int getDataDevice(int deviceCount);
/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);
/// muestra por pantalla las propiedades del dispositivo CUDA
int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties);
/// solicita al usuario el número de elementos que se sumaran
int requestNumberOfItems(int *numberOfItems);
/// inicializa el array del host
int loadHostData(int *hst_vector1, int *hst_vector2, int numberOfItems);
/// transfiere los datos del host al device
int dataTransferToDevice(int *hst_vector1, int *dev_vector1, int numberOfItems);
/// realiza la suma de los arrays en el device
__global__ void suma(const int *dev_vector1, int *dev_vector2, int *dev_result, int numberOfItems);
/// transfiere los datos del device al host
int dataTransferToHost(int *hst_result, int *hst_vector2, int *dev_result, int *dev_vector2, int numberOfItems );
/// muestra por pantalla los datos del host
int printData(int *hst_vector1, int *hst_vector2, int *hst_result, int numOfItems);
/// función que muestra por pantalla la salida del programa
int getAppOutput();

int main() {
    int deviceCount;
    int numberOfItems = 0;
    int blocks = 0;
    int *hst_vector1,*hst_vector2,*hst_result;
    int *dev_vector1,*dev_vector2,*dev_result;

    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        return getErrorDevice();
    } else {
        /// mostramos los datos de los dispositivos CUDA encontrados
        getDataDevice(deviceCount);
    }
    /// solicitamos al usuario la cantidad de elementos
    requestNumberOfItems(&numberOfItems);
    /// reserva del espacio de memoria en el host
    hst_vector1 = (int*)malloc( numberOfItems * sizeof(int) );
    hst_vector2 = (int*)malloc( numberOfItems * sizeof(int) );
    hst_result = (int*)malloc( numberOfItems * sizeof(int) );
    /// reserva del espacio de memoria en el device
    cudaMalloc( (void**)&dev_vector1, numberOfItems * sizeof(int) );
    cudaMalloc( (void**)&dev_vector2, numberOfItems * sizeof(int) );
    cudaMalloc( (void**)&dev_result, numberOfItems * sizeof(int) );
    /// cargamos los datos iniciales en el host
    loadHostData(hst_vector1, hst_vector2, numberOfItems);
    /// transferimos los datos del host al device
    dataTransferToDevice(hst_vector1, dev_vector1, numberOfItems);
    /// calculamos el número de bloques a lanzar
    blocks = (int) numberOfItems/MAX_THREADS_PER_BLOCK + 1;
    /// mostramos los datos con los que llamamos al device
    printf("Lanzamiento de: %d bloques y %d hilos (%d hilos) \n", blocks, MAX_THREADS_PER_BLOCK, blocks*MAX_THREADS_PER_BLOCK);
    /// sumamos los items
    suma<<< blocks, MAX_THREADS_PER_BLOCK >>>(dev_vector1, dev_vector2, dev_result, numberOfItems);
    /// transferimos los datos del device al host
    dataTransferToHost(hst_result,hst_vector2,dev_result,dev_vector2,numberOfItems);
    /// muestra por pantalla los datos del host
    printData(hst_vector1,hst_vector2,hst_result,numberOfItems);
    /// función que muestra por pantalla la salida del programa
    getAppOutput();
    /// liberamos los recursos del device
    cudaFree(dev_vector1);
    cudaFree(dev_vector2);
    cudaFree(dev_result);
    return 0;
}

int getErrorDevice() {
    printf("¡No se ha encontrado un dispositivo CUDA!\n");
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 1;
}

int getDataDevice(int deviceCount) {
    printf("Se han encontrado %d dispositivos CUDA:\n", deviceCount);
    for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
        ///obtenemos las propiedades del dispositivo CUDA
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceID);
        getDeviceProperties(deviceID, getCudaCores(deviceProp), deviceProp);
    }
    return 0;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties) {
    int SM = cudaProperties.multiProcessorCount;
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceId, cudaProperties.name);
    printf("***************************************************\n");
    printf("- Capacidad de Computo            \t: %d.%d\n", cudaProperties.major, cudaProperties.minor);
    printf("- No. de MultiProcesadores        \t: %d \n", SM);
    printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
    printf("- Memoria Global (total)          \t: %zu MiB\n", cudaProperties.totalGlobalMem / MB);
    printf("- No. maximo de Hilos (por bloque)\t: %d\n", cudaProperties.maxThreadsPerBlock);
    printf("  > [Eje x -> %d]\n", cudaProperties.maxThreadsDim[0]);
    printf("  > [Eje y -> %d]\n", cudaProperties.maxThreadsDim[1]);
    printf("  > [Eje z -> %d]\n", cudaProperties.maxThreadsDim[2]);
    printf("- No. maximo de Bloques (por eje)\n");
    printf("  > [Eje x -> %d]\n", cudaProperties.maxGridSize[0]);
    printf("  > [Eje y -> %d]\n", cudaProperties.maxGridSize[1]);
    printf("  > [Eje z -> %d]\n", cudaProperties.maxGridSize[2]);
    printf("***************************************************\n");
    return 0;
}

int requestNumberOfItems(int *numberOfItems) {
    int status = 0;
    while (status == 0) {
        printf("Introduce el numero de elementos: \n");
        scanf_s("%d", numberOfItems);
        if (MAX_INT_NUMBER < *numberOfItems || *numberOfItems < 0) {
            printf("Numero maximo de elementos superado: %d \n", MAX_INT_NUMBER);
        } else {
            printf("El numero de elementos elegido es: %d \n", *numberOfItems);
            status = 1;
        }
    }
    return 0;
}

int loadHostData(int *hst_vector1, int *hst_vector2, int numberOfItems) {
    srand ( (int)time(nullptr) );
    for (int i=0; i<numberOfItems; i++)  {
        /// inicializamos hst_vector1 con numeros aleatorios entre 0 y 1
        hst_vector1[i] = (int) rand() % 10;
        /// inicializamos hst_vector2 con ceros
        hst_vector2[i] = 0;
    }
    return 0;
}

int dataTransferToDevice(int *hst_vector1, int *dev_vector1, int numberOfItems ) {
    /// transfiere datos de hst_A a dev_A
    cudaMemcpy(dev_vector1,hst_vector1, numberOfItems * sizeof(int),cudaMemcpyHostToDevice);
    return 0;
}

__global__ void suma(const int *dev_vector1, int *dev_vector2, int *dev_result, int numberOfItems) {
    /// identificador del hilo
    unsigned int idGlobal = threadIdx.x + blockDim.x * blockIdx.x;
    if (numberOfItems > idGlobal) {
        /// inicializamos el vector 2
        dev_vector2[idGlobal] = dev_vector1[numberOfItems - idGlobal - 1];
        /// sumamos los dos vectores y escribimos el resultado
        dev_result[idGlobal] = dev_vector1[idGlobal] + dev_vector2[idGlobal];
    }
}

int dataTransferToHost(int *hst_result, int *hst_vector2, int *dev_result, int *dev_vector2, int numberOfItems ) {
    /// transfiere datos de dev_vector2 a hst_vector2
    cudaMemcpy(hst_vector2, dev_vector2, numberOfItems * sizeof(int), cudaMemcpyDeviceToHost);
    /// transfiere datos de dev_result a hst_result
    cudaMemcpy(hst_result,dev_result,numberOfItems * sizeof(int),cudaMemcpyDeviceToHost);
    return 0;
}

int printData(int *hst_vector1, int *hst_vector2, int *hst_result, int numOfItems) {
    printf("VECTOR 1:\n");
    for (int i = 0; i < numOfItems; i++)  {
        printf("%d ", hst_vector1[i]);
    }
    printf("\n");
    printf("VECTOR 2:\n");
    for (int i = 0; i < numOfItems; i++)  {
        printf("%d ", hst_vector2[i]);
    }
    printf("\n");
    printf("RESULTADO:\n");
    for (int i = 0; i < numOfItems; i++)  {
        printf("%d ", hst_result[i]);
    }
    printf("\n");
    return 0;
}

int getAppOutput() {
    /// salida del programa
    time_t fecha;
    time(&fecha);
    printf("***************************************************\n");
    printf("Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}
