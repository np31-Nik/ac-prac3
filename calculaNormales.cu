/*----------------------------------------------------------------------------*/
/*  FICHERO:       calculaNormales.cu									          */
/*  AUTOR:         Jorge Azorin											  */
/*						
* 
*							TEAM DEEPWEB:
*								Nikita Polyanskiy
*								Serhii Vidernikov
*								Juan Carlos Sanchez Gonzalez
*								Juan Ramon Morales Gomez
*								Sohaib Laihi
*/
/*  RESUMEN												                      */
/*  ~~~~~~~												                      */
/* Ejercicio grupal para el cálculo de las normales de una superficie          */
/*----------------------------------------------------------------------------*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "calculaNormales.h"
#include <Windows.h>



#define ERROR_CHECK { cudaError_t err; if ((err = cudaGetLastError()) != cudaSuccess) { printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

typedef LARGE_INTEGER timeStamp;
double getTime();

/*----------------------------------------------------------------------------*/
/*  FUNCION A PARALELIZAR  (versión secuencial-CPU)  				          */
/*	Cálculo de las normales de una superficie definida por una                */
/*  una malla de vtotal x utotal puntos 3D                                    */
/*----------------------------------------------------------------------------*/
int CalculoNormalesCPU()
{
	    TPoint3D direct1, direct2, normal;
		int vecindadU[9]={-1,0,1,1,1,0,-1,-1,-1}; // Vecindad 8 + 1 para calcular todas las rectas
		int vecindadV[9]={-1,-1,-1,0,1,1,1,0,-1};
		int vV,vU;
		int numDir;
		int oKdir1,oKdir2;
		/* La vencidad es:
		*--*--*
		|  |  |
		*--X--*
		|  |  |
		*--*--*
		*/
		int cont=0;

		for (int u = 0; u<S.UPoints; u++)			// Recorrido de todos los puntos de la superficie
		{
			for (int v = 0; v<S.VPoints; v++)
			{
				normal.x=0;
				normal.y=0;
				normal.z=0;
				numDir=0;
				for (int nv = 0; nv < 8 ; nv ++)  // Para los puntos de la vecindad
				{
					    vV=v+vecindadV[nv];
						vU=u+vecindadU[nv];
						if (vV >= 0 && vU >=0 && vV<S.VPoints && vU<S.UPoints)
						{
							direct1.x=S.Buffer[v][u].x-S.Buffer[vV][vU].x;
							direct1.y=S.Buffer[v][u].y-S.Buffer[vV][vU].y;
							direct1.z=S.Buffer[v][u].z-S.Buffer[vV][vU].z;
							oKdir1=1;
						}else
						{
							direct1.x=0.0;
							direct1.y=0.0;
							direct1.z=0.0;
							oKdir1=0;
						}
						vV=v+vecindadV[nv+1];
						vU=v+vecindadU[nv+1];

						if (vV >= 0 && vU >=0 && vV<S.VPoints && vU<S.UPoints)
						{
						   direct2.x=S.Buffer[v][u].x-S.Buffer[vV][vU].x;
						   direct2.y=S.Buffer[v][u].y-S.Buffer[vV][vU].y;
						   direct2.z=S.Buffer[v][u].z-S.Buffer[vV][vU].z;
						   oKdir2=1;
						}else
						{
							direct2.x=0.0;
							direct2.y=0.0;
							direct2.z=0.0;
							oKdir2=0;
						}
						if (oKdir1 ==1 && oKdir2==1)
						{
						  normal.x +=  direct1.y*direct2.z-direct1.z*direct2.y;
						  normal.y += direct1.x*direct2.z-direct1.z*direct2.x;
						  normal.z += direct1.x*direct2.y-direct1.y*direct2.x;
						  numDir++;
						}
				}
				NormalUCPU[cont]=normal.x/(float)numDir;
				NormalVCPU[cont]=normal.y/(float)numDir;
				NormalWCPU[cont]=normal.z/(float)numDir;
				cont++;
			}
		}

	return OKCALC;									// Simulación CORRECTA
}

// ---------------------------------------------------------------
// ---------------------------------------------------------------
// FUNCION A IMPLEMENTAR POR EL GRUPO (paralelización de CalculoNormalesCPU)
// ---------------------------------------------------------------
// ---------------------------------------------------------------
__global__ void calculadorNormales(TPoint3D* d_Buffer, float* d_NormalUGPU, float* d_NormalVGPU, float* d_NormalWGPU, int U, int V) {

	int id = blockDim.x * blockIdx.x + threadIdx.x;

	int vecindad, okDir1, okDir2, numDir = 0, v, u, vV, vU;
	TPoint3D normal, direct1, direct2;

	if (U * V > id) {

		int d_vecindadU[9] = { -1, 0, 1, 1, 1, 0, -1, -1, -1 };
		int d_vecindadV[9] = { -1, -1, -1, 0, 1, 1, 1, 0, -1 };

		normal.x = 0;
		normal.y = 0;
		normal.z = 0;

		for (unsigned nv = 0; nv < 8; nv++) {

			v = id % V;
			u = id / V;

			vV = v + d_vecindadV[nv];
			vU = u + d_vecindadU[nv];

			if (vV >= 0 && vU >= 0 && vV < V && vU < U) {
				vecindad = vU * V + vV;

				direct1.x = d_Buffer[id].x - d_Buffer[vecindad].x;
				direct1.y = d_Buffer[id].y - d_Buffer[vecindad].y;
				direct1.z = d_Buffer[id].z - d_Buffer[vecindad].z;
				okDir1 = 1;
			}
			else
			{
				direct1.x = 0.0;
				direct1.y = 0.0;
				direct1.z = 0.0;
				okDir1 = 0;
			}

			vV = v + d_vecindadV[nv + 1];
			vU = v + d_vecindadU[nv + 1];

			if (vV >= 0 && vU >= 0 && vV < V && vU < U) {
				vecindad = vU * V + vV;
				direct2.x = d_Buffer[id].x - d_Buffer[vecindad].x;
				direct2.y = d_Buffer[id].y - d_Buffer[vecindad].y;
				direct2.z = d_Buffer[id].z - d_Buffer[vecindad].z;
				okDir2 = 1;
			}
			else
			{
				direct2.x = 0.0;
				direct2.y = 0.0;
				direct2.z = 0.0;
				okDir2 = 0;
			}

			if (okDir1 == 1 && okDir2 == 1) {
				normal.x += direct1.y * direct2.z - direct1.z * direct2.y;
				normal.y += direct1.x * direct2.z - direct1.z * direct2.x;
				normal.z += direct1.x * direct2.y - direct1.y * direct2.x;
				numDir++;
			}
		}

		d_NormalUGPU[id] = normal.x / (float)numDir;
		d_NormalVGPU[id] = normal.y / (float)numDir;
		d_NormalWGPU[id] = normal.z / (float)numDir;
	}
}

int CalculoNormalesGPU()
{
	unsigned U = S.UPoints, V = S.VPoints, k = 0;

	float* d_NormalVGPU, * d_NormalUGPU, * d_NormalWGPU;

	TPoint3D* h_Buffer, * d_Buffer;

	h_Buffer = (TPoint3D*)malloc(sizeof(TPoint3D) * U * V);

	for (unsigned i = 0; i < U; i++) {
		for (unsigned j = 0; j < V; j++) {
			h_Buffer[k] = S.Buffer[j][i];
			k++;
		}
	}

	cudaMalloc(&d_Buffer, sizeof(TPoint3D)* U * V);
	cudaMalloc(&d_NormalVGPU, sizeof(float) * U * V);
	cudaMalloc(&d_NormalUGPU, sizeof(float) * U * V);
	cudaMalloc(&d_NormalWGPU, sizeof(float) * U * V);

	cudaMemcpy(d_Buffer, h_Buffer, sizeof(TPoint3D) * U * V, cudaMemcpyHostToDevice);

	calculadorNormales << < U * V / 512 + 1, 512 >> > (d_Buffer, d_NormalUGPU, d_NormalVGPU, d_NormalWGPU, U, V);

	cudaMemcpy(NormalVGPU, d_NormalVGPU, U * V * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(NormalUGPU, d_NormalUGPU, U * V * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(NormalWGPU, d_NormalWGPU, U * V * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_Buffer);
	cudaFree(d_NormalVGPU);
	cudaFree(d_NormalUGPU);
	cudaFree(d_NormalWGPU);

	return OKCALC;
}
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------
 // ---------------------------------------------------------------

 // Declaraciones adelantadas de funciones
 int LeerSuperficie(const char *fichero);



////////////////////////////////////////////////////////////////////////////////
//PROGRAMA PRINCIPAL
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{


	double gpu_start_time, gpu_end_time;
	double cpu_start_time, cpu_end_time;

	/* Numero de argumentos */
	if (argc != 2)
	{
		fprintf(stderr, "Numero de parametros incorecto\n");
		fprintf(stderr, "Uso: %s superficie\n", argv[0]);
		return;
	}

	/* Apertura de Fichero */
	printf("Cálculo de las normales de la superficie...\n");
	/* Datos de la superficie */
	if (LeerSuperficie((char *)argv[1]) == ERRORCALC)
	{
		fprintf(stderr, "Lectura de superficie incorrecta\n");
		return;
	}
	int numPuntos;
	numPuntos=S.UPoints*S.VPoints;

	// Creación buffer resultados para versiones CPU y GPU
	NormalVCPU = (float*)malloc(numPuntos*sizeof(float));
	NormalUCPU = (float*)malloc(numPuntos*sizeof(float));
    NormalWCPU = (float*)malloc(numPuntos*sizeof(float));
	NormalVGPU = (float*)malloc(numPuntos*sizeof(float));
	NormalUGPU = (float*)malloc(numPuntos*sizeof(float));
	NormalWGPU = (float*)malloc(numPuntos*sizeof(float));

	/* Algoritmo a paralelizar */
	cpu_start_time = getTime();
	if (CalculoNormalesCPU() == ERRORCALC)
	{
		fprintf(stderr, "Cálculo CPU incorrecta\n");
		BorrarSuperficie();
		if (NormalVCPU != NULL) free(NormalVCPU);
		if (NormalUCPU != NULL) free(NormalUCPU);
	    if (NormalWCPU != NULL) free(NormalUCPU);
		if (NormalVGPU != NULL) free(NormalVGPU);
		if (NormalWGPU != NULL) free(NormalVGPU);
		if (NormalUGPU != NULL) free(NormalUGPU);		exit(1);
	}
	cpu_end_time = getTime();
	/* Algoritmo a implementar */
	gpu_start_time = getTime();
	if (CalculoNormalesGPU() == ERRORCALC)
	{
		fprintf(stderr, "Cálculo GPU incorrecta\n");
		BorrarSuperficie();
		if (NormalVCPU != NULL) free(NormalVCPU);
		if (NormalUCPU != NULL) free(NormalUCPU);
	    if (NormalWCPU != NULL) free(NormalUCPU);
		if (NormalVGPU != NULL) free(NormalVGPU);
		if (NormalUGPU != NULL) free(NormalUGPU);
		if (NormalVGPU != NULL) free(NormalVGPU);
		return;
	}
	gpu_end_time = getTime();
	// Comparación de corrección
	int comprobar = OKCALC;
	for (int i = 0; i<numPuntos; i++)
	{
		if (((int)NormalVCPU[i] * 1000 != (int)NormalVGPU[i] * 1000) || ((int)NormalUCPU[i] * 1000 != (int)NormalUGPU[i] * 1000) || ((int)NormalWCPU[i] * 1000 != (int)NormalWGPU[i] * 1000))
		{
			comprobar = ERRORCALC;
			fprintf(stderr, "Fallo en el punto %d, valor correcto V=%f U=%f W=%f\n", i, NormalVCPU[i], NormalUCPU[i],NormalWCPU[i]);
		}
	}
	// Impresion de resultados
	if (comprobar == OKCALC)
	{
		printf("Cálculo correcto!\n");

	}
	// Impresión de resultados
	printf("Tiempo ejecución GPU : %fs\n", \
		gpu_end_time - gpu_start_time);
	printf("Tiempo de ejecución en la CPU : %fs\n", \
		cpu_end_time - cpu_start_time);
	printf("Se ha conseguido un factor de aceleración %fx utilizando CUDA\n", (cpu_end_time - cpu_start_time) / (gpu_end_time - gpu_start_time));
	// Limpieza de buffers
	BorrarSuperficie();
	if (NormalVCPU != NULL) free(NormalVCPU);
	if (NormalUCPU != NULL) free(NormalUCPU);
    if (NormalWCPU != NULL) free(NormalWCPU);
	if (NormalVGPU != NULL) free(NormalVGPU);
	if (NormalUGPU != NULL) free(NormalUGPU);
	if (NormalWGPU != NULL) free(NormalWGPU);
	return;
}

int
main(int argc, char** argv)
{
	runTest(argc, argv);
	getchar();
}

/* Funciones auxiliares */
double getTime()
{
	timeStamp start;
	timeStamp dwFreq;
	QueryPerformanceFrequency(&dwFreq);
	QueryPerformanceCounter(&start);
	return double(start.QuadPart) / double(dwFreq.QuadPart);
}



/*----------------------------------------------------------------------------*/
/*	Función:  LeerSuperficie(char *fichero)						              */
/*													                          */
/*	          Lee los datos de la superficie de un fichero con formato .FOR   */
/*----------------------------------------------------------------------------*/
int LeerSuperficie(const char *fichero)
{
	int i, j, count;		/* Variables de bucle */
	int utotal,vtotal;		/* Variables de tamaño de superficie */
	FILE *fpin; 			/* Fichero */
	double x, y, z;

	/* Apertura de Fichero */
	if ((fpin = fopen(fichero, "r")) == NULL) return ERRORCALC;
	/* Lectura de cabecera */
	if (fscanf(fpin, "Ancho=%d\n", &utotal)<0) return ERRORCALC;
	if (fscanf(fpin, "Alto=%d\n", &vtotal)<0) return ERRORCALC;
	if (utotal*vtotal <= 0) return ERRORCALC;
	/* Localizacion de comienzo */
	if (feof(fpin)) return ERRORCALC;
	/* Inicialización de parametros geometricos */
	if (CrearSuperficie(utotal, vtotal) == ERRORCALC) return ERRORCALC;
	/* Lectura de coordenadas */
	count = 0;
	for (i = 0; i<utotal; i++)
	{
		for (j = 0; j<vtotal; j++)
		{
			if (!feof(fpin))
			{
				fscanf(fpin, "%lf %lf %lf\n", &x, &y, &z);
				S.Buffer[j][i].x = x;
				S.Buffer[j][i].y = y;
				S.Buffer[j][i].z = z;
				count++;
			}
			else break;
		}
	}
	fclose(fpin);
	if (count != utotal*vtotal) return ERRORCALC;
	return OKCALC;
}



