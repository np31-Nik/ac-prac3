# ac-prac3

//Intento de hacerlo en GPU
//no acabado

__global__ void calculadorNormales(float* resultado, TSurf aux, char dim) {
  int id = threadIdx.x + (blockIdx.x * blockDim.x);
	switch (dim) {
    case 'x':
     if (id < dim)
       {
       resultado[id] = v1[id] + v2[id];
       }


      break;

    case 'y':

      break;

    case 'z':

      break;
    }
}
