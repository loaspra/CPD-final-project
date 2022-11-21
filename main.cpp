#include <omp.h>
#include "RngStream.h"
#include <iostream>
#include <math.h>

int main()
{
    // numero de procesos
    const int nP = 4;
    // maximo valor random posible con el que se trabaja
    const int max_rand = 1;
    omp_set_num_threads(nP);
    // Inicializando RngStream  
    unsigned long seed[6] = {1806547166 , 3311292359 ,
                                643431772 , 1162448557 ,
                                3335719306 , 4161054083};
    
    RngStream::SetPackageSeed(seed);
    RngStream RngArray[nP];

    int world_rank, N, is_triangle = 0, is_obtuse = 0;

    // numero de experimientos independidentes para la simulacion Monte-Carlo
    N = 10;
    double rand = -1.1l, a, b, s;
    double segmentos[3] = {0, 0, 0};

    #pragma omp parallel private(world_rank, s, a, b, segmentos) firstprivate(rand)
    {
        world_rank = omp_get_thread_num();
        #pragma omp for reduction(+ : is_triangle, is_obtuse)
        for (size_t i = 0; i < N; i++)
        {
            #pragma omp critical
            {
                // se generaon los 2 puntos aleatorios en el rango {0,1}
                a = RngArray[world_rank].RandU01();
                b = RngArray[world_rank].RandU01();
                // volviendo a los segmentos positivos
                // armando los segmentos
                if(a > b){
                    segmentos[0] = b;
                    segmentos[1] = a - b;
                    segmentos[2] = max_rand - a;
                }else{
                    segmentos[0] = a;
                    segmentos[1] = b - a;
                    segmentos[2] = max_rand - b;
                }
                // El segmento tiene una longitud de max_rand (1), por lo tanto:
                // a + b + c = 1 aka las sumas de los segmentos deben de dar la longitud total
                if(segmentos[0] <= 0.5 && segmentos[1] <= 0.5 && segmentos[2] <= 0.5)
                {
                    printf("P: %d Segmentos: a: %f b: %f c: %f\n", world_rank, segmentos[0], segmentos[1], segmentos[2]);
                    // si es que ninguno de los segmentos es mayor a la mitad del segmento total
                    // es un triangulo
                    is_triangle++;

                    // calcular el lado con mayor longitud "s"
                    if(segmentos[0] > segmentos[1])
                        if(segmentos[0] > segmentos[2])
                        {
                            s = segmentos[0];
                            a = segmentos[1];
                            b = segmentos[2];
                        }
                        else
                        {
                            s = segmentos[2];
                            a = segmentos[1];
                            b = segmentos[0];
                        }
                    else
                        if(segmentos[1] > segmentos[2])
                        {
                            s = segmentos[1];
                            a = segmentos[0];
                            b = segmentos[2];
                        }
                        else
                        {
                            s = segmentos[2];
                            a = segmentos[1];
                            b = segmentos[0];
                        }
                    // ver si el triangulo es obtuso
                    if(pow(s, 2) >= (pow(a, 2) + pow(b, 2)))
                        is_obtuse++;
                }
            }
        }
    }
    
    printf("Triu: %d, Obtuse: %d\n", is_triangle, is_obtuse);
    return 0;
}