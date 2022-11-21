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

    unsigned long seed[6] = {1806547166 , 3311292359 ,
                                643431772 , 1162448557 ,
                                3335719306 , 4161054083};
    
    RngStream::SetPackageSeed(seed);
    RngStream RngArray[nP];

    int world_rank, N, is_triangle = 0, is_obtuse = 0;
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
                a = RngArray[world_rank].RandU01();
                b = RngArray[world_rank].RandU01();
                if(a > b){
                    segmentos[0] = b;
                    segmentos[1] = a - b;
                    segmentos[2] = max_rand - a;
                    printf("p: %d segmentos: a: %f | b: %f | c: %f\n", world_rank, segmentos[0], segmentos[1], segmentos[2]);
                    if(b <= ((a - b) + (1 - a)) && (a - b) <= (b + 1 - a) && (1 - a) <= (a - b + 1 - a))
                    {    
                        is_triangle++;
                        
                        s = pow(segmentos[1], 2) + pow(segmentos[2], 2) - pow(segmentos[0], 2);
                        s = s/(2*segmentos[1]*segmentos[2]);
                        printf("p: %d %f\n", world_rank, s);
                        if(s <= 0)
                            is_obtuse++;
                        else{
                            s = pow(segmentos[0], 2) + pow(segmentos[2], 2) - pow(segmentos[1], 2);
                            s = s/(2*segmentos[0]*segmentos[2]);
                            printf("p: %d %f\n", world_rank, s);
                            if(s <= 0)
                                is_obtuse++;
                            else{
                            s = pow(segmentos[0], 2) + pow(segmentos[1], 2) - pow(segmentos[2], 2);
                            s = s/(2*segmentos[0]*segmentos[1]);
                            printf("p: %d %f\n", world_rank, s);
                            if(s <= 0)
                                is_obtuse++;
                            }
                        }
                    }
                }else{
                    segmentos[0] = a;
                    segmentos[1] = b - a;
                    segmentos[2] = 1 - b;
                }
            }
        }
    }
    printf("Triu: %d, Obtuse: %d\n", is_triangle, is_obtuse);
    return 0;
}
