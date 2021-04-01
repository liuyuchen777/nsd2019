import time
import random
import _matrix

if __name__ == "__main__":
    print("start benchmark")
    A_nrow, A_ncol = 1000, 1000
    B_nrow, B_ncol = 1000, 1000
    total_size = A_nrow * B_ncol

    population_size = 1000000
    random_population = [int(random.randrange(100, 1000)) for _ in range(int(population_size))]
   #print(type(random_population))

    naive_time = 0
    mkl_time = 0
    loop_time = 5
    for i in range(loop_time):
        A_content = random.choices(random_population, k=A_nrow * A_ncol)
        A = _matrix.Matrix(A_nrow, A_ncol)
        A.set_buffer(A_content)

        B_content = random.choices(random_population, k=B_nrow * B_ncol)
        B = _matrix.Matrix(B_nrow, B_ncol)
        B.set_buffer(B_content)

        start = time.time()
        _matrix.multiply_naive(A, B)
        end = time.time()
        naive_time += end-start

        start = time.time()
        _matrix.multiply_mkl(A, B)
        end = time.time()
        mkl_time += end-start

    diff = naive_time / mkl_time
    s1 = "Strat multiply_naive, take "+str(loop_time)+" times in min =  "+str(naive_time)+ " seconds\n"
    s2 = "Strat multiply_mkl, take "+str(loop_time)+"times in min =  "+str(mkl_time)+ " seconds\n"
    s3 = "MKL speed-up over naive: "+str(diff)+" x\n"

    with open("performance.txt", "w") as f:
        f.writelines(s1)
        f.writelines(s2)
        f.writelines(s3)

    print("Finish benchmark...")