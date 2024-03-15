import numpy as np
import time


np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0/values[i]
    return output

values = np.random.randint(1,10,50000)
start = time.time()
out = compute_reciprocals(values)
total_time = time.time() - start
print(values)
print(out)
print(total_time)

#ufuncversion
start = time.time()
out = 1/values
total_time = time.time() - start
print(values)
print(out)
print(total_time)



values =   np.random.randint(1,10,10)
print(np.add(1,values))


