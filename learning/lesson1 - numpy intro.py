import numpy as np

mylist = [1,2,3,4,5,6,4,4,True,4, False, 1.0]
nplist = np.array(mylist, dtype="float32") # numpy data types correspond to C data types
nplist = np.array(mylist, dtype="int8")
print(mylist)
print(nplist)
nplist = np.array([range(i,i+3) for i in [2,3,4]])
print(nplist)
nplist = np.zeros(10, dtype=bool)

nplist = np.ones((3,4), dtype=bool)
print(nplist)

nplist = np.linspace(0,1,5) # 5 numbers equispaced in range 0 to 1
print(nplist)
nplist = np.random.random((2,2))
print(nplist)
nplist = np.random.normal(5,3,10) #10 numbers distributed about 5 with a sdev of 3
print(nplist)
nplist = np.random.randint(0,10,10) #10 ints randomly distributed between 0 and 10
print(nplist)
nplist = np.eye(3, dtype='bool') # identity matrix size 3
print(nplist)

nplist = np.random.randint(0,10,(3,5,3))
print ('array dims: ', nplist.ndim, nplist.shape, nplist.size) # num dims, shape, size

print(nplist[1][2])
nplist[1][2] = 10.9
print(nplist[1][2])

nplist = np.arange(10)
print(nplist[5::2])
subarray = nplist[4:7]
print(nplist[4])
subarray[0] = 99
print(nplist[4])
copiedarray = nplist[4:7].copy()
copiedarray[0] = 0
print(nplist[4])

nplist = np.random.randint(0,10,(3,3))
subarray = nplist[:2,1:2]
print(subarray)
subarray = nplist[:,1]
print(subarray)
subarray = nplist[1,:]
print(subarray)