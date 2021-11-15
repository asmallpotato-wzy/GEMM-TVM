import tvm
import numpy
import tvm.testing
from tvm import te
import timeit
import time


# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.

M = 15
K = 512
N = 2048

nc = 1024
mr = 5
nr = 16

# The default tensor type in tvm
dtype = "float32"

# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "llvm -mcpu=znver2"
#target = "llvm"
dev = tvm.device(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

np_repeat = 10
np_runing_time = timeit.timeit(
    setup="import numpy\n"
    "M = " + str(M) + "\n"
    "K = " + str(K) + "\n"
    "N = " + str(N) + "\n"
    'dtype = "float32"\n'
    "a = numpy.random.rand(M, K).astype(dtype)\n"
    "b = numpy.random.rand(K, N).astype(dtype)\n",
    stmt="answer = numpy.dot(a, b)",
    number=np_repeat,
)
print("Numpy running time: %fms %f GFLOPS" % (1000.0 * np_runing_time / np_repeat, 2.0 * M * N * K / np_runing_time * np_repeat / 1e9 ))

answer = numpy.dot(a.numpy(), b.numpy())

time.sleep(1)


# Algorithm
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
k = te.reduce_axis((0, K), 'k')

packedB_pre = te.compute(
    (N//nr,K,nr), lambda no,k,ni: B[k,no*nr+ni],
    name = 'packedB_pre'
)

s1 = te.create_schedule(packedB_pre.op)
packB_func = tvm.build(s1, [B,packedB_pre], target=target, name='fpackedB')
assert  packB_func
packed_b = tvm.nd.array(numpy.zeros((N//nr,K,nr), dtype=dtype), dev)
packB_func(b, packed_b)
packedB = te.placeholder((N//nr,K,nr),name= "packedB")


packedC = te.compute(
    (N//nc,M,nc), lambda no,m,ni:te.sum(A[m,k]*packedB[(no*nc+ni)//nr,k,tvm.tir.indexmod(no*nc+ni,nr)],axis = k),
    name = 'packedC'
)

C = te.compute(
    (M,N), lambda m,n: packedC[n//nc,m,tvm.tir.indexmod(n,nc)]
)


s = te.create_schedule(C.op)

no,m,ni  = s[packedC].op.axis
k = s[packedC].op.reduce_axis[0]
mo, nio, mi, nii = s[packedC].tile(m,ni,mr,nr)
s[packedC].reorder(no,mo,nio,k,mi,nii)
cm, cn = s[C].op.axis
cno,cni = s[C].split(cn,nc)
s[C].reorder(cno,cm,cni)
s[packedC].compute_at(s[C],cno)

func = tvm.build(s, [A, packedB, C], target=target, name='mmult')
assert func
print(func.get_source("asm"))
print(tvm.lower(s, [A, packedB, C], simple_mode=True))



c = tvm.nd.array(numpy.zeros((M,N), dtype=dtype), dev)
func(a, packed_b, c)
#tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, dev, number=10,repeat=50)
mean_time = evaluator(a, packed_b, c).mean

print("Opt1: %fms, %f GFLOPS" % (mean_time * 1000, 2.0 * M * N * K / mean_time / 1e9))
'''
libpath = "/data/asm.so"
func.export_library(libpath)
'''
print((M*K+M*N+K*N)/1024*4)
