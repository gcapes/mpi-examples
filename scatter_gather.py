from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    sendbuf = np.arange(15.0)

    # count: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)

    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(nprocs)]
    displ = np.array(displ)
else:
    sendbuf = None
    # initialize count on worker processes
    count = np.zeros(nprocs, dtype=int)
    displ = None

# broadcast count
comm.Bcast(count, root=0)

# initialize recvbuf on all processes
recvbuf = np.zeros(count[rank])

comm.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)
print('After Scatterv, process {} has data:'.format(rank), recvbuf)

sendbuf2 = recvbuf
recvbuf2 = np.zeros(sum(count))
comm.Gatherv(sendbuf2, [recvbuf2, count, displ, MPI.DOUBLE], root=0)

if comm.Get_rank() == 0:
    print('After Gatherv, process 0 has data:', recvbuf2)

