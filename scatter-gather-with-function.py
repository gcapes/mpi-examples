from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

print(rank, nprocs)

def process_data(data):
    print('Process {} has data:'.format(rank), data)
    data = data + rank
    return data

if rank == 0:
    data = np.arange(5)
    data = np.tile(data, (10, 1))

    # determine the size of each sub-task
    ave, res = divmod(data.shape[0], nprocs)
    counts = [ave + 1 if p < res else ave for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    # converts data into a list of arrays
    data = [data[starts[p]:ends[p]] for p in range(nprocs)]
else:
    data = None

data = comm.scatter(data, root=0)

processed_data = process_data(data)

gathered_data = comm.gather(processed_data, root=0)

if rank == 0:
    print("Gathered data:")
    for item in gathered_data:
        print(item)
    print("Concatenated data:")
    concat_data = np.concatenate(gathered_data)
    print(concat_data)

print(f"rank: {rank}, gathered data: {gathered_data}")
