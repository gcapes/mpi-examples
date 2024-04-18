from mpi4py import MPI
import numpy as np
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

def process_data(data):
    print(f'Process {rank} has data: {data}')
    data = data + rank
    return data

def split_data_for_scatter(data: list, n_ranks: int) -> list:
    """Split a list into a list of lists for MPI scattering"""
    data_length = len(data)
    quot, rem = divmod(data_length, n_ranks)

    if quot == 0:
        print("Error: Length of data list should be >= number of MPI ranks!")
        sys.stdout.flush()
        comm.Abort()

    # determine the size of each sub-task
    counts = [quot + 1 if n < rem else quot for n in range(n_ranks)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:n]) for n in range(n_ranks)]
    ends = [sum(counts[:n + 1]) for n in range(n_ranks)]

    # converts data into a list of arrays
    scatter_data = [data[starts[n]:ends[n]] for n in range(n_ranks)]
    return scatter_data

if rank == 0:
    data = np.arange(5)
    data = np.tile(data, (10, 1))

    data = split_data_for_scatter(data, nprocs)
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
