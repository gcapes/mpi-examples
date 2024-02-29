# Point to point communication example
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:	
	num_data = 10
	comm.send(num_data, dest=1)

	data = np.linspace(0.0, 3.14, num_data)
	# Capital S needed for non-generic python objects
	comm.Send(data, dest=1)
elif rank == 1:
	num_data = comm.recv(source=0)
	print('Number of data to receieve: ', num_data)

	data = np.empty(num_data, dtype='d')
	comm.Recv(data, source=0)

	print('data received: ', data)
