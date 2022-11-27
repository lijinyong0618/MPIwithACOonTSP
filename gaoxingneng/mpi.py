import mpi4py.MPI as MPI
import numpy as np

# print("111")
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
send_obj = {'a': [1, 2.4, 'abc', -2.3+3.4J],
            'b': {2, 3, 4}}
if comm_rank == 0:
    for i in range(1,10):
        send_str = comm_rank,i
        comm.send(send_str, dest=i,tag = i)
        recv_obj = comm.recv(source=i,tag = 0)
        # print(comm_rank,'get',recv_obj)
    print("!11")
# elif comm_rank == 1:
#     send_str = comm_rank
#     recv_obj = comm.recv(source=0)
#     print(comm_rank,'get',recv_obj)
#     comm.send(send_str, dest=0)
else:
    recv_obj = comm.recv(source=0,tag = comm_rank)
    print(comm_rank,'get',recv_obj)
    send_str = comm_rank
    comm.send(send_str, dest=0,tag = 0)
    print(comm_rank,"send down")


