import numpy as np
import os,sys,time
import gc

from utils import *
from model import resnet


def empirical_K(model, data, number_samples, device,
        sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1, # For He-normal init (fan_in mode)
        empirical_kernel_batch_size=256,
        truncated_init_dist=False,
        store_partial_kernel=False, # True will not average the kernel at thn end.
        partial_kernel_n_proc=1,
        partial_kernel_index=0):

    # Here model should be the coresponding CNN poped the lase fc layer.

    if device == 'cuda':
        sigmaw = torch.tensor(sigmaw).to(device)
        sigmab = torch.tensor(sigmab).to(device)


    #number_samples = data.shape[0] # Number of MC samples
    num_tasks = number_samples

    if store_partial_kernel:
        size = partial_kernel_n_proc
        rank = partial_kernel_index
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        # print(rank)
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)

    print("Doing process %d of %d" % (rank, size))

    m = len(data)
    if device == 'cuda':
        covs = torch.zeros(m, m).to(device)
    else:
        covs = np.zeros((m,m), dtype=np.float32)
    local_index = 0
    update_chunk = 20000
    num_chunks = covs.shape[0]//update_chunk
    print("num_chunks",num_chunks)

    for index in tasks:
        start_time = time.time()
        print("sample for kernel", index)

        model.apply(he_init)

        if covs.shape[0] > update_chunk: # This whole bit needs debugging
            for i in range(num_chunks):
                covs[i*update_chunk:(i+1)*update_chunk] += (
                        (sigmaw**2/X.shape[1]) * np.matmul(
                            X[i*update_chunk:(i+1) * update_chunk], X.T)
                        + (sigmab**2)*np.ones((update_chunk,X.shape[0]), dtype=np.float32)
                        )
            last_bits = slice(update_chunk*num_chunks,covs.shape[0])
            covs[last_bits] += (
                    (sigmaw**2/X.shape[1]) *
                    np.matmul(X[last_bits],X.T) +
                    (sigmab**2) *
                    np.ones((last_bits.stop-last_bits.start,X.shape[0]), dtype=np.float32)
                    )
        else:
            X = model_predict(model, data,
                    min(empirical_kernel_batch_size, len(data)), 4, device)
            if device == 'cuda':
                if len(X.shape) == 1:
                    X.unsqueeze_(0)
                covs += (sigmaw**2 / X.shape[1]) * torch.matmul(X,X.T) + sigmab**2
            else:
                if len(X.shape) == 1:
                    X = np.expand_dims(X, 0)
                covs += (sigmaw**2 / X.shape[1]) * np.matmul(X,X.T) + sigmab**2

        sys.stdout.flush()
        local_index += 1
        gc.collect()
        print("--- %s seconds ---" % (time.time() - start_time))

    if size > 1 and not store_partial_kernel:
        covs1_recv = None
        covs2_recv = None
        if rank == 0:
            if device == 'cuda':
                covs1_recv = torch.zeros_like(covs[:25000,:])
                covs2_recv = torch.zeros_like(covs[25000:,:])
            else:
                covs1_recv = np.zeros_like(covs[:25000,:])
                covs2_recv = np.zeros_like(covs[25000:,:])
        #print(covs[25000:,:])
        comm.Reduce(covs[:25000,:], covs1_recv, op=MPI.SUM, root=0)
        comm.Reduce(covs[25000:,:], covs2_recv, op=MPI.SUM, root=0)

        if rank == 0:
            if device == 'cuda':
                covs_recv = torch.cat([covs1_recv,covs2_recv],0)
            else:
                covs_recv = np.concatenate([covs1_recv,covs2_recv],0)
            return covs_recv/number_samples
        else:
            return None
    else:
        #if covs.shape[0] > update_chunk:
        #    #make matrix symmetric
        #    covs = np.maximum(covs,covs.trasnpose())
        if store_partial_kernel:
            return covs
        else:
            return covs/number_samples


if __name__ == '__main__':
    model = resnet.ResNet_pop_fc_50(num_classes=1) # Actually num_classes doesn't matter
                                                   # because the fc layer was removed.
    from main import trainset, device
    K = empirical_K(model, trainset, 100, device,
            sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1,
            empirical_kernel_batch_size=256,
            truncated_init_dist=False,
            store_partial_kernel=False,
            partial_kernel_n_proc=1,
            partial_kernel_index=0)
    print(K.shape)

