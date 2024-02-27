import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.cluster import dbscan


class DBSCANFunction(Function):
    @staticmethod
    def forward(ctx, data):
        '''
        ## Parameters:
            `data: torch.Tensor`: Tensor of shape (num_samples, num_features)
            
            `k: int`: The number of clusters
            
        ## Returns:
            `torch.Tensor`: The centroids from the datapoints
        '''
        np_data = data.clone().detach().cpu().numpy()
        cores, indices = dbscan(np_data)
        cores = data[cores]
        indices = torch.from_numpy(indices).to(data.device)
        
        ctx.save_for_backward(data, indices)
        return cores, indices
    
    @staticmethod
    def backward(ctx, centroids_grad_output, indices_grad_output):
        data, indices = ctx.saved_tensors
        
        grad_data = torch.zeros_like(data)
        grad_data[indices] = centroids_grad_output
        
        return grad_data, None

class DBSCAN(nn.Module):
    def __init__(self):
        super(DBSCAN, self).__init__()
        
    def forward(self, data, return_indices: bool=False):
        cores, indices = DBSCANFunction.apply(data
        if return_indices:
            return cores, indices
        else:
            return cores


if __name__ == '__main__':
    # Create synthetic data
    num_samples = 100
    num_features = 2
    data = torch.rand(num_samples, num_features, requires_grad=True)

    # Create the DifferentiableKMeansPP module
    num_clusters = 3
    kmeans_module = DBSCAN()

    for _ in range(10):
        # Apply the module to get initial centroids
        cores = kmeans_module(data)
        print(cores.shape)

        # Backward test
        loss = torch.cdist(cores, cores).mean()
        loss.backward()

        with torch.no_grad():
            data += data.grad * 0.3
            data.grad.zero_()
        print(loss.item())
