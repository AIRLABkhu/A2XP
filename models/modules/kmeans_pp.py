import torch
import torch.nn as nn
from torch.autograd import Function
from sklearn.cluster import kmeans_plusplus


class KMeansPPFunction(Function):
    @staticmethod
    def forward(ctx, data, k):
        '''
        ## Parameters:
            `data: torch.Tensor`: Tensor of shape (num_samples, num_features)
            
            `k: int`: The number of clusters
            
        ## Returns:
            `torch.Tensor`: The centroids from the datapoints
        '''
        np_data = data.clone().detach().cpu().numpy()
        centroids, indices = kmeans_plusplus(np_data, n_clusters=k)
        centroids = torch.from_numpy(centroids).to(data.device)
        indices = torch.from_numpy(indices).to(data.device)
        
        ctx.save_for_backward(data, centroids, indices)
        return centroids, indices
    
    @staticmethod
    def backward(ctx, centroids_grad_output, indices_grad_output):
        data, centroids, indices = ctx.saved_tensors
        
        grad_data = torch.zeros_like(data)
        grad_data[indices] = centroids_grad_output
        
        return grad_data, None
    
def kmeans_pp(data, k):
    return KMeansPPFunction.apply(data, k)

class KMeansPP(nn.Module):
    def __init__(self, k: int):
        super(KMeansPP, self).__init__()
        self.__k = k
        
    def forward(self, data, return_indices: bool=False):
        centroids, indices = kmeans_pp(data, self.__k)
        if return_indices:
            return centroids, indices
        else:
            return centroids
    
    @property
    def k(self):
        return self.__k
    
def find_kmeans_cluster(centroids: torch.Tensor, data: torch.Tensor):
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if data.ndim != 2:
        raise RuntimeError(f'Tensor of (batch_size, dim) or (dim,) is required, '
                           f'but got {tuple(data.shape)}.')
    
    distances = torch.cdist(centroids, data)
    return distances.argmin(dim=0)


if __name__ == '__main__':
    # Create synthetic data
    num_samples = 100
    num_features = 2
    data = torch.rand(num_samples, num_features, requires_grad=True)

    # Create the DifferentiableKMeansPP module
    num_clusters = 3
    kmeans_module = KMeansPP(num_clusters)

    for _ in range(10):
        # Apply the module to get initial centroids
        centroids = kmeans_module(data)

        # Backward test
        loss = torch.cdist(centroids, centroids).mean()
        loss.backward()

        with torch.no_grad():
            data += data.grad * 0.1
            data.grad.zero_()
        print(loss.item())
