# Code for: Ordering Loss
# Author: Nithin Babu
# Created Date: Unknown
# Last Modified Date: 16 April 2024
# Last Modified Author: Shika
# Modifications : Added the feature blended ordering loss, the dry run for logic of the loss, loss dummy test function


import torch
import torchvision.transforms as transforms
from torch import nn
from random import randint
import torch.nn.functional as F


class CustomBlendedOrderingLoss(nn.Module):
    def __init__(self, alpha=0.1, loss_type='square_max', append='10', adaptive_alpha='none', dist_avg_scaling=1.0, args=None):
        super(CustomBlendedOrderingLoss, self).__init__()
        self.alpha = alpha
        self.args = args
        self.loss_type = loss_type
        self.append = append
        self.adaptive_alpha = adaptive_alpha
        self.dist_avg_scaling = dist_avg_scaling
        if loss_type == 'square_max' or loss_type == 'square_softplus':
            self.distance_metric = torch.square
        elif loss_type == 'abs_max_square':
            self.distance_metric = torch.abs

    def forward(self, x):  # x => B x 5 or d x 1

        if self.append == '10':
            x = torch.cat((torch.ones(x.size(0), 1, device=x.device), x, torch.zeros(
                x.size(0), 1, device=x.device)), dim=-1)   # x => B x (D+2)
        elif self.append == '0':
            # x => B x (D+1)
            x = torch.cat(
                (x, torch.zeros(x.size(0), 1, device=x.device)), dim=-1)

        d_p = self.distance_metric(x[:, :-1] - x[:, 1:]) # positives are immediate next elemnent so getting that difference
        d_n = self.distance_metric(x[:, :-2] - x[:, 2:]) # negatives are 2 elements apart so getting that difference

        loss_terms = torch.maximum(torch.cat((d_p[:, :-1] - d_n + 0.5*d_p[:, :-1] + 0.5*d_n,
                                   d_p[:, 1:] - d_n + 0.5*d_p[:, 1:] + 0.5*d_n), dim=0), torch.tensor(0, device=x.device))
        # Look at bottom of file for dry run and explanantion of vectorization

        return torch.mean(loss_terms)
        # Might have to init abs max square check nthins paper

# Same as above but will work with features
class FeatureBlendedOrderingLoss(nn.Module):
    def __init__(self, alpha= 0.1, adaptive_alpha= "none"):
        super(FeatureBlendedOrderingLoss, self).__init__()
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def forward(self, x):  # x => B x 5 x 4096

        d_p = torch.sqrt(torch.sum((x[:, :-1, :] - x[:, 1:, :]) ** 2, dim=-1))  # positives are immediate next element so getting that difference
        d_n = torch.sqrt(torch.sum((x[:, :-2, :] - x[:, 2:, :]) ** 2, dim=-1))  # negatives are 2 elements apart so getting that difference
        
        if self.adaptive_alpha == 'none':
            loss_terms = torch.maximum(torch.cat((d_p[:, :-1] - d_n + self.alpha, d_p[:, 1:] - d_n + self.alpha), dim=0), torch.tensor(0, device=x.device))
        elif self.adaptive_alpha == 'distavg':
            loss_terms = torch.maximum(torch.cat((d_p[:, :-1] - d_n + 0.5*d_p[:, :-1] + 0.5*d_n , 
                                                  d_p[:, 1:] - d_n + 0.5*d_p[:, 1:] + 0.5*d_n), dim=0), torch.tensor(0, device=x.device))
        
        return torch.mean(loss_terms)

# Test function for above losses
def test_custom_blended_ordering_loss():
    # loss = CustomBlendedOrderingLoss()
    # x = torch.randn(4, 3)

    loss = FeatureBlendedOrderingLoss()
    x = torch.randn(4, 5, 4096)

    output = loss(x)
    print(output)

if __name__ == "__main__":
    test_custom_blended_ordering_loss()


##### Dry Run for CustomBlendedOrderingLoss vectorization #####
# Aim: d_p has to be less than d_n. Then the loss is 0. If d_p is greater than d_n, then the loss is that value.
    
# x = [3, 5, 6, 7, 9]
# d_p = [2, 1, 1, 2] for anchors= [(3,5), (5,6), (6,7), (7,9)]
# d_n = [3, 2, 3] for anchors= [(3,6), (5,7), (6,9)]
# Now d_p - d_n the 2 terms that are concatenated are:
# [2-3, 1-2, 1-3],  [1-3, 1-2, 2-3] 
# This is because for: (look at above anchors list for d_p and d_n)
# anchor 3, 5 is +ve and 6 is -ve
# anchor 5, 6 is +ve and 7 is -ve. 
# anchor 6, 7 is +ve and 9 is -ve.
# anchor 6, 5 is +ve and 3 is -ve.
# anchor 7, 6 is +ve and 5 is -ve.
# anchor 9, 7 is +ve and 6 is -ve.
    
# Thus we get 2 loss terms for each element in the batch.
# Concatenate, take max with 0 and then take mean of all the terms.