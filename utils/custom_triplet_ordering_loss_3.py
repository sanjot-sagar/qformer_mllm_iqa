# Code for: Ordering Loss
# Author: Shika
#
# Last Modified Date: 9 May 2024
# Last Modified Author: Shika
# Modifications : Adapted Nithin's loss to take in FSIM matrices and calculate triplet loss along 2 directions, distortion and level


import torch
import torchvision.transforms as transforms
from torch import nn
from random import randint
import torch.nn.functional as F
import math


class LevelBlendedOrderingLoss(nn.Module):
    def __init__(self, alpha=0.1, adaptive_alpha="none"):
        super(LevelBlendedOrderingLoss, self).__init__()
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def forward(self, x, num_dist_types, num_levels):  # x => B x 17 x 4096

        # Triplet loss 1 of loss: For LEVELS working along the 3rd dimension to get B x 4 loss terms
        first_image_feature = x[:, 0, :].unsqueeze(
            1).unsqueeze(1)  # shape: B x 1 x 4096
        remaining_features = x[:, 1:, :]  # shape: B x 16 x 4096
        # shape: B x 4 x 4 x 4096
        reshaped_remaining_features = remaining_features.view(
            -1, num_dist_types, num_levels, 4096)
        # Add the first feature to each of the 4 rows to get a 4x5 feature vector # shape: B x 4 x 5 x 4096 (5 levels for the 4 distortion types)
        level_reshaped_x = torch.cat(
            (first_image_feature.expand(-1, num_dist_types, -1, -1), reshaped_remaining_features), dim=2)

        # positives are immediate next element so getting that difference
        d_p_level = torch.sqrt(torch.sum(
            (level_reshaped_x[:, :, :-1, :] - level_reshaped_x[:, :, 1:, :]) ** 2, dim=-1))
        # negatives are 2 elements apart so getting that difference
        d_n_level = torch.sqrt(torch.sum(
            (level_reshaped_x[:, :, :-2, :] - level_reshaped_x[:, :, 2:, :]) ** 2, dim=-1))

        if self.adaptive_alpha == 'none':
            loss_terms_level = torch.maximum(torch.cat(
                (d_p_level[:, :, :-1] - d_n_level + self.alpha, d_p_level[:, :, 1:] - d_n_level + self.alpha), dim=0), torch.tensor(0, device=x.device))
        elif self.adaptive_alpha == 'distavg':
            loss_terms_level = torch.maximum(torch.cat((d_p_level[:, :, :-1] - d_n_level + 0.5*d_p_level[:, :, :-1] + 0.5*d_n_level,
                                                        d_p_level[:, :, 1:] - d_n_level + 0.5*d_p_level[:, :, 1:] + 0.5*d_n_level), dim=0), torch.tensor(0, device=x.device))

        loss = torch.mean(loss_terms_level)
        return loss


# Opt1: Triplet loss along levels (4 loss terms) and triplet loss along different distortion type (17 loss terms) for each anchor image.
# Triplet loss along the levels for each distortion type like before. (4,5 vector to apply the triplet loss where we end up with 4 loss terms for levels).
# For an anchor img, we'll find closest SSIM value from a different distortion type than anchor to take as positive and farthest SSIM value from different distortion type than anchor to take as negative. (17,3 vector to apply the triplet loss where we end up with 17 loss terms).
class Opt1_DistBlendedOrderingLoss(nn.Module):
    def __init__(self, alpha=0.1, adaptive_alpha="none"):
        super(Opt1_DistBlendedOrderingLoss, self).__init__()
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def forward(self, x, annotator_matrix, num_dist_types, num_levels, anchor_negatives="min"):  # x => B x 17 x 4096

        # Triplet loss 2 of loss: For different DISTORTION TYPES getting B x 17 loss terms
        first_row = annotator_matrix[:, 0, :].unsqueeze(1)  # B x 1 x 17
        first_col = annotator_matrix[:, :, 0].unsqueeze(2)  # B x 17 x 1
        remaining_annotator_matrix = annotator_matrix[:, 1:, 1:]  # B x 16 x 16

        # because we're blocking out same distortion type values
        mask = torch.ones((num_levels, num_levels), device=x.device)
        # [mask] * num_dist_types: This creates a new list that repeats the elements of [mask] num_dist_types times. Then the * operator unpacks the list into individual arguments to block_diag
        masked_annotator_matrix = torch.block_diag(*[mask] * num_dist_types)
        masked_annotator_matrix = 1 - masked_annotator_matrix
        masked_annotator_matrix = masked_annotator_matrix.unsqueeze(
            0).expand(annotator_matrix.size(0), -1, -1)
        masked_annotator_matrix = masked_annotator_matrix * \
            remaining_annotator_matrix  # B x 16 x 16

        # Because we'll add the 0th element from row only
        first_col = first_col[:, 1:, :]
        final_masked_annotation_matrix = torch.cat(
            [first_row, torch.cat([first_col, masked_annotator_matrix], dim=2)], dim=1)
        final_masked_annotation_matrix[final_masked_annotation_matrix == 0] = float(
            'nan')

        # Find the indices of the max values in each row (B x 17)
        positive_indices = nanargmax2(
            final_masked_annotation_matrix, dim=2, second_highest=False)
        if anchor_negatives == "min":
            # Find the indices of the min values in each row (B x 17)
            negative_indices = nanargmin(final_masked_annotation_matrix, dim=2)
        elif anchor_negatives == "second-highest":
            # Find the indices of the second highest values in each row (B x 17)
            negative_indices = nanargmax2(
                final_masked_annotation_matrix, dim=2, second_highest=True)

        positive_features = x[torch.arange(
            positive_indices.size(0)).unsqueeze(1), positive_indices, :]
        negative_features = x[torch.arange(
            negative_indices.size(0)).unsqueeze(1), negative_indices, :]

        d_p_dist = torch.sqrt(
            torch.sum((x - positive_features) ** 2, dim=-1))  # B x 17
        d_n_dist = torch.sqrt(
            torch.sum((x - negative_features) ** 2, dim=-1))  # B x 17

        if self.adaptive_alpha == 'none':
            loss_terms_dist = torch.maximum(
                d_p_dist - d_n_dist + self.alpha, torch.tensor(0, device=x.device))
        elif self.adaptive_alpha == 'distavg':
            loss_terms_dist = torch.maximum(
                d_p_dist - d_n_dist + 0.5*d_p_dist + 0.5*d_n_dist, torch.tensor(0, device=x.device))

        loss = torch.mean(loss_terms_dist)
        return loss


# Opt2: Triplet loss along levels (4 loss terms) and triplet loss along first level of each distortion type (1 loss term).
# Triplet loss along the levels for each distortion type like before. (4,5 vector to apply the triplet loss where we end up with 4 loss terms for levels).
# Triplet loss along the first level of each distortion type. (1,5 vector to apply the triplet loss where we end up with 1 loss term).
# PROBLEM WITH THIS: For each distortion type, we're only considering the first level. This might not be the best way to do it.
class Opt2_DistBlendedOrderingLoss(nn.Module):
    def __init__(self, alpha=0.1, adaptive_alpha="none"):
        super(Opt2_DistBlendedOrderingLoss, self).__init__()
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha

    def forward(self, x, annotator_matrix, num_dist_types, num_levels):

        first_image_feature = x[:, 0, :].unsqueeze(
            1).unsqueeze(1)  # shape: B x 1 x 4096
        remaining_features = x[:, 1:, :]  # shape: B x 16 x 4096
        # shape: B x 4 x 4 x 4096
        reshaped_remaining_features = remaining_features.view(
            -1, num_dist_types, num_levels, 4096)

        # Triplet loss 2 of loss: For DISTORTION TYPES working along the 2nd dimension in B x 5 x 4096 to get B x 1 loss term
        dist_reshaped_x = reshaped_remaining_features[:, :, 0, :]

        # B x 16 row, column doesn't matter SSIM is symmetric
        annotator_matrix = annotator_matrix[:, 0, :]
        # Get only first level of each distortion type and not considering SSIM(ref, ref)
        annotator_matrix = annotator_matrix[:, 1::num_levels]
        _, indices = torch.sort(
            annotator_matrix, descending=True, dim=1)  # B x 4

        sorted_dist_x = torch.gather(dist_reshaped_x, 1, indices.unsqueeze(
            -1).expand(-1, -1, dist_reshaped_x.size(-1)))  # B x 4 x 4096
        first_image_feature.squeeze_(1)
        sorted_dist_x = torch.cat(
            (first_image_feature, sorted_dist_x), dim=1)  # shape: B x 5 x 4096

        # positives are immediate next element so getting that difference
        d_p_dist = torch.sqrt(
            torch.sum((sorted_dist_x[:, :-1, :] - sorted_dist_x[:, 1:, :]) ** 2, dim=-1))
        # negatives are 2 elements apart so getting that difference
        d_n_dist = torch.sqrt(
            torch.sum((sorted_dist_x[:, :-2, :] - sorted_dist_x[:, 2:, :]) ** 2, dim=-1))

        if self.adaptive_alpha == 'none':
            loss_terms_dist = torch.maximum(torch.cat(
                (d_p_dist[:, :-1] - d_n_dist + self.alpha, d_p_dist[:, 1:] - d_n_dist + self.alpha), dim=0), torch.tensor(0, device=x.device))
        elif self.adaptive_alpha == 'distavg':
            loss_terms_dist = torch.maximum(torch.cat((d_p_dist[:, :-1] - d_n_dist + 0.5*d_p_dist[:, :-1] + 0.5*d_n_dist,
                                                       d_p_dist[:, 1:] - d_n_dist + 0.5*d_p_dist[:, 1:] + 0.5*d_n_dist), dim=0), torch.tensor(0, device=x.device))

        loss = torch.mean(loss_terms_dist)
        return loss


# Util Helper Functions
def nanargmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).argmin(dim=dim, keepdim=keepdim)
    return output

# def nanargmax(tensor, dim=None, keepdim=False):
#     min_value = torch.finfo(tensor.dtype).min
#     output = tensor.nan_to_num(min_value).argmax(dim=dim, keepdim=keepdim)
#     return output


def nanargmax2(tensor, dim=None, second_highest=True):
    min_value = torch.finfo(tensor.dtype).min

    # Replace nan values with the minimum value, get the indices of the two highest values
    tensor_nan_to_num = tensor.nan_to_num(min_value)
    _, indices = torch.topk(tensor_nan_to_num, 2,
                            dim=dim, largest=True, sorted=True)

    if second_highest:
        output = indices[..., 1]
    else:
        output = indices[..., 0]
    return output

# Test function for above losses


def test_custom_blended_ordering_loss():
    loss_level = LevelBlendedOrderingLoss()
    loss_dist = Opt1_DistBlendedOrderingLoss()
    # loss_dist = Opt2_DistBlendedOrderingLoss()

    x = torch.randn(4, 5, 4096)  # B x 17 x 4096
    annotator_matrix = torch.randn(4, 5, 5)  # B x 17 x 17
    # Note: I want the diagonals where SSIM=1 to be 0 instead.
    num_dist_types = 2
    num_levels = 2

    output_level = loss_level(x, num_dist_types, num_levels)
    output_dist = loss_dist(x, annotator_matrix, num_dist_types, num_levels)
    print(output_level)
    print(output_dist)


if __name__ == "__main__":
    test_custom_blended_ordering_loss()


# For block matrix creation
# a = torch.ones((4, 4))
# for i in range(2):
#     b = torch.zeroes((2, 2))
#     a[i*2:(i+1)*2, i*2:(i+1)*2] = b
