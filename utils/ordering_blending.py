# ordering loss
# Nithin's sample code

class PentetContrastV1(nn.Module):

    def __init__(self, alpha=0.1, loss_type='square_max', append='10', adaptive_alpha='none', dist_avg_scaling=1.0, args=None):
        super(PentetContrastV1, self).__init__()
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

    def forward(self, x):  # x => B x D

        if self.append == '10':
            x = torch.cat((torch.ones(x.size(0), 1, device=x.device), x, torch.zeros(x.size(0), 1, device=x.device)), dim=-1)   # x => B x (D+2)
        elif self.append == '0':
            x = torch.cat((x, torch.zeros(x.size(0), 1, device=x.device)), dim=-1)   # x => B x (D+1)

        d_p = self.distance_metric(x[:, :-1] - x[:, 1:])
        d_n = self.distance_metric(x[:, :-2] - x[:, 2:])

        if self.loss_type == 'square_softplus':
            if self.adaptive_alpha == 'none':
                loss_terms = F.softplus(torch.cat((d_p[:, :-1] - d_n + self.alpha, d_p[:, 1:] - d_n + self.alpha), dim=0))
            elif self.adaptive_alpha == 'distAvg':
                # loss_terms = F.softplus(torch.cat((d_p[:, :-1] - d_n + self.dist_avg_scaling*(0.5*d_p[:, :-1] + 0.5*d_n), d_p[:, 1:] - d_n + self.dist_avg_scaling*(0.5*d_p[:, 1:] + 0.5*d_n)), dim=0))
                loss_terms = F.softplus(torch.cat((d_p[:, :-1] - d_n + 0.5*d_p[:, :-1] + 0.5*d_n, d_p[:, 1:] - d_n + 0.5*d_p[:, 1:] + 0.5*d_n), dim=0))
        else:
            if self.adaptive_alpha == 'none':
                loss_terms = torch.maximum(torch.cat((d_p[:, :-1] - d_n + self.alpha, d_p[:, 1:] - d_n + self.alpha), dim=0), torch.tensor(0, device=x.device))
            elif self.adaptive_alpha == 'distAvg':
                loss_terms = torch.maximum(torch.cat((d_p[:, :-1] - d_n + 0.5*d_p[:, :-1] + 0.5*d_n , d_p[:, 1:] - d_n + 0.5*d_p[:, 1:] + 0.5*d_n), dim=0), torch.tensor(0, device=x.device))
        
        if self.loss_type == 'abs_max_square':
            loss_terms = torch.square(loss_terms)

        return torch.mean(loss_terms)


# blending code

with torch.no_grad():
    ## lowQ_patches => bbs x 3 x H x W
    # batch_to_grid(x_h.cpu(), 8,'./datasets/im_test/high_clip.png')
    
    if args.blend_extremes == 'population':
        x_l_batch = x_l[random.sample(range(x_l.size(0)), k=args.blend_batch_size)]
        x_h_batch = x_h[random.sample(range(x_h.size(0)), k=args.blend_batch_size)]
    elif args.blend_extremes == 'batch':
        x_idx = torch.argsort(score_frozen[:, 0])
        low_idx = random.sample(x_idx[:args.blend_batch_size].tolist(), k=args.blend_batch_size)
        high_idx = random.sample(x_idx[-args.blend_batch_size:].tolist(), k=args.blend_batch_size)
        x_l_batch = x1[low_idx, -1]
        x_h_batch = x1[high_idx, 0]
        # batch_to_grid(torch.cat((x_l_batch, x_h_batch), dim=0).cpu(), args.blend_batch_size, './datasets/im_test/batch_extremes.png')
        # exit()
    if args.blend_extremes == 'pop_batch':
        x_l_batch0 = x_l[random.sample(range(x_l.size(0)), k=args.blend_batch_size//2)]
        x_h_batch0 = x_h[random.sample(range(x_h.size(0)), k=args.blend_batch_size//2)]
        x_idx = torch.argsort(score_frozen[:, 0])
        low_idx = random.sample(x_idx[:args.blend_batch_size].tolist(), k=args.blend_batch_size//2)
        high_idx = random.sample(x_idx[-args.blend_batch_size:].tolist(), k=args.blend_batch_size//2)
        x_l_batch = torch.cat((x_l_batch0, x1[low_idx, -1]), dim=0)
        x_h_batch = torch.cat((x_h_batch0, x1[high_idx, 0]), dim=0)

    
    blend_vec = torch.linspace(0, 1, args.blend_levels, device=args.device).view(1, 1, 1, 1, args.blend_levels)
    blends = blend_vec * x_l_batch.unsqueeze(-1) + (1-blend_vec) * x_h_batch.unsqueeze(-1)
    # blends = torch.clip(blend_vec * x_h.unsqueeze(-1) + (1-blend_vec) * x_l.unsqueeze(-1), 0, 1)
    ## blends => bbs x 3 x H x W x bl
    blends = torch.permute(blends, (0, 4, 1, 2, 3))
    ## blends => bbs x bl x 3 x H x W
    # batch_to_grid(blends[:4].contiguous().view(-1, blends.size(-3), blends.size(-2), blends.size(-1)).cpu(), blends.size(1), './datasets/im_test/blend_clip.png')
    # exit()