import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import numpy as np
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from .actionPredictor import (
    ActionPredictor,
)
import datetime
import random
import os
# Assuming Conv1dBlock, Rearrange, SinusoidalPosEmb, Downsample1d, Upsample1d are predefined


class CrossAttention(nn.Module):
    def __init__(self, observation_dim, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=False)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.linear = nn.Linear(observation_dim, embed_dim)
    def forward(self, x, context):
        context = context.unsqueeze(2)
        x = einops.rearrange(x, 'b t c -> c b t')
        context = einops.rearrange(context, 'b s c -> c b s')
        context = self.linear(context)
        attn_output, _ = self.multihead_attn(x, context, context)
        x = x + attn_output
        x = self.layer_norm(x)
        x = x + self.ffn(x)
        x = einops.rearrange(x, 'c b t -> b t c')
        return x


def visualize_features(cross_features, sample_rate=40):
    if random.random() > 0.1:
        return

    # Scientific style parameters
    plt_style = {
        'figure': {
            'figsize': (15, 10),
            'dpi': 300,
            'facecolor': 'white',
        },
        'subplot': {
            'wspace': 0.25,
            'hspace': 0.3,
        },
        'font': {
            'family': 'Arial',
            'title_size': 16,
            'label_size': 12,
            'tick_size': 10,
        },
        'heatmap': {
            'cmap': 'GnBu',
            'center': 0,
            'robust': True,
            'square': True,
            'cbar_width': 0.015,
            'cbar_pad': 0.05,
            'vmin': -2,
            'vmax': 2,
        },
        'grid': {
            'color': '#E0E0E0',
            'linestyle': '-',
            'linewidth': 0.5,
            'alpha': 0.5
        }
    }

    # Apply custom style settings directly
    plt.rcParams.update({
        'font.family': plt_style['font']['family'],
        'font.size': plt_style['font']['label_size'],
        'axes.titlesize': plt_style['font']['title_size'],
        'axes.labelsize': plt_style['font']['label_size'],
        'xtick.labelsize': plt_style['font']['tick_size'],
        'ytick.labelsize': plt_style['font']['tick_size'],
        'axes.grid': True,
        'grid.color': plt_style['grid']['color'],
        'grid.linestyle': plt_style['grid']['linestyle'],
        'grid.linewidth': plt_style['grid']['linewidth'],
        'grid.alpha': plt_style['grid']['alpha'],
        'figure.facecolor': plt_style['figure']['facecolor'],
        'axes.facecolor': plt_style['figure']['facecolor'],
    })

    output_dir = 'feature_maps'
    os.makedirs(output_dir, exist_ok=True)

    # Data preprocessing
    features = cross_features.detach().cpu().numpy(
    ) if torch.is_tensor(cross_features) else cross_features

    # Downsampling (添加保护)
    if sample_rate > 1:
        features_downsampled = [
            feat[::sample_rate, ::sample_rate] for feat in features]
        features = np.array(features_downsampled)

    # Standardization
    # features = np.array(
        # [(feat - np.mean(feat)) / (np.std(feat) + 1e-6) for feat in features])

    # Create figure
    fig, axes = plt.subplots(6, 2,
                             figsize=plt_style['figure']['figsize'],
                             dpi=plt_style['figure']['dpi'])
    plt.subplots_adjust(wspace=plt_style['subplot']['wspace'],
                        hspace=plt_style['subplot']['hspace'])

    # Plot heatmaps
    for i in range(12):
        row, col = i // 2, i % 2

        sns.heatmap(features[i],
                    ax=axes[row, col],
                    cmap=plt_style['heatmap']['cmap'],
                    center=plt_style['heatmap']['center'],
                    robust=plt_style['heatmap']['robust'],
                    xticklabels=False,
                    yticklabels=False,
                    square=plt_style['heatmap']['square'],
                    cbar=False,
                    vmin=plt_style['heatmap']['vmin'],
                    vmax=plt_style['heatmap']['vmax'])

        axes[row, col].set_title(f'Feature Map {i+1}',
                                 fontsize=plt_style['font']['title_size'],
                                 pad=10,
                                 fontfamily=plt_style['font']['family'])

    # Add colorbar with fixed range
    norm = plt.Normalize(vmin=plt_style['heatmap']['vmin'],
                         vmax=plt_style['heatmap']['vmax'])
    sm = plt.cm.ScalarMappable(cmap=plt_style['heatmap']['cmap'], norm=norm)
    cbar_ax = fig.add_axes([
        0.92 + plt_style['heatmap']['cbar_pad'],
        0.15,
        plt_style['heatmap']['cbar_width'],
        0.7
    ])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=plt_style['font']['tick_size'])

    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'feature_maps_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath,
                bbox_inches='tight',
                facecolor=plt_style['figure']['facecolor'])
    plt.close()


class ResidualTemporalBlock(nn.Module):

    def __init__(self, observation_dim, inp_channels, out_channels, embed_dim, kernel_size=3, num_heads=4):
        super().__init__()

        self.out_channels = out_channels

        # Define a sequence of convolutional blocks
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size, if_zero=True)
        ])

        # Time embedding block
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # Residual connection
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

        # Cross attention block
        self.cross_attention = CrossAttention(
            observation_dim, out_channels, num_heads)

    def forward(self, x, t, context=None):
        out = self.blocks[0](x) + self.time_mlp(t)
        if context is not None:
            out2 = self.cross_attention(out, context)
            out = out2 + out
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        args,
        dim=256,
        dim_mults=(1, 2, 4),
        num_heads=4
    ):
        super().__init__()

        self.args = args

        self.if_context = args.if_context

        transition_dim = args.action_dim + args.observation_dim + args.class_dim

        # Determine the dimensions at each stage
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim

        # Define the time embedding module (for diffusion models)
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        self.block_num = 0

        # Create downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(args.observation_dim,
                                      dim_in, dim_out, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(args.observation_dim,
                                      dim_out, dim_out, embed_dim=time_dim, num_heads=num_heads),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

            self.block_num += 2

        mid_dim = dims[-1]

        # Define the middle blocks
        self.mid_block1 = ResidualTemporalBlock(args.observation_dim,
                                                mid_dim, mid_dim, embed_dim=time_dim, num_heads=num_heads)
        self.mid_block2 = ResidualTemporalBlock(args.observation_dim,
                                                mid_dim, mid_dim, embed_dim=time_dim, num_heads=num_heads)

        self.block_num += 2

        # Create upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(args.observation_dim,
                                      dim_out * 2, dim_in, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(args.observation_dim,
                                      dim_in, dim_in, embed_dim=time_dim, num_heads=num_heads),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            self.block_num += 2

        # Final convolutional block
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

        # self.block_num = 3
        self.ActionPredictor = ActionPredictor(self.args,
                                               self.args.observation_dim,
                                               self.args.observation_dim,
                                               dim,
                                               self.block_num
                                               )

    # x shape (batch_size,horizon,dimension)

    def forward(self, x, time, observation_img=None, if_visualize=False):

        # print(x.shape)torch.Size([256, 3, 1659])
        # print(time.shape)torch.Size([256])
        #  args.action_dim + args.observation_dim + args.class_dim

        # shape (num_frames, batch_size, observation_dim)
        cross_features = self.ActionPredictor(x[:, 0, self.args.class_dim + self.args.action_dim:],
                                              x[:, -1, self.args.class_dim + self.args.action_dim:])

        # # print(cross_features.shape) torch.Size([256, 12, 1536])
        cross_features = cross_features.permute(1, 0, 2)
        # cross_features = cross_features.repeat_interleave(4, dim=0)

        # if if_visualize:
        #     visualize_features(cross_features)

        # print("cross_features shape")
        # print(cross_features.shape)  # torch.Size([12, 256, 1536])
        # observation_img = observation_img.permute(1, 0, 2)  # [L, B, D]
        # seq_len = observation_img.size(0)

        # print('seq_len', seq_len)

        # if seq_len == 7:
        #     # First frame repeated once
        #     first_frame = observation_img[0:1]  # 1 frame
        #     # Middle frames repeated twice -> 10 frames
        #     middle_frames = observation_img[1:-1].repeat_interleave(2, dim=0)
        #     # Last frame repeated once
        #     last_frame = observation_img[-1:]  # 1 frame
        #     cross_features = torch.cat(
        #         [first_frame, middle_frames, last_frame], dim=0)
        # elif seq_len == 4:
        #     # Repeat each frame 3 times -> 12 frames
        #     cross_features = observation_img.repeat_interleave(3, dim=0)
        # elif seq_len == 6:
        #     # Repeat each frame 2 times -> 12 frames
        #     cross_features = observation_img.repeat_interleave(2, dim=0)
        # elif seq_len == 5:
        #     # Repeat first and last frames 3 times, middle frames 2 times -> 12 frames
        #     first_frame = observation_img[0:1].repeat(3, 1, 1)  # 3 frames
        #     # 6 frames
        #     middle_frames = observation_img[1:-1].repeat_interleave(2, dim=0)
        #     last_frame = observation_img[-1:].repeat(3, 1, 1)  # 3 frames
        #     cross_features = torch.cat(
        #         [first_frame, middle_frames, last_frame], dim=0)
        # else:
        #     raise ValueError(f"Unexpected sequence length: {seq_len}")

        # x shape (batch_size,horizon,dimension)
        # Rearrange input tensor dimensions
        x = einops.rearrange(x, 'b h t -> b t h')

        # Get the time embedding (for diffusion models)
        # Uncomment the following line for Noise and Deterministic Baselines
        # t = None

        # t = torch.randint(0, self.n_timesteps, (batch_size,),
        #    device=x.device).long()  # Random timestep for diffusion
        t = self.time_mlp(time)   # for diffusion

        # print(t.shape)torch.Size([256, 256])
        h = []
        index = 0

        if self.if_context == 0:

            # Forward pass through downsampling blocks
            for resnet, resnet2, downsample in self.downs:
                x = resnet(x, t, cross_features[index])
                index += 1
                x = resnet2(x, t, cross_features[index])
                index += 1
                h.append(x)
                x = downsample(x)

            # Forward pass through middle blocks
            x = self.mid_block1(x, t, cross_features[index])
            index += 1
            x = self.mid_block2(x, t, cross_features[index])
            index += 1

            # Forward pass through upsampling blocks
            for resnet, resnet2, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, t, cross_features[index])
                index += 1
                x = resnet2(x, t, cross_features[index])
                index += 1
                x = upsample(x)

        elif self.if_context == 1:
            # Forward pass through downsampling blocks
            for resnet, resnet2, downsample in self.downs:
                x = resnet(x, t)
                index += 1
                x = resnet2(x, t)
                index += 1
                h.append(x)
                x = downsample(x)

            # Forward pass through middle blocks
            x = self.mid_block1(x, t, cross_features[index])
            index += 1
            x = self.mid_block2(x, t, cross_features[index])
            index += 1

            # Forward pass through upsampling blocks
            for resnet, resnet2, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, t, cross_features[index])
                index += 1
                x = resnet2(x, t, cross_features[index])
                index += 1
                x = upsample(x)

        elif self.if_context == 2:
            # Forward pass through downsampling blocks
            for resnet, resnet2, downsample in self.downs:
                x = resnet(x, t)
                index += 1
                x = resnet2(x, t)
                index += 1
                h.append(x)
                x = downsample(x)

            # Forward pass through middle blocks
            x = self.mid_block1(x, t)
            index += 1
            x = self.mid_block2(x, t)
            index += 1

            # Forward pass through upsampling blocks
            for resnet, resnet2, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, t, cross_features[index])
                index += 1
                x = resnet2(x, t, cross_features[index])
                index += 1
                x = upsample(x)

        elif self.if_context == 3:
            # Forward pass through downsampling blocks
            for resnet, resnet2, downsample in self.downs:
                x = resnet(x, t)
                index += 1
                x = resnet2(x, t)
                index += 1
                h.append(x)
                x = downsample(x)

            # Forward pass through middle blocks
            x = self.mid_block1(x, t)
            index += 1
            x = self.mid_block2(x, t)
            index += 1

            # Forward pass through upsampling blocks
            for resnet, resnet2, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, t)
                index += 1
                x = resnet2(x, t)
                index += 1
                x = upsample(x)
        # Final convolution and rearrange dimensions back

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
        # return x


# shape of x

# torch.Size([12, 256, 1536])

# start-------------------------------
# torch.Size([256, 256, 3])
# torch.Size([256, 256, 3])
# up--------------
# torch.Size([256, 512, 2])
# torch.Size([256, 512, 2])
# up--------------
# torch.Size([256, 1024, 1])
# torch.Size([256, 1024, 1])
# up--------------
# middle-------------
# torch.Size([256, 1024, 1])
# torch.Size([256, 1024, 1])
# middle-------------
# torch.Size([256, 512, 1])
# torch.Size([256, 512, 1])
# down--------------
# torch.Size([256, 256, 2])
# torch.Size([256, 256, 2])
# down--------------
# final-----------------------
# torch.Size([256, 256, 3])
# torch.Size([256, 1659, 3])
# torch.Size([256, 3, 1659])
