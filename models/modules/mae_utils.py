import numpy as np
import torch

def index_sequence(x, ids):
    return torch.gather(x, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, x.shape[2]))

def random_masking(x, rng, keep_len, padding_mask=None):
    batch, length, dim = x.shape
    noise = torch.random.uniform(rng, (length,), dtype=torch.float32)
    ids_shuffle = torch.argsort(noise, axis=0)
    ids_restore = torch.argsort(ids_shuffle, axis=0)
    kept = index_sequence(x, ids_shuffle[:keep_len])
    mask = torch.ones([batch, length], dtype=torch.float32)
    mask = mask.at[:, :keep_len].set(0.0)
    mask = index_sequence(mask, ids_restore)

    if padding_mask is None:
        return kept, mask, ids_restore

    padding_mask_kept = index_sequence(padding_mask, ids_shuffle[:keep_len])
    return kept, mask, ids_restore, padding_mask_kept

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
    
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(np.expand_dims(pos_embed, 0))

def get_1d_sincos_pos_embed(embed_dim, grid_size):
    return torch.from_numpy(np.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim,np.arange(grid_size, dtype=np.float32)),
        0
    ))

def cross_entropy_loss_and_accuracy(logits, tokens, val=None):
    if val is None:
        val = all_mask(tokens)
    val_text_length = torch.maximum(torch.sum(val, dim=-1), torch.Tensor([1e-5]).to(val.device))

    token_log_prob = torch.squeeze(
        torch.gather(
            torch.nn.functional.log_softmax(logits, dim=-1),
            index=torch.unsqueeze(tokens, -1),
            dim=-1,
        ),
        -1,
    ).to(val.device)
    token_log_prob = torch.where(val > 0.0, token_log_prob, torch.Tensor([0.0]).to(val.device))
    loss = -torch.mean(torch.sum(token_log_prob, dim=-1) / val_text_length)
    correct = torch.where(
        val > 0.0,
        torch.argmax(logits, axis=-1) == tokens,
        torch.tensor([False]).to(val.device)
    )
    accuracy = torch.mean(torch.sum(correct, dim=-1) / val_text_length)
    return loss, accuracy

def all_mask(x):
    return torch.ones(x.shape[:2])

def patch_mse_loss(patch_output, patch_target, val=None):
    if val is None:
        val = all_mask(patch_target)
    val_ratio = torch.sum(val, dim=1) / val.shape[1]
    return torch.mean(
        torch.mean(
            torch.where(
                val.squeeze(-1) > 0.0,
                torch.mean(torch.square(patch_target - patch_output), dim=-1),
                torch.Tensor([0.0]).to(patch_output.device),
            ),
            dim=-1,
        ) / val_ratio
    )

def mask_intersection(mask1, mask2):
    return torch.logical_and(mask1 > 0, mask2 > 0)#.type(torch.float32)

def mask_not(mask):
    return 1.0 - mask

def mask_select(mask, this, other=None):
    if this is None:
        this = torch.array(0, dtype=this.dtype)
    if len(this.shape) == 3:
        mask = torch.unsqueeze(mask, axis=-1)
    return torch.where(mask == 0.0, this, other)

def merge_patches(inputs, patch_size):
    batch, length, _ = inputs.shape
    height = width = int(length**0.5)
    x = torch.reshape(inputs, (batch, height, width, patch_size, patch_size, -1))
    x = torch.swapaxes(x, 2, 3)
    x = torch.reshape(x, (batch, height * patch_size, width * patch_size, -1))
    return x

def select_masked(x, mask, len_masked):
    mask = mask.type(torch.int).to(x.device)
    mask_pos = torch.nonzero(mask, as_tuple=True)[1]    
    mask_pos = mask_pos.reshape(len(mask_pos) // len_masked, -1)
    masked_x = torch.gather(x, dim=1, index=mask_pos.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
    
    return masked_x