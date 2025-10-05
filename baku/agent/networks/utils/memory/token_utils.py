import torch
import torch.nn.functional as F

def gumbel_noise(shape, device):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + 1e-9) + 1e-9)

def gumbel_soft_topk(scores, k, temperature=0.5, hard=False):
    """
    Gumbel-Softmax Top-k (可微近似)
    返回 soft/hard 近似 mask
    """
    B, N = scores.shape
    device = scores.device

    gumbel_scores = scores + gumbel_noise((B, N), device)
    probs = F.softmax(gumbel_scores / temperature, dim=-1)  # [B, N]

    # 取前 k
    topk_probs, topk_idx = torch.topk(probs, k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(1, topk_idx, topk_probs)

    if hard:
        hard_mask = torch.zeros_like(probs)
        hard_mask.scatter_(1, topk_idx, 1.0)
        mask = hard_mask.detach() - mask.detach() + mask

    return mask  # [B, N]

class ActionGuidedTokenSelector:
    def __init__(self, temperature: float = 0.5, hard: bool = True):
        """
        Action-guided Token Selection (仅保留 top-k tokens)
        参数
        ----
        temperature : Gumbel-Softmax 温度
        hard : True = 前向硬选择，反向可微 (straight-through)
        """
        self.temperature = temperature
        self.hard = hard

    def forward(self, X, K, action, keep_ratio=0.5):
        """
        参数
        ----
        X : [B, N, D] tokens
        K : [B, N, D] keys
        action : [B, D] action token
        keep_ratio : float, 保留比例 (e.g. 0.5 表示 top-50%)

        返回
        ----
        X_kept : [B, N', D] 选择后的 tokens
        K_kept : [B, N', D] 选择后的 keys
        mask   : [B, N] soft/hard mask
        """
        B, N, D = X.shape
        k_keep = int(N * keep_ratio)

        # Step1: cross-attention scores
        q_action = F.normalize(action, dim=-1).unsqueeze(1)  # [B,1,D]
        k_norm = F.normalize(K, dim=-1)                     # [B,N,D]
        scores = torch.matmul(q_action, k_norm.transpose(-1, -2)).squeeze(1)  # [B,N]

        # Step2: soft/hard top-k mask
        mask = gumbel_soft_topk(scores, k_keep, self.temperature, self.hard)  # [B,N]

        # Step3: 应用 mask (只保留 topk tokens)
        kept_tokens = []
        kept_keys = []
        for b in range(B):
            idx = (mask[b] > 0).nonzero(as_tuple=True)[0]  # 被选中的 indices
            kept_tokens.append(X[b, idx])
            kept_keys.append(K[b, idx])

        X_kept = torch.nn.utils.rnn.pad_sequence(kept_tokens, batch_first=True)
        K_kept = torch.nn.utils.rnn.pad_sequence(kept_keys, batch_first=True)

        return X_kept, K_kept, mask
    

class CenterAwareTokenMerging:
    def __init__(self, H: int, W: int, k: int):
        """
        中心保留更多信息的 Token Merging 策略
        参数
        ----
        H, W : 图片 patch 网格大小
        k : 全局 partition factor 基准
        """
        self.H, self.W = H, W
        self.k_center = max(1, k // 2)     # 中心区更密集
        self.k_edge = max(1, int(1.5 * k)) # 边缘更稀疏

    def _partition_indices(self, h, w, k, device):
        """
        对某个区域 [h, w] 展平成序列并划分 target/source
        """
        N = h * w
        idx = torch.arange(N, device=device)
        is_target = (idx % k) == 0
        return idx[is_target], idx[~is_target]

    def forward(self, X, K):
        """
        参数
        ----
        X : [B, N, D]
        K : [B, N, D]

        返回
        ----
        Y : [B, T, D]
        """
        B, N, D = X.shape
        H, W = self.H, self.W
        assert N == H * W, "N must equal H*W"

        # reshape 成 [B, H, W, D]
        X_2d = X.view(B, H, W, D)
        K_2d = K.view(B, H, W, D)

        # 定义中心区
        h0, h1 = H // 4, 3 * H // 4
        w0, w1 = W // 4, 3 * W // 4

        # 中心 tokens
        X_center = X_2d[:, h0:h1, w0:w1, :].reshape(B, -1, D)
        K_center = K_2d[:, h0:h1, w0:w1, :].reshape(B, -1, D)

        # 边缘 tokens
        mask = torch.ones(H, W, dtype=torch.bool, device=X.device)
        mask[h0:h1, w0:w1] = False
        X_edge = X_2d[:, mask, :].reshape(B, -1, D)
        K_edge = K_2d[:, mask, :].reshape(B, -1, D)

        # --- 对中心做 token merging ---
        Y_center = self._merge_region(X_center, K_center, self.k_center)

        # --- 对边缘做 token merging ---
        Y_edge = self._merge_region(X_edge, K_edge, s000elf.k_edge)

        # 拼接结果
        Y = torch.cat([Y_center, Y_edge], dim=1)
        return Y

    def _merge_region(self, X, K, k):
        """
        在一个区域上执行普通的 token merging
        X, K : [B, N, D]
        """
        B, N, D = X.shape
        device = X.device

        # step1: partition
        idx = torch.arange(N, device=device)
        tgt_idx = idx[(idx % k) == 0]
        src_idx = idx[(idx % k) != 0]

        # step2: match sources -> targets
        K_tgt = F.normalize(K[:, tgt_idx], dim=-1)  # [B, T, D]
        K_src = F.normalize(K[:, src_idx], dim=-1)  # [B, S, D]
        sims = torch.matmul(K_src, K_tgt.transpose(-1, -2))  # [B, S, T]
        best_idx = sims.argmax(dim=-1)  # [B, S]
        matched_tgt_idx = tgt_idx[best_idx]  # [B, S]

        # step3: average pooling
        T = tgt_idx.shape[0]
        Y = torch.zeros(B, T, D, device=device)
        counts = torch.zeros(B, T, 1, device=device)

        for b in range(B):
            # target 自己
            Y[b].scatter_add_(0,
                              torch.arange(T, device=device).unsqueeze(-1).expand(-1, D),
                              X[b, tgt_idx])
            counts[b] += 1

            # source
            for s, t_global in zip(src_idx, matched_tgt_idx[b]):
                t_local = (tgt_idx == t_global).nonzero(as_tuple=True)[0].item()
                Y[b, t_local] += X[b, s]
                counts[b, t_local] += 1

        return Y / counts


class UniformTokenMerging:
    def __init__(self, k: int, one_based_indexing: bool = False):
        """
        Token Merging (ToMe) for [B, N, D] inputs

        参数
        ----
        k : 分组大小 (stride)
        one_based_indexing : 若为 True，则按 (i+1) % k == 0 作为 target
                             否则 (默认) i % k == 0 作为 target
        """
        self.k = k
        self.one_based_indexing = one_based_indexing

    def _partition_by_mod(self, N: int, device: torch.device):
        idx = torch.arange(N, device=device)

        if self.one_based_indexing:
            is_target = ((idx + 1) % self.k) == 0
        else:
            is_target = (idx % self.k) == 0

        target_idx = idx[is_target]   # [T]
        source_idx = idx[~is_target]  # [S]
        return target_idx, source_idx

    def _match_sources_to_targets(self, K, target_idx, source_idx):
        """
        K: [B, N, D]
        """
        B, N, D = K.shape
        K_tgt = F.normalize(K[:, target_idx], dim=-1)  # [B, T, D]
        K_src = F.normalize(K[:, source_idx], dim=-1)  # [B, S, D]

        # 计算余弦相似度: [B, S, T]
        sims = torch.matmul(K_src, K_tgt.transpose(-1, -2))

        # 每个 source 选相似度最大的 target
        best_idx = sims.argmax(dim=-1)             # [B, S] (在 target_idx 内部的下标)
        matched_target_idx = target_idx[best_idx]  # [B, S] (全局下标)

        return matched_target_idx

    def _merge_tokens(self, X, target_idx, source_idx, matched_target_idx):
        """
        X: [B, N, D]
        """
        B, N, D = X.shape
        T = target_idx.shape[0]

        Y = torch.zeros(B, T, D, device=X.device)
        counts = torch.zeros(B, T, 1, device=X.device)

        # 每个 batch 单独做 scatter
        for b in range(B):
            # 先加 target 自己
            Y[b].scatter_add_(0,
                              torch.arange(T, device=X.device).unsqueeze(-1).expand(-1, D),
                              X[b, target_idx])
            counts[b] += 1  # 每个 target 至少有自己

            # 再加 source
            for s, t_global in zip(source_idx, matched_target_idx[b]):
                t_local = (target_idx == t_global).nonzero(as_tuple=True)[0].item()
                Y[b, t_local] += X[b, s]
                counts[b, t_local] += 1

        return Y / counts

    def forward(self, X, K):
        """
        执行 Token Merging

        参数
        ----
        X : [B, N, D] token 表示
        K : [B, N, D] key 向量 (来自最近一层 self-attention)

        返回
        ----
        Y : [B, T, D] 合并后的序列
        """
        B, N, D = X.shape
        target_idx, source_idx = self._partition_by_mod(N, X.device)
        matched_idx = self._match_sources_to_targets(K, target_idx, source_idx)
        Y = self._merge_tokens(X, target_idx, source_idx, matched_idx)
        return Y