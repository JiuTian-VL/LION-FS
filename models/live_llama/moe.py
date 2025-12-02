import typing

import torch
import torch.nn as nn

class GroupedMoE(nn.Module):
    def __init__(
        self, 
        num_experts=2, 
        seq_dim=256, 
        router_method: typing.Literal["mlp", "add"]="mlp"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.router_method = router_method
        if router_method == 'mlp':
            self.router = nn.Sequential(
                nn.Linear(seq_dim, seq_dim),
                nn.GELU(),
                nn.Linear(seq_dim, num_experts),
            )
        else:
            assert router_method == "add", f"router_method({router_method}) not supported"

    def forward(self, inputs_embeds, seq_embeds):
        """
        inputs_embeds (List of `torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`)
        """
        hidden_states = torch.zeros_like(inputs_embeds[0])
        bs = inputs_embeds[0].shape[0]
        if self.router_method == 'add':
            ratios = torch.ones(bs, len(inputs_embeds), device=inputs_embeds[0].device)
        else:
            ratios = self.router(seq_embeds)
            ratios = torch.softmax(ratios, dim=-1)
        for i in range(self.num_experts):
            hidden_states += ratios[:,i].view(-1,1,1) * inputs_embeds[i]
        return hidden_states