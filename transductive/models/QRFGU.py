
import torch
import torch.nn as nn

class QRFGU(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                                  nn.Sigmoid())
        self.hidden_trans = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh()
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, message: torch.Tensor, query_r: torch.Tensor, hidden_state: torch.Tensor):
        update_value, reset_value = self.gate(torch.cat([message, query_r, hidden_state], dim=1)).chunk(2, dim=1)
        hidden_candidate = self.hidden_trans(torch.cat([message, reset_value * hidden_state], dim=1))
        hidden_state = (1 - update_value) * hidden_state + update_value * hidden_candidate
        return hidden_state
