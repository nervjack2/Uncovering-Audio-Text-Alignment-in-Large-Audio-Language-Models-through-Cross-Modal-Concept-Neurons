"""Forward pre-hooks on every FFN `down_proj`. Their input is the SwiGLU output
act(gate_proj(x)) * up_proj(x), whose dimensions are the neurons studied in the paper."""
import torch


def ffn_layers(model) -> dict[str, torch.nn.Module]:
    found = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2:] == ["mlp", "down_proj"] and "visual" not in parts:
            found[name] = module
    order = sorted(found, key=lambda n: [int(p) if p.isdigit() else p for p in n.split(".")])
    return {n: found[n] for n in order}


class ActivationRecorder:
    """Records the FFN activations of the last forward pass."""

    def __init__(self, model):
        self.layers = list(ffn_layers(model))
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = [m.register_forward_pre_hook(self._hook(n)) for n, m in ffn_layers(model).items()]

    def _hook(self, name):
        def hook(module, args):
            self.acts[name] = args[0].detach().cpu().float()
        return hook

    def pooled(self, positions: list[int]) -> list[torch.Tensor]:
        """Mean activation over the given token positions, one (D,) vector per layer."""
        idx = torch.tensor(positions, dtype=torch.long)
        return [self.acts[n][0, idx].mean(dim=0) for n in self.layers]

    def remove(self):
        for h in self.handles:
            h.remove()


class NeuronIntervention:
    """Deactivates a neuron set (sets it to zero) or adds coef * vector to it, at every token position."""

    def __init__(self, model):
        self.modules = list(ffn_layers(model).values())
        self.handles = []
        self.coef = 0.0

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def zero(self, neurons: dict[int, list[int]]):
        """`neurons` maps a layer index to neuron indices."""
        self.clear()
        for layer, idx in neurons.items():
            idx = torch.tensor(sorted(idx), dtype=torch.long)

            def hook(module, args, idx=idx):
                x = args[0].clone()
                x[..., idx.to(x.device)] = 0
                return (x, *args[1:])
            self.handles.append(self.modules[layer].register_forward_pre_hook(hook))

    def add(self, vectors: dict[int, tuple[list[int], list[float]]]):
        """`vectors` maps a layer index to (neuron indices, values to add per unit of coef)."""
        self.clear()
        for layer, (idx, vec) in vectors.items():
            idx_t = torch.tensor(idx, dtype=torch.long)
            vec_t = torch.tensor(vec, dtype=torch.float32)

            def hook(module, args, idx=idx_t, vec=vec_t):
                x = args[0].clone()
                i = idx.to(x.device)
                x[..., i] = x[..., i] + self.coef * vec.to(x.device, x.dtype)
                return (x, *args[1:])
            self.handles.append(self.modules[layer].register_forward_pre_hook(hook))
