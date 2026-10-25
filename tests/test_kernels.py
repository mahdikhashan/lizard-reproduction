import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

import kernels  # noqa: E402
import lizard_attention as lz  # noqa: E402
from test_lizard import tiny_config, tiny_rope  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMALL = ((16, 16), (16, 16))
FORWARD, BACKWARD, ATOL = 1e-2, 3e-2, 1e-3


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def close(a, b, tol):
    a, b = a.float(), b.float()
    return (a - b).norm() <= tol * b.norm() + ATOL * b.numel() ** 0.5


def leaves(*tensors):
    return [t.detach().clone().requires_grad_() for t in tensors]


def compare(kernel, reference, inputs, grad):
    ours = leaves(*inputs)
    theirs = leaves(*inputs)
    y = kernel(*ours)
    y.backward(grad.to(y.dtype))
    expected = reference(*theirs)
    expected.backward(grad.to(expected.dtype))
    assert y.dtype == torch.bfloat16
    assert rel(y, expected) < FORWARD
    for a, b in zip(ours, theirs):
        assert close(a.grad, b.grad, BACKWARD)


def gla_inputs(length, batch=2, heads=3, features=24, dim=20):
    bf = torch.bfloat16
    fq = torch.randn(batch, heads, length, features).softmax(-1).to(bf)
    fk = torch.randn(batch, heads, length, features).softmax(-1).to(bf)
    v = torch.randn(batch, heads, length, dim).to(bf)
    z = 2 * torch.randn(batch, length)
    return [t.to(DEVICE) for t in (fq, fk, v, z)]


def swa_inputs(length, batch=2, heads=3, dim=20, sinks=4):
    q, k, v = (torch.randn(batch, heads, length, dim).to(torch.bfloat16)
               for _ in range(3))
    return [t.to(DEVICE) for t in (q, k, v, torch.randn(sinks))]


@pytest.mark.parametrize("length, blocks", [(1, SMALL), (77, SMALL),
                                            (150, None)])
def test_gla_kernel_matches_torch(length, blocks):
    torch.manual_seed(0)
    inputs = gla_inputs(length)
    blocks = blocks or (kernels.gla.__defaults__)

    def kernel(fq, fk, v, z):
        return kernels.gla(fq, fk, v, F.logsigmoid(z), *blocks)

    def reference(fq, fk, v, z):
        return lz.gla(fq.float(), fk.float(), v.float(), torch.sigmoid(z))

    compare(kernel, reference, inputs, torch.randn_like(inputs[2].float()))


@pytest.mark.parametrize("length, window, blocks", [(1, 8, SMALL),
                                                    (77, 32, SMALL),
                                                    (150, 128, None)])
def test_swa_kernel_matches_torch(length, window, blocks):
    torch.manual_seed(0)
    inputs = swa_inputs(length)
    blocks = blocks or kernels.swa.__defaults__

    def kernel(q, k, v, meta):
        return kernels.swa(q, k, v, meta, window, *blocks)

    def reference(q, k, v, meta):
        return lz.awa(q.float(), k.float(), v.float(), meta, window)

    compare(kernel, reference, inputs, torch.randn_like(inputs[2].float()))


def test_swa_kernel_without_sinks_matches_torch():
    torch.manual_seed(0)
    q, k, v, _ = swa_inputs(40)
    meta = torch.empty(0, device=DEVICE)
    y = kernels.swa(q, k, v, meta, 16, *SMALL)
    expected = lz.awa(q.float(), k.float(), v.float(),
                      torch.full((1,), -1e4, device=DEVICE), 16)
    assert rel(y, expected) < FORWARD


def test_triton_layer_matches_torch_layer():
    torch.manual_seed(0)
    layer = lz.LizardAttention(tiny_config(), 0, window=16, feature_dim=8)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
    layer = layer.to(DEVICE)
    triton_layer = copy.deepcopy(layer)
    triton_layer.backend = "triton"
    x = torch.randn(2, 40, 128, device=DEVICE)
    rope = tuple(t.float().to(DEVICE) for t in tiny_rope(40))
    grad = torch.randn(2, 40, 128, device=DEVICE)
    y = layer(x, rope)[0]
    y.backward(grad)
    y_triton = triton_layer(x, rope)[0]
    y_triton.backward(grad)
    assert rel(y_triton, y) < 2 * FORWARD
    for (name, a), b in zip(triton_layer.named_parameters(),
                            layer.parameters()):
        assert close(a.grad, b.grad, BACKWARD), name
