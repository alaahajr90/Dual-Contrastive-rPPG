import torch
from rppg.stage1.model import Encoder, ProjectionHead
from rppg.stage2.attention import apply_rope
from rppg.stage2.model import RPPGModel


def test_rotary_shapes():
    q = torch.randn(1, 8, 12, 2)
    k = torch.randn_like(q)
    a, b = apply_rope(q, k)
    assert a.shape == b.shape == q.shape
    assert torch.allclose(a.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_encoder_and_decoder_share_import_without_duplicate_models():
    torch.set_num_threads(2)
    model = RPPGModel(Encoder(), memory_blocks=1, long_range_blocks=1)
    model.eval()
    with torch.no_grad():
        video = torch.rand(1, 3, 8, 16, 16)
        seq = model.encoder(video)
        assert seq.shape == (1, 8, 64)
        projected = ProjectionHead()(seq)
        assert projected.shape == (1, 128)
        outputs = model(video)
    assert len(outputs) == 4
    assert all(p.shape == (1, 8) for p in outputs)
