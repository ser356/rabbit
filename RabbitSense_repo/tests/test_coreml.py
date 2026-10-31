import json

import numpy as np
import pytest

from rabbitsense import coreml, ytacts
from test_auto import frozen_model, late_video
from test_actmodel import Stub
from test_ytacts import Bright

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')


def tiny_owl():
    config = transformers.Owlv2Config(
        text_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2, vocab_size=99,
                         max_position_embeddings=16),
        vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=1, num_attention_heads=2,
                           image_size=64, patch_size=16), projection_dim=32)
    torch.manual_seed(0)
    tokens = dict(input_ids=torch.randint(1, 99, (1, 16)), attention_mask=torch.ones(1, 16, dtype=torch.long))
    return transformers.Owlv2ForObjectDetection(config).eval(), tokens


def tiny_vjepa():
    config = transformers.VJEPA2Config(crop_size=32, frames_per_clip=16, patch_size=16, tubelet_size=2, hidden_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, mlp_ratio=2, pred_hidden_size=16,
                                       pred_num_hidden_layers=1, pred_num_attention_heads=2)
    torch.manual_seed(0)
    return transformers.VJEPA2Model(config).eval()


def test_owl_image_wrapper_matches_the_full_model():
    model, tokens = tiny_owl()
    pixels = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        full = model(input_ids=tokens['input_ids'].repeat(2, 1), attention_mask=tokens['attention_mask'].repeat(2, 1),
                     pixel_values=pixels)
        logits, boxes = coreml.OwlImageWrapper(model, coreml.owl_query(model, tokens))(pixels)
    assert torch.allclose(logits, full.logits[..., 0], atol=1e-5) and torch.allclose(boxes, full.pred_boxes, atol=1e-6)


class TorchAsCoreML:
    """Hace de MLModel: predict({nombre: array}) con el módulo trazable en PyTorch."""

    def __init__(self, module, input_name, outputs):
        self.module, self.input_name, self.outputs = module.eval(), input_name, outputs

    def predict(self, inputs):
        with torch.no_grad():
            result = self.module(torch.from_numpy(np.asarray(inputs[self.input_name], np.float32)))
        result = result if isinstance(result, tuple) else (result,)
        return {name: value.numpy() for name, value in zip(self.outputs, result)}


def test_coreml_detector_glue_gives_the_same_detections_as_pytorch(tmp_path):
    model, tokens = tiny_owl()
    processor = transformers.Owlv2ImageProcessor(size={'height': 64, 'width': 64})
    reference = ytacts.Detector('cpu', 'tiny', 'a rabbit', parts=(model, processor, tokens))
    (tmp_path / 'modelos.json').write_text(json.dumps(dict(owl=dict(model_id='tiny', text='a rabbit'))))
    fake = TorchAsCoreML(coreml.OwlImageWrapper(model, coreml.owl_query(model, tokens)), 'pixels', ['logits', 'boxes'])
    detector = coreml.CoreMLDetector(tmp_path, model=fake, processor=processor)
    images = [np.random.default_rng(k).integers(0, 255, (48, 80, 3)).astype(np.uint8) for k in range(3)]
    for a, b in zip(reference(images), detector(images)):
        assert len(a) == len(b) and all(np.allclose(x, y, atol=1e-4) for x, y in zip(a, b))


def test_coreml_vjepa_glue_matches_pytorch(tmp_path):
    from rabbitsense.actmodel import VJEPA
    model = tiny_vjepa()
    model.save_pretrained(tmp_path / 'tiny')
    reference = VJEPA('cpu', str(tmp_path / 'tiny'))
    (tmp_path / 'modelos.json').write_text(json.dumps(dict(vjepa=dict(
        model_id='tiny', size=32, frames=16, mean=reference.mean.flatten().tolist(), std=reference.std.flatten().tolist()))))
    fake = TorchAsCoreML(coreml.VJEPAWrapper(reference.model), 'video', ['embedding'])
    encoder = coreml.CoreMLVJEPA(tmp_path, model=fake)
    clip = np.random.default_rng(0).integers(0, 255, (16, 32, 32, 3)).astype(np.uint8)
    assert encoder.size == 32 and np.allclose(encoder(clip), reference(clip), atol=1e-4)


def test_conversion_writes_both_packages(tmp_path):
    pytest.importorskip('coremltools')
    model, tokens = tiny_owl()
    meta = coreml.convert(tmp_path, vjepa=tiny_vjepa(), owl=model, tokens=tokens)
    assert (tmp_path / 'vjepa2.mlpackage').is_dir() and (tmp_path / 'owlv2.mlpackage').is_dir()
    assert meta['vjepa']['size'] == 32 and meta['owl']['size'] == 64
    assert json.loads((tmp_path / 'modelos.json').read_text())['vjepa']['frames'] == 16


def test_check_applies_the_equivalence_criterion(tmp_path):
    late_video(tmp_path / 'late.mp4')
    result = coreml.check(None, None, [tmp_path / 'late.mp4'], tmp_path / 'out', 'cpu', torch_models=(Bright(), Stub()),
                          coreml_models=(Bright(), Stub()), model_file=frozen_model(tmp_path / 'm.json'))
    assert result['windows'] == 2 and result['act_agreement'] == 1 and result['detection_agreement'] == 1
    assert result['adopt_coreml'] and set(result['seconds_per_call']) == {'detector_torch', 'detector_coreml',
                                                                         'vjepa_torch', 'vjepa_coreml'}
    record = tmp_path / 'out' / 'equivalencia.json'
    assert not coreml.adopted(tmp_path / 'sin_modelos', record)  # sin modelos convertidos no se usa
    (tmp_path / 'modelos').mkdir()
    (tmp_path / 'modelos' / 'modelos.json').write_text('{}')
    assert coreml.adopted(tmp_path / 'modelos', record)
