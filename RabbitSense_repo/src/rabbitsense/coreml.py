"""Versiones Core ML de V-JEPA 2 y del detector OWLv2 (2026-10-31).

Core ML puede repartir el cálculo entre la GPU, la CPU y el Neural Engine, y deja la GPU más libre para la pantalla.
Se convierten los dos modelos de la fase de actos:
- V-JEPA 2: entrada (1, 16, 3, 256, 256) ya normalizada; salida, la media de los tokens (igual que actmodel.VJEPA).
- OWLv2: solo la parte de imagen. La consulta «a photo of a rabbit» se calcula una vez en PyTorch y queda guardada en el
  modelo; el preprocesado (relleno a cuadrado, 960 px, normalización) lo sigue haciendo el procesador de transformers.
SAM2 y CoTracker3 siguen en PyTorch: el SAM2 de Core ML de Apple es de imagen suelta (sin la memoria entre fotogramas
que usa la segmentación de vídeo) y CoTracker3 tiene bucles y tamaños variables que Core ML no admite bien.

`check` comprueba en el Mac que Core ML da lo mismo que PyTorch antes de usarlo (criterio ADOPT_COREML, fijado antes de
ejecutar): mismo acto en ≥ 95 % de las ventanas, misma detección (conejo sí/no e IoU ≥ 0,9) en ≥ 95 % de los
fotogramas y similitud coseno mediana de V-JEPA 2 ≥ 0,99. Si no se cumple, `analyze` sigue con PyTorch.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .actmodel import FRAMES, IMAGENET, MODEL_ID as VJEPA_ID
from .motion import sha256
from .ytacts import DETECTOR, detections

REPO = Path(__file__).resolve().parents[2]
FOLDER = REPO / 'weights' / 'coreml'
ADOPT_COREML = dict(act_agreement=.95, detection_agreement=.95, box_iou=.9, embedding_cosine=.99)


class VJEPAWrapper:
    """Módulo trazable: vídeo normalizado → media de los tokens del codificador."""

    def __new__(cls, model):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, video):
                return self.model.get_vision_features(video).mean(dim=1)
        return Wrapper()


class OwlImageWrapper:
    """Módulo trazable: píxeles preprocesados → (logit de la consulta fija, cajas) por parche.

    Reproduce image_embedder, class_predictor y box_predictor de transformers con operaciones que Core ML admite: la
    consulta fija ya normalizada sustituye al einsum y las difusiones de forma se hacen con broadcasting."""

    def __new__(cls, model, query):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                q = query.reshape(-1, query.shape[-1])[0]
                self.register_buffer('query', (q / (torch.sqrt((q * q).sum()) + 1e-6)).reshape(-1, 1))

            def forward(self, pixels):
                m = self.model
                hidden = m.owlv2.vision_model.post_layernorm(m.owlv2.vision_model(pixel_values=pixels)[0])
                features = m.layer_norm(hidden[:, 1:, :] * hidden[:, :1, :])
                head = m.class_head
                embeds = head.dense0(features)
                embeds = embeds / (torch.sqrt((embeds * embeds).sum(dim=-1, keepdim=True)) + 1e-6)
                logits = torch.matmul(embeds, self.query)[..., 0]
                scale = head.elu(head.logit_scale(features))[..., 0] + 1
                logits = (logits + head.logit_shift(features)[..., 0]) * scale
                return logits, torch.sigmoid(m.box_head(features) + m.box_bias)
        return Wrapper()


def owl_query(model, tokens):
    """Representación de la consulta de texto, calculada una vez con el modelo completo."""
    import torch
    size = model.config.vision_config.image_size
    with torch.no_grad():
        output = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
                       pixel_values=torch.zeros(1, 3, size, size))
    return output.text_embeds[:1]


def to_coreml(module, example, input_name, output_names, path):
    import coremltools as ct
    import torch
    module = module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example, strict=False)
    model = ct.convert(traced, inputs=[ct.TensorType(name=input_name, shape=tuple(example.shape))],
                       outputs=[ct.TensorType(name=n) for n in output_names], convert_to='mlprogram',
                       minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16)
    model.save(str(path))
    return path


def convert(out=FOLDER, vjepa=None, owl=None, tokens=None, frames=FRAMES):
    """Convierte ambos modelos (los reales de Hugging Face si no se dan) y guarda su descripción."""
    import torch
    import transformers
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(transformers=transformers.__version__, torch=torch.__version__, python=platform.python_version())
    if vjepa is None:
        vjepa = transformers.AutoModel.from_pretrained(VJEPA_ID)
        try:
            processor = transformers.AutoVideoProcessor.from_pretrained(VJEPA_ID)
            mean, std = list(processor.image_mean), list(processor.image_std)
        except (OSError, ValueError, TypeError, AttributeError):
            mean, std = IMAGENET
        meta['vjepa'] = dict(model_id=VJEPA_ID)
    else:
        mean, std = IMAGENET
        meta['vjepa'] = dict(model_id='test')
    size = int(getattr(vjepa.config, 'crop_size', 256))
    to_coreml(VJEPAWrapper(vjepa), torch.rand(1, frames, 3, size, size), 'video', ['embedding'], out / 'vjepa2.mlpackage')
    meta['vjepa'].update(size=size, frames=frames, mean=mean, std=std)
    if owl is None:
        processor = transformers.Owlv2Processor.from_pretrained(DETECTOR['model_id'])
        owl = transformers.Owlv2ForObjectDetection.from_pretrained(DETECTOR['model_id'])
        tokens = processor(text=[DETECTOR['text']], return_tensors='pt')
        meta['owl'] = dict(model_id=DETECTOR['model_id'], text=DETECTOR['text'])
    else:
        meta['owl'] = dict(model_id='test', text='test')
    owl = owl.eval()
    query = owl_query(owl, tokens)
    image = owl.config.vision_config.image_size
    to_coreml(OwlImageWrapper(owl, query), torch.rand(1, 3, image, image), 'pixels', ['logits', 'boxes'],
              out / 'owlv2.mlpackage')
    meta['owl'].update(size=image, threshold=DETECTOR['threshold'])
    (out / 'modelos.json').write_text(json.dumps(meta, indent=2))
    print(f'Modelos Core ML en {out}', flush=True)
    return meta


def load(path, compute='all'):
    import coremltools as ct
    units = dict(all=ct.ComputeUnit.ALL, ne=ct.ComputeUnit.CPU_AND_NE, gpu=ct.ComputeUnit.CPU_AND_GPU)[compute]
    return ct.models.MLModel(str(path), compute_units=units)


class CoreMLVJEPA:
    """Misma interfaz que actmodel.VJEPA, con el modelo convertido."""

    def __init__(self, folder=FOLDER, compute='all', model=None):
        meta = json.loads((Path(folder) / 'modelos.json').read_text())['vjepa']
        self.model = model or load(Path(folder) / 'vjepa2.mlpackage', compute)
        self.size, self.frames = meta['size'], meta['frames']
        self.mean = np.array(meta['mean'], np.float32).reshape(1, 1, 3, 1, 1)
        self.std = np.array(meta['std'], np.float32).reshape(1, 1, 3, 1, 1)
        self.info = dict(model_id=meta['model_id'], backend='coreml', compute=compute, size=self.size, frames=self.frames)

    def __call__(self, clip):
        rgb = np.ascontiguousarray(clip[..., ::-1]).transpose(0, 3, 1, 2)[None].astype(np.float32) / 255.
        return np.asarray(self.model.predict({'video': (rgb - self.mean) / self.std})['embedding'], np.float32)[0]


class CoreMLDetector:
    """Misma interfaz que ytacts.Detector, con la parte de imagen de OWLv2 convertida."""

    def __init__(self, folder=FOLDER, compute='all', model=None, processor=None):
        meta = json.loads((Path(folder) / 'modelos.json').read_text())['owl']
        self.model = model or load(Path(folder) / 'owlv2.mlpackage', compute)
        if processor is None:
            import transformers
            processor = transformers.Owlv2ImageProcessor.from_pretrained(meta['model_id'])
        self.processor = processor
        self.info = dict(model_id=meta['model_id'], text=meta['text'], threshold=DETECTOR['threshold'], backend='coreml',
                         compute=compute)

    def __call__(self, images, batch=1):
        found = []
        for image in images:
            pixels = self.processor(images=[np.ascontiguousarray(image[..., ::-1])], return_tensors='np')['pixel_values']
            output = self.model.predict({'pixels': pixels.astype(np.float32)})
            scores = 1 / (1 + np.exp(-np.asarray(output['logits'], np.float32)[0]))
            found.append(detections(scores, np.asarray(output['boxes'], np.float32)[0], image.shape))
        return found


def box_iou(a, b):
    from .ytacts import iou
    return iou(a[1:], b[1:])


def check(manifest, windows, videos, out, device='mps', compute='all', torch_models=None, coreml_models=None,
          model_file=None):
    """Core ML frente a PyTorch en las ventanas de los 17 vídeos y de los vídeos nuevos (ADOPT_COREML)."""
    from .actmodel import cases_with_windows
    from .acts import square
    from .auto import square_around, stream_windows
    from .pointtrack import frame_times
    from .ytacts import predict, probe_frames, window_detection
    frozen = json.loads(Path(model_file or REPO / 'benchmarks' / 'ytactos_modelo.json').read_text())
    if torch_models is None:
        from .actmodel import VJEPA
        from .ytacts import Detector
        torch_models = (Detector(device), VJEPA(device))
    coreml_models = coreml_models or (CoreMLDetector(compute=compute), CoreMLVJEPA(compute=compute))
    sources = []
    if manifest:
        table = pd.read_csv(windows, dtype={'case': str})
        sources += [(video, {round(float(s), 3) for s in mine.start}) for _, video, _, mine in
                    cases_with_windows(manifest, table)]
    sources += [(Path(v).resolve(), None) for v in videos or []]
    rows, frames_rows, timing = [], [], dict(detector_torch=[], detector_coreml=[], vjepa_torch=[], vjepa_coreml=[])
    for video, wanted in sources:
        times = frame_times(video)
        for k, frames, images in stream_windows(video, times):
            start = round(k * 2., 3)
            if (wanted is not None and start not in wanted) or len(frames) < 2:
                continue
            probe = [images[f] for f in probe_frames(frames)]
            found = {}
            for name, detector in (('torch', torch_models[0]), ('coreml', coreml_models[0])):
                began = time.perf_counter()
                found[name] = detector(probe)
                timing[f'detector_{name}'].append((time.perf_counter() - began) / len(probe))
            for a, b in zip(found['torch'], found['coreml']):
                frames_rows.append(dict(video=video.stem, start=start, torch=bool(a), coreml=bool(b),
                                        iou=box_iou(a[0], b[0]) if a and b else np.nan))
            summary = window_detection(found['torch'])
            if not summary['union']:
                continue
            box = square_around(*summary['union'], images[frames[0]].shape)
            picks = probe_frames(frames, FRAMES)
            vectors = {}
            for name, encoder in (('torch', torch_models[1]), ('coreml', coreml_models[1])):
                clip = np.stack([square(images[f], box, encoder.size) for f in picks])
                began = time.perf_counter()
                vectors[name] = np.asarray(encoder(clip), np.float32)
                timing[f'vjepa_{name}'].append(time.perf_counter() - began)
            a, b = vectors['torch'], vectors['coreml']
            acts = predict(frozen, np.stack([a, b]))
            rows.append(dict(video=video.stem, start=start, cosine=float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
                             act_torch=acts[0], act_coreml=acts[1]))
        print(f'{video.name}: {sum(r["video"] == video.stem for r in rows)} ventanas', flush=True)
    windows_table, frames_table = pd.DataFrame(rows), pd.DataFrame(frames_rows)
    same_detection = (frames_table.torch == frames_table.coreml) & (frames_table.iou.fillna(1.) >= ADOPT_COREML['box_iou'])
    result = dict(
        criterion=ADOPT_COREML, windows=int(len(windows_table)), frames=int(len(frames_table)),
        act_agreement=float((windows_table.act_torch == windows_table.act_coreml).mean()),
        detection_agreement=float(same_detection.mean()),
        box_iou_median=float(frames_table.iou.median()), embedding_cosine_median=float(windows_table.cosine.median()),
        embedding_cosine_min=float(windows_table.cosine.min()),
        seconds_per_call={k: float(np.median(v)) if v else None for k, v in timing.items()},
        compute=compute, device=device, source_sha256=sha256(Path(__file__)))
    result['adopt_coreml'] = bool(result['act_agreement'] >= ADOPT_COREML['act_agreement'] and
                                  result['detection_agreement'] >= ADOPT_COREML['detection_agreement'] and
                                  result['embedding_cosine_median'] >= ADOPT_COREML['embedding_cosine'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows_table.to_csv(out / 'ventanas.csv', index=False)
    frames_table.to_csv(out / 'detecciones.csv', index=False)
    (out / 'equivalencia.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    speed = result['seconds_per_call']
    print(f"Mismo acto {result['act_agreement']:.3f} | misma detección {result['detection_agreement']:.3f} | coseno "
          f"{result['embedding_cosine_median']:.4f} | V-JEPA {speed['vjepa_torch']} s → {speed['vjepa_coreml']} s | "
          f"OWLv2 {speed['detector_torch']} s → {speed['detector_coreml']} s → "
          f"{'se adopta Core ML' if result['adopt_coreml'] else 'se mantiene PyTorch'}", flush=True)
    return result


def adopted(folder=FOLDER, record=REPO / 'cierre' / 'coreml' / 'equivalencia.json'):
    """Core ML se usa por defecto solo si los modelos existen, pasaron la comprobación de equivalencia y son más rápidos
    que PyTorch en los dos modelos (añadido el 2026-10-31 tras ver que, con compute='all', eran más lentos; no afecta a
    la validez, solo a la velocidad). Con `--backend coreml` se usan igualmente."""
    record = Path(record)
    if not (Path(folder) / 'modelos.json').is_file() or not record.is_file():
        return False
    result = json.loads(record.read_text())
    speed = result.get('seconds_per_call', {})
    faster = all(speed.get(f'{m}_coreml') is not None and speed.get(f'{m}_torch') is not None and
                 speed[f'{m}_coreml'] < speed[f'{m}_torch'] for m in ('detector', 'vjepa'))
    return bool(result.get('adopt_coreml', False) and faster)


def main():
    parser = argparse.ArgumentParser(description='Modelos Core ML de la fase de actos')
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('convert')
    command.add_argument('--out', default=str(FOLDER))
    command = sub.add_parser('check', help='Core ML frente a PyTorch (ADOPT_COREML)')
    command.add_argument('--manifest')
    command.add_argument('--windows', default=str(REPO / 'cierre' / 'actos' / 'ventanas_indice.csv'))
    command.add_argument('--videos', nargs='*')
    command.add_argument('--out', default=str(REPO / 'cierre' / 'coreml'))
    command.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    command.add_argument('--compute', choices=['all', 'ne', 'gpu'], default='all')
    args = parser.parse_args()
    if args.command == 'convert':
        convert(args.out)
    else:
        check(args.manifest, args.windows, args.videos, args.out, args.device, args.compute)


if __name__ == '__main__':
    main()
