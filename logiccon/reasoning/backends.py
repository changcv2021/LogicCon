"""Local HF VLM and a standard chat-completions HTTP adapter; no cloud calls at import."""
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from .schema import require


@dataclass
class Response:
    text: str
    usage: dict = field(default_factory=dict)


class TransformersBackend:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        require(torch.cuda.is_available(), "A CUDA GPU in an approved allocation is required")
        self.config, self.torch = config, torch
        torch.manual_seed(config.seed)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        options = dict(revision=config.revision, local_files_only=config.local_files_only,
                       trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(config.model, **options,
                                                       min_pixels=config.min_pixels, max_pixels=config.max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model, **options, dtype=getattr(torch, config.dtype),
            device_map=config.device_map, attn_implementation="sdpa", low_cpu_mem_usage=True)
        self.model.eval()
        devices = getattr(self.model, "hf_device_map", {})
        require(not any(str(v) in {"cpu", "disk"} for v in devices.values()),
                "Weights spilled to CPU/disk; request sufficient GPUs or use a smaller model")

    def generate(self, prompt, image=None):
        from PIL import Image
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        kwargs = dict(text=[rendered], return_tensors="pt", padding=True)
        if image is not None:
            with Image.open(image) as original:
                kwargs["images"] = [original.convert("RGB")]
        # Avoid automatic truncation: it can remove the claim or evidence being checked.
        inputs = self.processor(**kwargs).to(self.model.device)
        prefix = inputs.input_ids.shape[-1]
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens,
                                            do_sample=False, use_cache=True)
        continuation = generated[:, prefix:]
        text = self.processor.batch_decode(continuation, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        return Response(text, {"input_tokens": prefix, "output_tokens": continuation.shape[-1]})


class HTTPBackend:
    """For an explicitly configured compatible endpoint, e.g. a vLLM server.

    Keys are read from the named environment variable and never included in metadata.
    Remote images are embedded as data URLs; the server must support image_url inputs.
    """
    def __init__(self, config):
        self.config = config

    def generate(self, prompt, image=None):
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            mime = mimetypes.guess_type(image)[0]
            require(mime in {"image/jpeg", "image/png", "image/webp"}, "Unsupported image MIME type")
            encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
            content.insert(0, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        body = {"model": self.config.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": self.config.max_new_tokens, "seed": self.config.seed}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.config.base_url.rstrip("/") + "/chat/completions",
                                         json.dumps(body).encode(), headers)
        for attempt in range(self.config.transport_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as stream:
                    result = json.load(stream)
                text = result["choices"][0]["message"]["content"]
                require(isinstance(text, str) and bool(text.strip()), "Endpoint returned no text")
                usage = result.get("usage", {})
                return Response(text, {"input_tokens": usage.get("prompt_tokens", 0),
                                       "output_tokens": usage.get("completion_tokens", 0)})
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.config.transport_retries:
                    raise RuntimeError(f"Endpoint HTTP {exc.code}; response body omitted to avoid leaking credentials") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.config.transport_retries:
                    raise RuntimeError("Endpoint network failure/timeout") from None
            time.sleep(min(2 ** attempt, 30))


def create_backend(config):
    return TransformersBackend(config) if config.kind == "transformers" else HTTPBackend(config)
