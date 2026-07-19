from __future__ import annotations

import builtins
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from research.kmd2_ablation.architecture import TARGET_LAYERS


def _execute_pickle_marker(path: str, payload_kind: str) -> object:
    Path(path).write_text("pickle executed", encoding="utf-8")
    if payload_kind == "data":
        return {
            "train": [{"example_id": "e0", "input_ids": [0, 1, 2]}],
            "eval": [{"example_id": "eval0", "input_ids": [0, 1, 2]}],
        }
    return {}


class _PickleMarkerPayload:
    def __init__(self, path: Path, payload_kind: str) -> None:
        self.path = str(path)
        self.payload_kind = payload_kind

    def __reduce__(self):
        return _execute_pickle_marker, (self.path, self.payload_kind)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_R1_SUFFIXES = (
    "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight",
    "conv1d.weight", "dt_bias", "A_log", "norm.weight", "out_proj.weight",
    "rot_proj.weight", "rot_proj.bias", "decay_chan", "bw_off",
)


def _canonical_architecture_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, key_head_dim: int = 4,
):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=key_head_dim, linear_value_head_dim=3, linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    class Block(torch.nn.Module):
        def __init__(self): super().__init__(); self.linear_attn = torch.nn.Linear(2, 2)
    class Backbone(torch.nn.Module):
        def __init__(self): super().__init__(); self.layers = torch.nn.ModuleList([Block() for _ in range(23)])
    class Model(torch.nn.Module):
        def __init__(self): super().__init__(); self.config = config; self.model = Backbone(); self.outside = torch.nn.Parameter(torch.ones(1))
    class Manager:
        def __init__(self, model): self.model = model
        def apply_upgrade(self):
            for index in TARGET_LAYERS:
                self.model.model.layers[index].linear_attn = KMD2NativeAttn(config, layer_idx=index)
            return list(TARGET_LAYERS)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    reference = KMD2NativeAttn(config, layer_idx=0)
    reference_state = reference.state_dict()
    assert tuple(sorted(reference_state)) == tuple(sorted(_R1_SUFFIXES))
    checkpoint = {
        f"model.layers.{index}.linear_attn.{suffix}": reference_state[suffix].detach().clone()
        for index in TARGET_LAYERS for suffix in _R1_SUFFIXES
    }
    model_path, checkpoint_path, data_path = (tmp_path / "model", tmp_path / "checkpoint.pt", tmp_path / "data")
    model_path.write_bytes(b"model"); torch.save(checkpoint, checkpoint_path); data_path.write_bytes(b"data")
    def spec():
        return QwenArmLoadSpec(arm="native", job_id="canonical-r1", model_asset=_asset("model", model_path), native_checkpoint=_asset("native_checkpoint", checkpoint_path), data_asset=_asset("data", data_path), cache_resume=None, trainable_names=("model.layers.0.linear_attn.in_proj_qkv.weight",), pre_replacement_checkpoint_sha256=_sha256(checkpoint_path), architecture_arm_id="gdn2-channel-r1", architecture_registry_sha256=registry_sha256())
    return Model, Manager, KMD2NativeAttn, checkpoint, checkpoint_path, spec, config


def test_canonical_architecture_checkpoint_accepts_exact_18_by_13_and_orders_events(tmp_path, monkeypatch):
    from research.kmd2_ablation.qwen_backend import load_qwen_arm
    Model, Manager, Native, checkpoint, _path, spec, config = _canonical_architecture_case(tmp_path, monkeypatch)
    class Replacement(Native):
        def transformation_manifest(self):
            return {"copied": tuple(self.state_dict()), "transformed": (), "new": ()}
    calls = []
    def factory(clone, _config): clone.__class__ = Replacement; calls.append(clone.layer_idx); return clone
    events = []
    loaded = load_qwen_arm(spec(), model_config=config, cache_config=None, base_model_loader=lambda *_a, **_k: Model(), manager_factory=lambda model, _c: Manager(model), architecture_factory=factory, architecture_expected_type=Replacement, architecture_verifier=lambda _m, indices: (indices == TARGET_LAYERS) or (_ for _ in ()).throw(AssertionError(indices)), event=events.append)
    assert len(checkpoint) == 18 * 13
    assert loaded.upgraded_indices == TARGET_LAYERS
    assert calls == list(TARGET_LAYERS)
    assert events == ["validate_assets", "load_model", "native_install_r1", "checkpoint_overlay_complete", "prepare_replacements", "swap_replacements", "configure_trainables", "verify_conversion"]


def test_production_architecture_rejects_nonexact_trainable_manifest(tmp_path, monkeypatch):
    from research.kmd2_ablation.qwen_backend import load_qwen_arm
    Model, Manager, _Native, _checkpoint, _path, spec, config = _canonical_architecture_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="architecture_trainable_manifest_mismatch"):
        load_qwen_arm(
            spec(), model_config=config, cache_config=None,
            base_model_loader=lambda *_a, **_k: Model(),
            manager_factory=lambda model, _c: Manager(model),
        )


@pytest.mark.parametrize("arm,module_name", [
    ("gdn2-mimo-r4-braid-shared-hola-w64", "QwenSharedBraidHybrid"),
    ("gdn2-mimo-r4-braid-four-state-hola-w64", "QwenFourStateHybrid"),
])
def test_real_loader_accepts_canonical_hybrid_trainable_manifest(tmp_path, monkeypatch, arm, module_name):
    from research.kmd2_ablation.qwen_backend import load_qwen_arm
    from research.kmd2_ablation.qwen_hybrid_math import REFERENCE_IMPLEMENTATION
    Model, Manager, Native, _checkpoint, _path, spec_factory, config = _canonical_architecture_case(
        tmp_path, monkeypatch, key_head_dim=(4 if "shared" in arm else 8),
    )
    if "shared" in arm:
        from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid as Hybrid
    else:
        from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid as Hybrid
    prototype = Hybrid.from_native(Native(config, layer_idx=0))
    names = tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS for suffix, _ in prototype.named_parameters()
    ))
    spec = dataclasses.replace(spec_factory(), architecture_arm_id=arm, trainable_names=names)
    loaded = load_qwen_arm(spec, model_config=config, cache_config=None,
        base_model_loader=lambda *_a, **_k: Model(), manager_factory=lambda model, _c: Manager(model))
    assert loaded.architecture_implementation == REFERENCE_IMPLEMENTATION
    assert "hybrid_r4_scan" not in loaded.architecture_implementation
    assert "triton" not in loaded.architecture_implementation.lower()
    assert all(type(loaded.model.model.layers[i].linear_attn).__name__ == module_name for i in TARGET_LAYERS)
    assert loaded.trainable_names == names


@pytest.mark.parametrize("arm", [
    "gdn2-mimo-r4-braid-shared-hola-w64",
    "gdn2-mimo-r4-braid-four-state-hola-w64",
])
def test_hybrid_checkpoint_element_counts_match_materialized_module(monkeypatch, arm):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_checkpoint import hybrid_tensor_element_counts

    key_head_dim = 4 if arm.endswith("shared-hola-w64") else 8
    config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=key_head_dim,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1e-6,
    )
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(config, layer_idx=0)
    if arm.endswith("shared-hola-w64"):
        from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
        module = QwenSharedBraidHybrid.from_native(native)
    else:
        from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
        module = QwenFourStateHybrid.from_native(native)
    parameter_names = {name for name, _parameter in module.named_parameters()}
    persistent_buffers = {
        name: tensor
        for name, tensor in module.state_dict().items()
        if name not in parameter_names
    }

    counts = hybrid_tensor_element_counts(
        architecture_arm_id=arm,
        heads=2,
        key_dim=key_head_dim,
        value_dim=3,
        hidden_size=12,
        conv_kernel=3,
    )

    assert counts["parameter_elements"] == sum(
        parameter.numel() for parameter in module.parameters()
    )
    assert counts["persistent_buffer_elements"] == sum(
        tensor.numel() for tensor in persistent_buffers.values()
    )


def test_hybrid_checkpoint_element_counts_reject_overflow():
    from research.kmd2_ablation.qwen_checkpoint import hybrid_tensor_element_counts

    with pytest.raises(ValueError, match="exact element-count bound"):
        hybrid_tensor_element_counts(
            architecture_arm_id="gdn2-mimo-r4-braid-four-state-hola-w64",
            heads=16,
            key_dim=128,
            value_dim=128,
            hidden_size=1 << 62,
            conv_kernel=4,
        )


@pytest.mark.parametrize("shared", [True, False])
def test_metadata_state_history_formula_matches_materialized_cache(monkeypatch, shared):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.resource_probes import _hybrid_state_history_components

    key_head_dim = 4 if shared else 8
    native_config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=key_head_dim,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1e-6,
    )
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(native_config, layer_idx=0)
    if shared:
        from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
        module = QwenSharedBraidHybrid.from_native(native)
    else:
        from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
        module = QwenFourStateHybrid.from_native(native)
    cache = module._initial_cache(torch.zeros(2, 1, 12))
    actual = sum(
        value.numel() * value.element_size()
        for name, value in cache.__dict__.items()
        if name != "hola_state"
    )
    experiment = SimpleNamespace(model=SimpleNamespace(
        num_layers=1, num_heads=2, state_key_dim=key_head_dim, state_value_dim=3,
    ))

    components = _hybrid_state_history_components(
        experiment,
        shared=shared,
        hidden_size=12,
        conv_kernel=3,
        batch_size=2,
        convolution_element_bytes=4,
    )

    assert sum(components.values()) == actual
    assert module.resource_report(batch_size=2)["persistent_bytes"] == actual


def test_hybrid_conversion_preloader_negative_identity_matrix_never_calls_loader(tmp_path, monkeypatch):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_backend import (ExternalAssetIdentity,
        NativeCheckpointError, QwenArmLoadSpec, _directory_identity, load_qwen_arm)
    from research.kmd2_ablation.qwen_checkpoint import (QwenHybridCheckpointIdentity,
        build_qwen_architecture_checkpoint, expected_hybrid_tensor_contract,
        source_conversion_sha256)
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    frozen = {"hidden_size":12,"linear_num_value_heads":2,"linear_num_key_heads":2,
        "linear_key_head_dim":4,"linear_value_head_dim":3,"linear_conv_kernel_dim":3,
        "rms_norm_eps":1e-6,"rms_norm_type":"RMSNorm","dtype":"torch.float32",
        "use_cache":False,"num_hidden_layers":23,"tie_word_embeddings":False,
        "rope_theta":10000.0,"rope_scaling":None,"max_position_embeddings":4096,
        "partial_rotary_factor":1.0}
    model_dir=tmp_path/"model"; model_dir.mkdir(); (model_dir/"config.json").write_text(json.dumps(frozen),encoding="utf-8")
    size,model_hash=_directory_identity(model_dir)
    model_asset=ExternalAssetIdentity("model",model_dir,"directory",size,model_hash)
    data=tmp_path/"data"; teacher=tmp_path/"teacher"; data.write_bytes(b"data"); teacher.write_bytes(b"teacher")
    class Block(torch.nn.Module):
        def __init__(self, index): super().__init__(); self.linear_attn=KMD2NativeAttn(config,index) if index in TARGET_LAYERS else torch.nn.Linear(2,2)
    class Source(torch.nn.Module):
        def __init__(self): super().__init__(); self.model=torch.nn.Module(); self.model.layers=torch.nn.ModuleList([Block(i) for i in range(23)])
    monkeypatch.setenv("GDN3_KMD2_ROUT","1"); source=Source()
    arm="gdn2-mimo-r4-braid-shared-hola-w64"
    targets=tuple(sorted(f"model.layers.{i}.linear_attn" for i in TARGET_LAYERS))
    provisional=QwenHybridCheckpointIdentity(architecture_registry_sha256=registry_sha256(),
        implementation_sha256="2"*64,model_tree_sha256=model_hash,ordered_examples_sha256=_sha256(data),
        pre_replacement_checkpoint_sha256=source_conversion_sha256(source,targets),teacher_sha256=_sha256(teacher),
        frozen_qwen_config=frozen,cache_policy={"policy":"exact_outer","width":64,"block_size":256},
        trainable_manifest=({"name":"placeholder","shape":[],"dtype":"torch.float32"},),target_module_names=targets)
    contract=expected_hybrid_tensor_contract(provisional,arm)
    manifest=tuple(sorted(({"name":f"{target}.{name}","shape":list(shape),"dtype":dtype}
        for target in targets for name,(shape,dtype) in contract.items()),key=lambda row:row["name"]))
    identity=dataclasses.replace(provisional,trainable_manifest=manifest)
    payload=build_qwen_architecture_checkpoint(source,target_module_names=targets,architecture_arm_id=arm,identity=identity)
    checkpoint=tmp_path/"conversion.pt"; torch.save(payload,checkpoint)
    bad_teacher=tmp_path/"bad-teacher"; bad_teacher.write_bytes(b"different teacher")
    bad_model=tmp_path/"bad-model"; bad_model.mkdir(); (bad_model/"config.json").write_text(json.dumps(frozen),encoding="utf-8"); (bad_model/"extra").write_bytes(b"x")
    bad_size,bad_model_hash=_directory_identity(bad_model)
    bad_model_asset=ExternalAssetIdentity("model",bad_model,"directory",bad_size,bad_model_hash)
    def make_spec(**changes):
        base=QwenArmLoadSpec(arm="native",job_id="hybrid",model_asset=model_asset,
            native_checkpoint=_asset("checkpoint",checkpoint),data_asset=_asset("data",data),cache_resume=None,
            trainable_names=tuple(row["name"] for row in manifest),pre_replacement_checkpoint_sha256=_sha256(checkpoint),
            architecture_arm_id=arm,architecture_registry_sha256=registry_sha256(),teacher_asset=_asset("teacher",teacher),
            architecture_implementation_sha256="2"*64,source_checkpoint_sha256=identity.pre_replacement_checkpoint_sha256,
            frozen_qwen_config=frozen,architecture_cache_policy={"policy":"exact_outer","width":64,"block_size":256})
        return dataclasses.replace(base,**changes)
    cases=[{"architecture_implementation_sha256":None},{"architecture_implementation_sha256":"9"*64},
        {"source_checkpoint_sha256":"9"*64},{"frozen_qwen_config":{**frozen,"rope_theta":9.0}},
        {"architecture_cache_policy":{"policy":"recency","width":64,"block_size":256}},
        {"trainable_names":tuple(row["name"] for row in manifest[:-1])},
        {"teacher_asset":_asset("teacher",bad_teacher)},{"model_asset":bad_model_asset}]
    calls=[]
    for changes in cases:
        with pytest.raises(NativeCheckpointError):
            load_qwen_arm(make_spec(**changes),model_config=config,cache_config=None,
                base_model_loader=lambda *_a,**_k:calls.append(1))
    base_payload=copy.deepcopy(payload)
    corruptions=[]
    duplicate=copy.deepcopy(base_payload); duplicate["identity"]["target_module_names"][1]=duplicate["identity"]["target_module_names"][0]; corruptions.append(duplicate)
    wrong=copy.deepcopy(base_payload); first=next(iter(wrong["model_state"])); wrong["model_state"][first]=wrong["model_state"][first][:-1]
    for row in wrong["tensor_manifest"]:
        if row["name"]==first: row["shape"]=list(wrong["model_state"][first].shape)
    target,suffix=first.rsplit(".",1)[0],None
    for layer_name,layer in wrong["conversion_manifest"]["layers"].items():
        prefix=layer_name+"."
        if first.startswith(prefix):
            local=first[len(prefix):]
            for row in layer["target_tensors"]:
                if row["name"]==local: row["shape"]=list(wrong["model_state"][first].shape)
    corruptions.append(wrong)
    for corrupted in corruptions:
        torch.save(corrupted,checkpoint)
        with pytest.raises(NativeCheckpointError):
            load_qwen_arm(make_spec(),model_config=config,cache_config=None,
                base_model_loader=lambda *_a,**_k:calls.append(1))
    assert calls == []


def test_realistic_hf_qwen35_config_normalizes_to_frozen_identity(tmp_path):
    from research.kmd2_ablation.qwen_backend import (
        NativeCheckpointError, _read_frozen_model_config,
    )
    expected = {"hidden_size":1024,"linear_num_value_heads":16,"linear_num_key_heads":16,
        "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
        "rms_norm_eps":1e-6,"rms_norm_type":"RMSNorm","dtype":"torch.bfloat16",
        "use_cache":False,"num_hidden_layers":24,"tie_word_embeddings":True,
        "rope_theta":10000000,"rope_scaling":{"mrope_interleaved":True,
            "mrope_section":[11,11,10],"rope_type":"default"},
        "max_position_embeddings":262144,"partial_rotary_factor":0.25}
    text = {key:value for key,value in expected.items()
            if key not in {"dtype","rms_norm_type","rope_theta","rope_scaling",
                           "partial_rotary_factor"}}
    text["dtype"]="bfloat16"; text["use_cache"]=True
    text["model_type"]="qwen3_5_text"
    text["rope_parameters"]={**expected["rope_scaling"],
        "rope_theta":expected["rope_theta"],
        "partial_rotary_factor":expected["partial_rotary_factor"]}
    raw = {"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"],
           "text_config":text,"tie_word_embeddings":True}
    model=tmp_path/"qwen"; model.mkdir(); (model/"config.json").write_text(json.dumps(raw),encoding="utf-8")
    assert _read_frozen_model_config(model,expected) == expected
    raw["text_config"]["dtype"]="float32"
    (model/"config.json").write_text(json.dumps(raw),encoding="utf-8")
    float_expected={**expected,"dtype":"torch.float32"}
    assert _read_frozen_model_config(model,float_expected)["dtype"] == "torch.float32"
    raw["model_type"]="llama"; raw["architectures"]=["LlamaForCausalLM"]
    raw["text_config"]["model_type"]="llama"
    (model/"config.json").write_text(json.dumps(raw),encoding="utf-8")
    with pytest.raises(NativeCheckpointError,match="rms_norm_type"):
        _read_frozen_model_config(model,float_expected)


def test_default_loader_extracts_official_multimodal_text_model(monkeypatch, tmp_path):
    import transformers
    from research.kmd2_ablation.qwen_backend import _default_base_model_loader

    text_config = SimpleNamespace(model_type="qwen3_5_text")
    config = SimpleNamespace(text_config=text_config)
    language_model = torch.nn.Linear(3, 3)
    lm_head = torch.nn.Linear(3, 5, bias=False)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = language_model
            self.model.visual = torch.nn.Linear(7, 7)
            self.lm_head = lm_head

    class Shell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1, device="meta")
            self.lm_head = torch.nn.Linear(1, 1, device="meta")

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: config
    )
    monkeypatch.setattr(
        transformers.AutoModelForMultimodalLM,
        "from_pretrained",
        lambda *_a, **_k: Wrapper(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_config",
        lambda *_a, **_k: Shell(),
    )

    loaded = _default_base_model_loader(tmp_path, dtype=torch.bfloat16)
    assert loaded.model is language_model
    assert loaded.lm_head is lm_head
    assert not any("visual" in name for name, _ in loaded.named_parameters())
    assert not any(parameter.is_meta for parameter in loaded.parameters())


@pytest.mark.parametrize("field,value,code", [
    ("output_width", None, "architecture_output_width_invalid"),
    ("output_width", "4", "architecture_output_width_invalid"),
    ("output_width", 3, "architecture_output_width_invalid"),
    ("r_out", None, "architecture_output_width_invalid"),
    ("r_out", True, "architecture_output_width_invalid"),
])
def test_rout_4_production_verifier_requires_exact_integer_width(field, value, code):
    from research.kmd2_ablation.qwen_backend import _verify_architecture_module
    class Widen(torch.nn.Module):
        output_width = 4
        r_out = 4
    module = Widen()
    if value is None:
        delattr(Widen, field)
    else:
        setattr(module, field, value)
    with pytest.raises(ValueError, match=code):
        _verify_architecture_module(module, Widen, 1, expected_output_width=4)


def test_rout_4_production_verifier_rejects_heterogeneous_widths():
    from research.kmd2_ablation.qwen_backend import _verify_architecture_modules
    class Widen(torch.nn.Module):
        def __init__(self, width):
            super().__init__(); self.output_width = width; self.r_out = width
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(linear_attn=Widen(4)), SimpleNamespace(linear_attn=Widen(3))]))
    with pytest.raises(ValueError, match="architecture_output_width_heterogeneous"):
        _verify_architecture_modules(model, (0, 1), Widen, 1, expected_output_width=4)


def test_rout_4_loader_rejects_bad_width_before_trainables_and_rolls_back(tmp_path, monkeypatch):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_backend import load_qwen_arm
    Model, Manager, Native, _checkpoint, _path, make_spec, config = (
        _canonical_architecture_case(tmp_path, monkeypatch)
    )
    class BadWiden(Native):
        output_width = 4
        r_out = 3
    trainables = tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS for suffix in ("q_slot_scale", "out_mix")
    ))
    spec = dataclasses.replace(
        make_spec(), architecture_arm_id="rout-4",
        architecture_registry_sha256=registry_sha256(), trainable_names=trainables,
    )
    captured = []
    def load_model(*_args, **_kwargs):
        model = Model(); captured.append(model); return model
    def factory(clone, _config):
        clone.__class__ = BadWiden
        clone.q_slot_scale = torch.nn.Parameter(torch.zeros(1))
        clone.out_mix = torch.nn.Parameter(torch.zeros(1))
        return clone
    events = []
    with pytest.raises(Exception, match="architecture_output_width_invalid"):
        load_qwen_arm(
            spec, model_config=config, cache_config=None,
            base_model_loader=load_model,
            manager_factory=lambda model, _c: Manager(model),
            architecture_factory=factory, architecture_expected_type=BadWiden,
            event=events.append,
        )
    assert "configure_trainables" not in events
    assert all(type(captured[0].model.layers[index].linear_attn) is Native
               for index in TARGET_LAYERS)


def test_rout_4_production_loader_accepts_real_converted_modules(tmp_path, monkeypatch):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_architecture import KMD2SharedQueryWideningAttn
    from research.kmd2_ablation.qwen_backend import load_qwen_arm
    Model, Manager, _Native, _checkpoint, _path, make_spec, config = (
        _canonical_architecture_case(tmp_path, monkeypatch)
    )
    trainables = tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS for suffix in ("q_slot_scale", "out_mix")
    ))
    spec = dataclasses.replace(
        make_spec(), architecture_arm_id="rout-4",
        architecture_registry_sha256=registry_sha256(), trainable_names=trainables,
    )
    loaded = load_qwen_arm(
        spec, model_config=config, cache_config=None,
        base_model_loader=lambda *_a, **_k: Model(),
        manager_factory=lambda model, _c: Manager(model),
    )
    assert loaded.trainable_names == trainables
    assert all(type(loaded.model.model.layers[index].linear_attn)
               is KMD2SharedQueryWideningAttn
               and loaded.model.model.layers[index].linear_attn.output_width == 4
               and loaded.model.model.layers[index].linear_attn.r_out == 4
               for index in TARGET_LAYERS)


@pytest.mark.parametrize(("arm_id", "rank"), [("mimo-r2", 2), ("mimo-r4", 4)])
def test_true_mimo_dispatch_contract_has_exact_rankwise_trainables(arm_id, rank):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_training import _architecture_dispatch_contract

    job = {
        "arm_id": arm_id,
        "architecture_registry_sha256": registry_sha256(),
    }
    config = {
        "architecture": {
            "arm_id": arm_id,
            "registry_sha256": registry_sha256(),
            "mimo_rank": rank,
        }
    }

    contract = _architecture_dispatch_contract(job, config)

    assert contract.arm == "native"
    assert contract.architecture_arm_id == arm_id
    assert contract.registry_sha256 == registry_sha256()
    assert contract.mimo_rank == rank
    assert contract.trainable_names == tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS
        for suffix in (
            "mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out"
        )
    ))
    assert len(contract.trainable_names) == 90
    assert not any(
        forbidden in name
        for name in contract.trainable_names
        for forbidden in ("erase_proj", "write_proj", "write_offset", "cache", "q_slot")
    )


@pytest.mark.parametrize("case", ["arm", "missing_hash", "hash", "rank", "combination"])
def test_true_mimo_dispatch_contract_rejects_invalid_identity_preconstruction(case):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        _architecture_dispatch_contract,
    )

    job = {"arm_id": "mimo-r2", "architecture_registry_sha256": registry_sha256()}
    architecture = {
        "arm_id": "mimo-r2", "registry_sha256": registry_sha256(), "mimo_rank": 2
    }
    if case == "arm":
        architecture["arm_id"] = "mimo-r4"
    elif case == "missing_hash":
        job.pop("architecture_registry_sha256")
    elif case == "hash":
        architecture["registry_sha256"] = "0" * 64
    elif case == "rank":
        architecture["mimo_rank"] = 4
    else:
        architecture["output_width"] = 4

    with pytest.raises(QwenRuntimeConfigurationError):
        _architecture_dispatch_contract(job, {"architecture": architecture})


@pytest.mark.parametrize("job_arm", ["native", "gdn2-channel-r1", "recency"])
def test_execute_rejects_mimo_config_arm_mismatch_before_any_dependency(job_arm):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError, execute_job,
    )

    job = _qwen_adapter_job("a" * 64, "b" * 64)
    job["arm_id"] = job_arm
    job["architecture_registry_sha256"] = registry_sha256()
    job["canonical_config"]["architecture"] = {
        "arm_id": "mimo-r2", "registry_sha256": registry_sha256(), "mimo_rank": 2,
    }
    calls = {"load_data": 0, "load_arm": 0}
    def touched(name):
        def fail(**_kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run")
        return fail

    with pytest.raises(QwenRuntimeConfigurationError, match="architecture_arm_mismatch"):
        execute_job(
            job, runtime={},
            dependencies={"load_data": touched("load_data"), "load_arm": touched("load_arm")},
        )
    assert calls == {"load_data": 0, "load_arm": 0}


@pytest.mark.parametrize(("arm_id", "rank"), [("mimo-r2", 2), ("mimo-r4", 4)])
def test_production_loader_overlays_native_r1_then_builds_rank_specific_true_mimo(
    tmp_path, monkeypatch, arm_id, rank
):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_architecture import KMD2TrueMIMOAttn
    from research.kmd2_ablation.qwen_backend import load_qwen_arm

    Model, Manager, _Native, _checkpoint, _path, make_spec, config = (
        _canonical_architecture_case(tmp_path, monkeypatch)
    )
    trainables = tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS
        for suffix in (
            "mimo_q_transform", "mimo_k_transform", "mimo_v", "mimo_z", "mimo_out"
        )
    ))
    spec = dataclasses.replace(
        make_spec(), architecture_arm_id=arm_id,
        architecture_registry_sha256=registry_sha256(), trainable_names=trainables,
    )
    events = []

    loaded = load_qwen_arm(
        spec, model_config=config, cache_config=None,
        base_model_loader=lambda *_a, **_k: Model(),
        manager_factory=lambda model, _c: Manager(model), event=events.append,
    )

    assert loaded.arm == "native"
    assert loaded.architecture_arm_id == arm_id
    assert loaded.architecture_registry_sha256 == registry_sha256()
    assert loaded.architecture_classification == "cold_redesign"
    assert loaded.architecture_identity_passed is False
    assert loaded.architecture_implementation == "qwen_architecture.KMD2TrueMIMOAttn.reference_fp32"
    assert loaded.architecture_tensor_manifest["mimo_rank"] == rank
    assert loaded.architecture_tensor_manifest["layer_count"] == 18
    assert len(loaded.architecture_tensor_manifest["new"]) == 18 * 5
    assert loaded.trainable_names == trainables
    assert len(loaded.trainable_names) == 90
    assert all(
        type(loaded.model.model.layers[index].linear_attn) is KMD2TrueMIMOAttn
        and loaded.model.model.layers[index].linear_attn.rank == rank
        for index in TARGET_LAYERS
    )
    assert events[:3] == ["validate_assets", "load_model", "native_install_r1"]
    assert "checkpoint_overlay_complete" in events


@pytest.mark.parametrize(("case", "code"), [
    ("missing", "native_checkpoint_tensor_missing"),
    ("unexpected", "native_checkpoint_tensor_unexpected"),
    ("wrong_layer", "native_checkpoint_target_invalid"),
    ("shape", "native_checkpoint_shape_mismatch"),
    ("dtype", "native_checkpoint_dtype_mismatch"),
])
def test_architecture_checkpoint_rejects_each_typed_contract_error_before_prepare(tmp_path, monkeypatch, case, code):
    from research.kmd2_ablation.qwen_backend import NativeCheckpointError, load_qwen_arm
    Model, Manager, Native, checkpoint, path, spec, config = _canonical_architecture_case(tmp_path, monkeypatch)
    key = f"model.layers.0.linear_attn.{_R1_SUFFIXES[0]}"
    if case == "missing": checkpoint.pop(key)
    elif case == "unexpected": checkpoint[key.replace(_R1_SUFFIXES[0], "q_slot_scale")] = torch.zeros(1)
    elif case == "wrong_layer": checkpoint[key.replace("layers.0", "layers.3")] = checkpoint.pop(key)
    elif case == "shape": checkpoint[key] = checkpoint[key][:-1]
    else: checkpoint[key] = checkpoint[key].double()
    torch.save(checkpoint, path)
    prepared = []
    with pytest.raises(NativeCheckpointError) as error:
        load_qwen_arm(spec(), model_config=config, cache_config=None, base_model_loader=lambda *_a, **_k: Model(), manager_factory=lambda model, _c: Manager(model), architecture_factory=lambda *_: prepared.append(1), architecture_expected_type=torch.nn.Module)
    assert error.value.code == code
    assert prepared == []


def test_qwen_load_spec_architecture_identity_is_atomic_and_canonical(tmp_path: Path):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec

    files = [tmp_path / name for name in ("model", "checkpoint", "data")]
    for path in files:
        path.write_bytes(b"x")
    common = dict(
        arm="native", job_id="job", model_asset=_asset("model", files[0]),
        native_checkpoint=_asset("native_checkpoint", files[1]),
        data_asset=_asset("data", files[2]), cache_resume=None,
        trainable_names=("x",), pre_replacement_checkpoint_sha256=_sha256(files[1]),
    )
    with pytest.raises(ValueError, match="architecture_identity_incomplete"):
        QwenArmLoadSpec(**common, architecture_arm_id="gdn2-channel-r1")
    spec = QwenArmLoadSpec(
        **common, architecture_arm_id="gdn2-channel-r1",
        architecture_registry_sha256=registry_sha256(),
    )
    assert spec.architecture_arm_id == "gdn2-channel-r1"


@pytest.mark.parametrize("arm", ["rot-off", "rot-constant", "rot-noncumulative", "rot-fixed-rope", "rot-moving-frame-oracle"])
def test_qwen_load_spec_accepts_exact_rotation_identity_and_explicit_diagnostic_flag(tmp_path: Path, arm: str):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec
    files = [tmp_path / name for name in ("model", "checkpoint", "data")]
    for path in files: path.write_bytes(b"x")
    spec = QwenArmLoadSpec(
        arm="native", job_id="job", model_asset=_asset("model", files[0]),
        native_checkpoint=_asset("native_checkpoint", files[1]), data_asset=_asset("data", files[2]),
        cache_resume=None, trainable_names=(), pre_replacement_checkpoint_sha256=_sha256(files[1]),
        architecture_arm_id=arm, architecture_registry_sha256=registry_sha256(),
        diagnostic_training=arm == "rot-moving-frame-oracle",
    )
    assert spec.architecture_arm_id == arm
    assert spec.diagnostic_training is (arm == "rot-moving-frame-oracle")


def test_qwen_load_spec_rejects_diagnostic_flag_for_nonmoving_and_legacy_non_native(tmp_path: Path):
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec
    files = [tmp_path / name for name in ("model", "checkpoint", "data")]
    for path in files: path.write_bytes(b"x")
    common = dict(job_id="job", model_asset=_asset("model", files[0]),
        native_checkpoint=_asset("native_checkpoint", files[1]), data_asset=_asset("data", files[2]),
        cache_resume=None, trainable_names=(), pre_replacement_checkpoint_sha256=_sha256(files[1]),
        architecture_arm_id="rot-off", architecture_registry_sha256=registry_sha256())
    with pytest.raises(ValueError, match="diagnostic_training"):
        QwenArmLoadSpec(arm="native", diagnostic_training=True, **common)
    with pytest.raises(ValueError, match="legacy_architecture_arm_mismatch"):
        QwenArmLoadSpec(arm="recency", diagnostic_training=False, **common)


def test_architecture_tensor_manifest_aggregates_and_qualifies_all_18_layers():
    from research.kmd2_ablation.qwen_backend import _aggregate_architecture_tensor_manifest
    manifest = {
        "copied": ("in_proj_qkv.weight",),
        "transformed": (("in_proj_b.weight", "erase_proj.weight", "row_copy_dk"),),
        "new": ("write_offset",),
    }
    model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    for _ in range(23):
        module = SimpleNamespace(transformation_manifest=lambda manifest=manifest: manifest)
        model.model.layers.append(SimpleNamespace(linear_attn=module))
    aggregated = _aggregate_architecture_tensor_manifest(model, TARGET_LAYERS)
    assert len(aggregated["copied"]) == 18
    assert len(aggregated["transformed"]) == 18
    assert len(aggregated["new"]) == 18
    assert aggregated["copied"][0].startswith("model.layers.0.linear_attn.")
    assert aggregated["copied"][-1].startswith("model.layers.22.linear_attn.")


@pytest.mark.parametrize("case", ["missing", "heterogeneous"])
def test_architecture_tensor_manifest_rejects_missing_or_heterogeneous_layers(case):
    from research.kmd2_ablation.qwen_backend import _aggregate_architecture_tensor_manifest
    base = {"copied": ("x",), "transformed": (("a", "b", "copy"),), "new": ()}
    model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    for index in range(23):
        if case == "missing" and index == TARGET_LAYERS[-1]:
            module = SimpleNamespace()
        else:
            value = base if not (case == "heterogeneous" and index == TARGET_LAYERS[-1]) else {**base, "copied": ("y",)}
            module = SimpleNamespace(transformation_manifest=lambda value=value: value)
        model.model.layers.append(SimpleNamespace(linear_attn=module))
    with pytest.raises(ValueError, match="architecture_tensor_manifest"):
        _aggregate_architecture_tensor_manifest(model, TARGET_LAYERS)


def test_qwen_backend_import_never_imports_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("research.kmd2_ablation.qwen_backend", None)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name == "transformers" or name.startswith("transformers."):
            raise AssertionError("qwen_backend imported Transformers eagerly")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module("research.kmd2_ablation.qwen_backend")
    assert hasattr(module, "load_qwen_arm")


def test_qwen_training_import_is_transformers_lazy_and_exposes_runner_entrypoint(
) -> None:
    import subprocess

    script = """
import sys
from importlib.abc import MetaPathFinder

class RejectTransformers(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] == 'transformers':
            raise AssertionError('qwen_training imported Transformers eagerly')
        return None

sys.meta_path.insert(0, RejectTransformers())
from research.kmd2_ablation import qwen_training
assert callable(qwen_training.run_job)
assert callable(qwen_training.build_job_dispatcher)
assert 'transformers' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.memory = torch.nn.Linear(2, 2, bias=False)


def _asset(name: str, path: Path):
    from research.kmd2_ablation.qwen_backend import ExternalAssetIdentity

    return ExternalAssetIdentity(
        name=name,
        path=path,
        kind="file",
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def test_qwen_arm_loader_validates_assets_orders_install_and_freezes_exact_names(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.json"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.jsonl"
    cache_resume = tmp_path / "resume.pt"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"examples")
    cache_resume.write_bytes(b"resume")

    events: list[object] = []
    model = _FakeQwen()

    def base_loader(path: Path, **kwargs: object) -> _FakeQwen:
        events.append(("base", path, kwargs))
        return model

    def manager_factory(received: object, config: object) -> object:
        assert received is model
        events.append(("manager", config))
        return SimpleNamespace(name="manager")

    def cache_installer(**kwargs: object) -> tuple[int, ...]:
        events.append(
            (
                "install",
                kwargs["native_checkpoint"],
                kwargs["cache_resume"],
                kwargs["expected_job_id"],
            )
        )
        assert kwargs["model"] is model
        assert getattr(kwargs["manager"], "name") == "manager"
        model.register_parameter(
            "cache_amplitude", torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        )
        return (1, 3)

    spec = QwenArmLoadSpec(
        arm="surprise",
        job_id="job-surprise",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=_asset("cache_resume", cache_resume),
        trainable_names=("memory.weight", "cache_amplitude"),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
        model_loader_kwargs={"torch_dtype": "bfloat16"},
    )
    loaded = load_qwen_arm(
        spec,
        model_config=SimpleNamespace(name="cfg"),
        cache_config=SimpleNamespace(score="exact_outer"),
        base_model_loader=base_loader,
        manager_factory=manager_factory,
        cache_installer=cache_installer,
    )

    assert [event[0] for event in events] == ["base", "manager", "install"]
    assert events[0][1] == model_asset.resolve()
    assert events[0][2] == {"torch_dtype": "bfloat16"}
    assert events[2][1:] == (
        native_checkpoint.resolve(),
        cache_resume.resolve(),
        "job-surprise",
    )
    assert loaded.model is model
    assert loaded.arm == "surprise"
    assert loaded.upgraded_indices == (1, 3)
    assert loaded.trainable_names == ("cache_amplitude", "memory.weight")
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == {
        "cache_amplitude": True,
        "backbone.bias": False,
        "backbone.weight": False,
        "memory.weight": True,
    }
    assert tuple(asset.name for asset in loaded.assets) == (
        "cache_resume",
        "data",
        "model",
        "native_checkpoint",
    )


def test_qwen_recency_arm_uses_real_default_install_and_runs_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = torch.nn.Linear(2, 2)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.model = Backbone()

    class Manager:
        def __init__(self, model: Model) -> None:
            self.model = model

        def apply_upgrade(self) -> list[int]:
            assert os.environ["GDN3_KMD2_NATIVE"] == "1"
            self.model.model.layers[0].linear_attn = KMD2NativeAttn(
                config,
                layer_idx=0,
            )
            return [0]

    model_asset = tmp_path / "model.bin"
    checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.jsonl"
    model_asset.write_bytes(b"model")
    torch.save({}, checkpoint)
    data_asset.write_bytes(b"examples")
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    model = Model()
    spec = QwenArmLoadSpec(
        arm="recency",
        job_id="job-recency-real-install",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=(
            "model.layers.0.linear_attn.cache_amplitude",
        ),
        pre_replacement_checkpoint_sha256=_sha256(checkpoint),
    )

    loaded = load_qwen_arm(
        spec,
        model_config=None,
        cache_config=CacheConfig(
            width=2,
            block_size=2,
            score="recency",
            read="rmsnorm",
            storage_dtype="fp32",
        ),
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=lambda received, _config: Manager(received),
    )

    layer = loaded.model.model.layers[0].linear_attn
    assert type(layer).__name__ == "KMD2RecencyCacheAttn"
    assert layer.cache_config.score == "recency"
    torch.manual_seed(1201)
    output = layer(torch.randn(2, 6, 12))
    assert output.shape == (2, 6, 12)
    assert bool(torch.isfinite(output).all())
    diagnostics = layer.last_cache_diagnostics
    assert diagnostics is not None
    torch.testing.assert_close(
        diagnostics.update_scores,
        torch.arange(1, 7, dtype=torch.float32).view(1, 6, 1).expand(2, 6, 2),
    )
    torch.testing.assert_close(
        diagnostics.final_selected_positions,
        torch.tensor([5, 4], dtype=torch.int64).view(1, 1, 2).expand(2, 2, 2),
    )


@pytest.mark.parametrize("arm", ["native", "recency", "surprise"])
@pytest.mark.parametrize(
    ("checkpoint_dtype", "expect_success"),
    [(torch.bfloat16, True), (torch.float32, False)],
)
def test_qwen_bfloat16_install_aligns_inherited_dtype_and_enforces_checkpoint_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    checkpoint_dtype: torch.dtype,
    expect_success: bool,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attn = torch.nn.Linear(2, 2).to(torch.bfloat16)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(13, 12).to(torch.bfloat16)
            self.model = Backbone()

    class Manager:
        def __init__(self, model: Model) -> None:
            self.model = model

        def apply_upgrade(self) -> list[int]:
            self.model.model.layers[0].linear_attn = KMD2NativeAttn(
                config, layer_idx=0
            )
            return [0]

    model_asset = tmp_path / f"{arm}-model.bin"
    checkpoint = tmp_path / f"{arm}-{checkpoint_dtype}.pt"
    data_asset = tmp_path / f"{arm}-data.jsonl"
    model_asset.write_bytes(b"model")
    torch.save(
        {
            "model.layers.0.linear_attn.in_proj_qkv.weight": torch.full(
                (22, 12), 0.125, dtype=checkpoint_dtype
            )
        },
        checkpoint,
    )
    data_asset.write_bytes(b"examples")
    model = Model()
    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    spec = QwenArmLoadSpec(
        arm=arm,
        job_id=f"job-{arm}-bf16",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=(
            f"model.layers.0.linear_attn.{('in_proj_qkv.weight' if arm == 'native' else 'cache_amplitude')}",
        ),
        pre_replacement_checkpoint_sha256=_sha256(checkpoint),
        model_loader_kwargs={"torch_dtype": torch.bfloat16},
    )
    cache_config = None
    if arm != "native":
        cache_config = CacheConfig(
            width=2,
            block_size=2,
            score="recency" if arm == "recency" else "exact_outer",
            read="rmsnorm",
            storage_dtype="fp32",
        )

    if not expect_success:
        with pytest.raises(ValueError, match="dtype"):
            load_qwen_arm(
                spec,
                model_config=None,
                cache_config=cache_config,
                base_model_loader=lambda *_args, **_kwargs: model,
                manager_factory=lambda received, _config: Manager(received),
            )
        return

    loaded = load_qwen_arm(
        spec,
        model_config=None,
        cache_config=cache_config,
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=lambda received, _config: Manager(received),
    )
    layer = loaded.model.model.layers[0].linear_attn
    cache_names = {
        "cache_gamma_q",
        "cache_gamma_k",
        "cache_sink_logit",
        "cache_amplitude",
    }
    for name, parameter in layer.named_parameters():
        expected = torch.float32 if name in cache_names else torch.bfloat16
        assert parameter.dtype == expected, name
    output = layer(torch.randn(1, 4, 12, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert bool(torch.isfinite(output.float()).all())


def test_qwen_arm_loader_rejects_asset_identity_before_loading(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        AssetIdentityError,
        ExternalAssetIdentity,
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    calls: list[str] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=ExternalAssetIdentity(
            name="model",
            path=model_asset,
            kind="file",
            size_bytes=model_asset.stat().st_size,
            sha256="0" * 64,
        ),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    with pytest.raises(AssetIdentityError, match="asset_hash_mismatch") as error:
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: calls.append("load"),
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (),
        )
    assert error.value.code == "asset_hash_mismatch"
    assert calls == []


def test_qwen_arm_loader_rejects_unknown_trainables_transactionally(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    model = _FakeQwen()
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("missing.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    with pytest.raises(KeyError, match="declared trainable"):
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: model,
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (0,),
        )
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before


def test_qwen_heal_load_spec_requires_a_native_checkpoint(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec

    model_asset = tmp_path / "model.bin"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    data_asset.write_bytes(b"data")

    with pytest.raises(ValueError, match="native_checkpoint_required"):
        QwenArmLoadSpec(
            arm="native",
            job_id="native-job",
            model_asset=_asset("model", model_asset),
            native_checkpoint=None,
            data_asset=_asset("data", data_asset),
            cache_resume=None,
            trainable_names=("memory.weight",),
            pre_replacement_checkpoint_sha256="a" * 64,
        )


def test_qwen_arm_loader_cross_checks_measured_pre_replacement_checkpoint_digest(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        AssetIdentityError,
        QwenArmLoadSpec,
        load_qwen_arm,
    )

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    calls: list[str] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("native_checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256="f" * 64,
    )

    with pytest.raises(AssetIdentityError, match="checkpoint_identity_mismatch") as error:
        load_qwen_arm(
            spec,
            model_config=object(),
            cache_config=None,
            base_model_loader=lambda *_args, **_kwargs: calls.append("load"),
            manager_factory=lambda *_args: object(),
            native_installer=lambda **_kwargs: (0,),
        )
    assert error.value.code == "checkpoint_identity_mismatch"
    assert calls == []


def test_qwen_arm_loader_uses_loaded_model_config_when_execution_passes_none(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import QwenArmLoadSpec, load_qwen_arm

    model_asset = tmp_path / "model.bin"
    native_checkpoint = tmp_path / "native.pt"
    data_asset = tmp_path / "data.bin"
    model_asset.write_bytes(b"model")
    native_checkpoint.write_bytes(b"native")
    data_asset.write_bytes(b"data")
    model = _FakeQwen()
    model.config = SimpleNamespace(name="loaded-config")
    seen: list[object] = []
    spec = QwenArmLoadSpec(
        arm="native",
        job_id="native-job",
        model_asset=_asset("model", model_asset),
        native_checkpoint=_asset("checkpoint", native_checkpoint),
        data_asset=_asset("data", data_asset),
        cache_resume=None,
        trainable_names=("memory.weight",),
        pre_replacement_checkpoint_sha256=_sha256(native_checkpoint),
    )

    def manager_factory(_model: object, config: object) -> object:
        seen.append(config)
        return object()

    def native_installer(**kwargs: object) -> tuple[int, ...]:
        seen.append(kwargs["model_config"])
        return (0,)

    load_qwen_arm(
        spec,
        model_config=None,
        cache_config=None,
        base_model_loader=lambda *_args, **_kwargs: model,
        manager_factory=manager_factory,
        native_installer=native_installer,
    )
    assert seen == [model.config, model.config]


def _heal_arm(arm: str):
    from research.kmd2_ablation.qwen_backend import QwenHealArmContract

    return QwenHealArmContract(
        arm=arm,
        job_id=f"job-{arm}",
        seed=17,
        pre_replacement_checkpoint_sha256="a" * 64,
        data_sha256="c" * 64,
        example_ids=("ruler-000", "ruler-001", "ruler-002"),
        token_budget=12_288,
        update_budget=3,
        curriculum=(64, 128, 256),
        optimizer={"name": "adamw", "lr_memory": 2.0e-5, "betas": [0.9, 0.95]},
        schedule={"name": "cosine", "warmup_updates": 1},
        stopping={"max_nonfinite": 0, "early_stopping": False},
        eval_cells=("512:4q", "16K:4q", "32K:8q"),
        cache_match=(
            None
            if arm == "native"
            else {
                "width": 64,
                "block_size": 256,
                "read": "rmsnorm",
                "read_init": "gamma_one_sink_zero_amplitude_zero",
                "storage_dtype": "bf16",
                "lr_cache": 2.0e-3,
            }
        ),
        selection_policy=(
            None if arm == "native" else "recency" if arm == "recency" else "exact_outer"
        ),
    )


def test_three_arm_pairing_is_order_invariant_and_has_independent_canonical_id() -> None:
    from research.kmd2_ablation.qwen_backend import validate_three_arm_pairing

    native = _heal_arm("native")
    recency = _heal_arm("recency")
    surprise = _heal_arm("surprise")
    paired = validate_three_arm_pairing((surprise, native, recency))
    repeated = validate_three_arm_pairing((recency, surprise, native))

    expected_payload = {
        "cache_match": dict(recency.cache_match or {}),
        "curriculum": [64, 128, 256],
        "eval_cells": ["512:4q", "16K:4q", "32K:8q"],
        "data_sha256": "c" * 64,
        "example_ids": ["ruler-000", "ruler-001", "ruler-002"],
        "optimizer": dict(native.optimizer),
        "policies": {"native": None, "recency": "recency", "surprise": "exact_outer"},
        "pre_replacement_checkpoint_sha256": "a" * 64,
        "schedule": dict(native.schedule),
        "seed": 17,
        "stopping": dict(native.stopping),
        "token_budget": 12_288,
        "update_budget": 3,
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert paired.pairing_id == expected
    assert repeated.pairing_id == expected
    assert tuple(item.arm for item in paired.arms) == ("native", "recency", "surprise")
    assert paired.canonical_bytes == repeated.canonical_bytes
    assert paired.example_ids == native.example_ids


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 18),
        ("pre_replacement_checkpoint_sha256", "b" * 64),
        ("data_sha256", "d" * 64),
        ("example_ids", ("ruler-001", "ruler-000", "ruler-002")),
        ("token_budget", 12_287),
        ("update_budget", 4),
        ("curriculum", (64, 256)),
        ("optimizer", {"name": "adamw", "lr_memory": 3.0e-5}),
        ("schedule", {"name": "constant", "warmup_updates": 1}),
        ("stopping", {"max_nonfinite": 1, "early_stopping": False}),
        ("eval_cells", ("512:4q", "32K:8q")),
    ],
)
def test_three_arm_pairing_rejects_every_shared_contract_mismatch(
    field: str, replacement: object
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    jobs = [_heal_arm("native"), _heal_arm("recency"), _heal_arm("surprise")]
    jobs[2] = dataclasses.replace(jobs[2], **{field: replacement})
    with pytest.raises(PairingContractError, match="pairing_mismatch") as error:
        validate_three_arm_pairing(tuple(jobs))
    assert error.value.code == "pairing_mismatch"
    assert field in str(error.value)


@pytest.mark.parametrize(
    "changed_cache",
    [
        {"width": 32},
        {"block_size": 128},
        {"read": "unit_l2"},
        {"read_init": "different"},
        {"storage_dtype": "fp32"},
        {"lr_cache": 1.0e-3},
    ],
)
def test_three_arm_pairing_requires_capacity_read_gate_and_budget_matched_cache(
    changed_cache: dict[str, object],
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    jobs = [_heal_arm("native"), _heal_arm("recency"), _heal_arm("surprise")]
    altered = dict(jobs[2].cache_match or {})
    altered.update(changed_cache)
    jobs[2] = dataclasses.replace(jobs[2], cache_match=altered)
    with pytest.raises(PairingContractError, match="cache_match_mismatch") as error:
        validate_three_arm_pairing(tuple(jobs))
    assert error.value.code == "cache_match_mismatch"


def test_three_arm_pairing_requires_exactly_one_preregistered_arm() -> None:
    from research.kmd2_ablation.qwen_backend import (
        PairingContractError,
        validate_three_arm_pairing,
    )

    with pytest.raises(PairingContractError, match="pairing_arm_set"):
        validate_three_arm_pairing(
            (_heal_arm("native"), _heal_arm("recency"), _heal_arm("recency"))
        )


def test_qwen_heal_causal_ce_matches_independent_shifted_fixture() -> None:
    from research.kmd2_ablation.qwen_training import (
        causal_cross_entropy,
    )

    student_logits = torch.tensor(
        [[[2.0, -1.0, 0.5], [0.2, 1.3, -0.7], [1.0, -0.5, 0.4], [0.1, 0.2, 0.3]]],
        dtype=torch.float64,
    )
    labels = torch.tensor([[0, 1, -100, 2]])
    expected_ce = F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(causal_cross_entropy(student_logits, labels), expected_ce)


def test_qwen_heal_kl_matches_canonical_full_logit_numeric_fixture() -> None:
    from research.kmd2_ablation.qwen_training import distillation_kl

    student_logits = torch.tensor(
        [
            [[2.0, -1.0, 0.5], [0.2, 1.3, -0.7]],
            [[-0.5, 0.7, 1.4], [1.1, -0.4, 0.3]],
        ],
        dtype=torch.float64,
    )
    teacher_logits = torch.tensor(
        [
            [[1.5, -0.2, 0.1], [0.8, 0.3, -0.1]],
            [[0.4, 0.1, 1.0], [0.3, 0.7, -0.4]],
        ],
        dtype=torch.float64,
    )
    temperature = 1.7
    student_log = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    expected_kl = (
        F.kl_div(
            student_log,
            teacher_log,
            reduction="batchmean",
            log_target=True,
        )
        * temperature**2
        / student_logits.shape[1]
    )

    assert torch.allclose(
        distillation_kl(student_logits, teacher_logits, temperature=temperature),
        expected_kl,
    )


def test_qwen_heal_layerwise_matches_canonical_normalized_residual_fixture() -> None:
    from research.kmd2_ablation.qwen_training import layerwise_alignment_loss

    student_hidden = (
        torch.tensor([[[99.0, -99.0], [50.0, -50.0]]], dtype=torch.float64),
        torch.tensor([[[2.0, 1.0], [6.0, 2.0]]], dtype=torch.float64),
        torch.tensor([[[1.0, 3.0], [5.0, 7.0]]], dtype=torch.float64),
    )
    teacher_hidden = (
        torch.zeros((1, 2, 2), dtype=torch.float64),
        torch.tensor([[[1.0, 1.0], [3.0, 1.0]]], dtype=torch.float64),
        torch.tensor([[[2.0, 2.0], [4.0, 8.0]]], dtype=torch.float64),
    )
    expected_layers = []
    for student, teacher in zip(student_hidden[1:], teacher_hidden[1:]):
        student = student.float()
        teacher = teacher.float()
        expected_layers.append(
            (student - teacher).square().mean()
            / teacher.square().mean().clamp_min(1.0e-8)
        )
    expected_layerwise = torch.stack(expected_layers).mean()

    assert torch.allclose(
        layerwise_alignment_loss(student_hidden, teacher_hidden),
        expected_layerwise,
    )


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    reason="CUDA BF16 is unavailable",
)
def test_qwen_heal_layerwise_cuda_bf16_rematerialization_matches_reference() -> None:
    from research.kmd2_ablation.qwen_training import layerwise_alignment_loss

    student_device = torch.device("cuda:0")
    teacher_device = (
        torch.device("cuda:1")
        if torch.cuda.device_count() > 1
        else student_device
    )
    generator = torch.Generator(device=student_device).manual_seed(73021)
    student_values = [
        torch.randn(
            1,
            1025,
            1024,
            device=student_device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(2)
    ]
    teacher_values = [
        torch.randn(
            value.shape,
            device=student_device,
            dtype=torch.bfloat16,
            generator=generator,
        ).to(teacher_device)
        for value in student_values
    ]

    actual_students = [value.clone().requires_grad_(True) for value in student_values]
    saved: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        actual = layerwise_alignment_loss(
            (student_values[0], *actual_students),
            (teacher_values[0], *teacher_values),
        )
    actual.backward()

    reference_students = [
        value.clone().requires_grad_(True) for value in student_values
    ]
    expected_layers = []
    for student, teacher in zip(reference_students, teacher_values, strict=True):
        teacher_float = teacher.to(student_device, dtype=torch.float32)
        expected_layers.append(
            (student.float() - teacher_float).square().mean()
            / teacher_float.square().mean().clamp_min(1.0e-8)
        )
    expected = torch.stack(expected_layers).mean()
    expected.backward()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=3.0e-7)
    for actual_student, reference_student in zip(
        actual_students, reference_students, strict=True
    ):
        torch.testing.assert_close(
            actual_student.grad,
            reference_student.grad,
            rtol=0.0,
            atol=3.0e-8,
        )
    assert all(
        tensor.dtype == torch.bfloat16 or tensor.numel() == 1
        for tensor in saved
    )


class _HealModel(torch.nn.Module):
    def __init__(self, *, nan_output: bool = False) -> None:
        super().__init__()
        self.memory_weight = torch.nn.Parameter(
            torch.tensor(
                [[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2], [0.2, 0.1, -0.3]],
                dtype=torch.float32,
            )
        )
        self.cache_amplitude = torch.nn.Parameter(torch.tensor([0.25]))
        self.backbone_weight = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
        self.nan_output = nan_output
        self.gradient_checkpointing_calls = 0
        self.forward_example_inputs: list[torch.Tensor] = []

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_calls += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        assert use_cache is False
        self.forward_example_inputs.append(input_ids.detach().clone())
        one_hot = F.one_hot(input_ids, num_classes=3).to(torch.float32)
        logits = one_hot @ self.memory_weight
        logits = logits + self.cache_amplitude.view(1, 1, 1) * one_hot
        if self.nan_output:
            logits = logits * torch.tensor(float("nan"))
        return SimpleNamespace(logits=logits, hidden_states=(one_hot, logits))


class _HealTeacher(torch.nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states is True
        assert use_cache is False
        one_hot = F.one_hot(input_ids, num_classes=3).to(torch.float32)
        logits = one_hot.roll(1, dims=-1) * 0.4
        return SimpleNamespace(logits=logits, hidden_states=(one_hot * 0.9, logits))


class _GuardProbeHealModel(_HealModel):
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool,
        use_cache: bool,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        return super().forward(
            input_ids,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )


def _training_config(**changes: object):
    from research.kmd2_ablation.qwen_training import QwenHealTrainingConfig

    values: dict[str, object] = {
        "objective": "language_model_heal",
        "ce_weight": 1.0,
        "kl_weight": 0.2,
        "layerwise_weight": 0.1,
        "temperature": 1.5,
        "accumulation_steps": 2,
        "max_updates": 1,
        "max_tokens": 6,
        "gradient_checkpointing": True,
    }
    values.update(changes)
    return QwenHealTrainingConfig(**values)


def _batch(example_id: str, tokens: tuple[int, int, int]) -> dict[str, object]:
    input_ids = torch.tensor([tokens], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": (example_id,),
    }


def _optimizer_and_scheduler(model: _HealModel):
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("memory_weight",),
        cache_parameter_names=("cache_amplitude",),
        learning_rate=0.05,
        lr_cache=0.1,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
    )
    return optimizer, scheduler


def test_qwen_heal_optimizer_groups_are_exact_named_and_zero_decay_cache() -> None:
    from research.kmd2_ablation.qwen_training import (
        build_qwen_heal_optimizer,
        project_cache_amplitudes_,
    )

    model = _HealModel()
    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("memory_weight",),
        cache_parameter_names=("cache_amplitude",),
        learning_rate=2.0e-5,
        lr_cache=2.0e-3,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    assert [group["name"] for group in optimizer.param_groups] == ["memory", "cache"]
    assert [group["parameter_names"] for group in optimizer.param_groups] == [
        ("memory_weight",),
        ("cache_amplitude",),
    ]
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert optimizer.param_groups[1]["lr"] == 2.0e-3
    assert optimizer.defaults["fused"] is None
    with torch.no_grad():
        model.cache_amplitude.fill_(1.7)
    projected = project_cache_amplitudes_(model)
    assert projected == ("cache_amplitude",)
    assert model.cache_amplitude.item() == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen_heal_optimizer_uses_fused_adamw_for_cuda_parameters() -> None:
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    model = _HealModel().to("cuda")
    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("memory_weight",),
        cache_parameter_names=("cache_amplitude",),
        learning_rate=2.0e-5,
        lr_cache=2.0e-3,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    assert optimizer.defaults["fused"] is True
    assert all(group["fused"] is True for group in optimizer.param_groups)
    assert optimizer._gdnx_cpu_state_offload is False
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_large_adam_state_phase_offload_preserves_exact_updates() -> None:
    from research.kmd2_ablation.qwen_training import (
        _move_optimizer_state_,
        _optimizer_state_is_offloaded,
    )

    generator = torch.Generator(device="cuda").manual_seed(718)
    resident_parameter = torch.nn.Parameter(
        torch.randn(257, device="cuda", dtype=torch.bfloat16, generator=generator)
    )
    offloaded_parameter = torch.nn.Parameter(resident_parameter.detach().clone())
    resident = torch.optim.AdamW(
        (resident_parameter,), lr=1.0e-3, betas=(0.9, 0.95), fused=True
    )
    offloaded = torch.optim.AdamW(
        (offloaded_parameter,), lr=1.0e-3, betas=(0.9, 0.95), fused=True
    )
    offloaded._gdnx_cpu_state_offload = True

    for _ in range(3):
        gradient = torch.randn(
            resident_parameter.shape,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        resident_parameter.grad = gradient.clone()
        offloaded_parameter.grad = gradient.clone()
        resident.step()
        _move_optimizer_state_(offloaded, to_parameter_devices=True)
        offloaded.step()
        _move_optimizer_state_(offloaded, to_parameter_devices=False)
        assert _optimizer_state_is_offloaded(offloaded)
        assert torch.equal(resident_parameter, offloaded_parameter)


def test_qwen_heal_one_update_accumulates_fixed_windows_projects_and_logs() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    torch.manual_seed(123)
    model = _HealModel()
    teacher = _HealTeacher()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = model.memory_weight.detach().clone()
    log = trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    )

    assert trainer.step == 1
    assert trainer.tokens_seen == 6
    assert trainer.example_cursor == 2
    assert model.gradient_checkpointing_calls == 1
    assert not torch.equal(model.memory_weight, before)
    assert 0.0 <= model.cache_amplitude.item() <= 1.0
    assert scheduler.last_epoch == 1
    record = log.as_dict()
    assert record["job_id"] == "job-surprise"
    assert record["pairing_id"] == "f" * 64
    assert record["arm"] == "surprise"
    assert record["update"] == 1
    assert record["tokens_seen"] == 6
    assert record["example_ids"] == ["e0", "e1"]
    assert record["microbatches"] == 2
    assert record["skipped_steps"] == 0
    assert set(record["losses"]) == {"total", "ce", "kl", "layerwise"}
    assert all(torch.isfinite(torch.tensor(value)) for value in record["losses"].values())
    assert record["learning_rates"] == {
        "cache": pytest.approx(0.05),
        "memory": pytest.approx(0.025),
    }

    with pytest.raises(RuntimeError, match="update_budget_exhausted"):
        trainer.train_update(
            (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
        )


@pytest.mark.parametrize(
    ("extra_inputs", "expected_code"),
    [
        ({"attention_mask": torch.tensor([[1, 1, 0]])}, "padding_unsupported"),
        ({"position_ids": torch.tensor([[0, 1, 0]])}, "position_reset"),
    ],
)
def test_qwen_heal_trainer_guards_padding_and_position_resets_before_forward(
    extra_inputs: dict[str, torch.Tensor], expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_exact_cache import FullRecomputeCallError
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    model = _GuardProbeHealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(
            objective="synthetic_only",
            kl_weight=0.0,
            layerwise_weight=0.0,
            accumulation_steps=1,
            max_tokens=3,
            gradient_checkpointing=False,
        ),
        job_id="guarded-train",
        pairing_id="a" * 64,
        arm="surprise",
        expected_example_windows=(("e0",),),
    )
    batch = _batch("e0", (0, 1, 2))
    batch.update(extra_inputs)

    with pytest.raises(FullRecomputeCallError) as caught:
        trainer.train_update((batch,))

    assert caught.value.code == expected_code
    assert model.forward_example_inputs == []
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0


def test_qwen_heal_routes_teacher_inputs_to_the_explicit_teacher_device() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    class RecordingTeacher(_HealTeacher):
        def __init__(self) -> None:
            super().__init__()
            self.input_devices: list[torch.device] = []

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            self.input_devices.append(input_ids.device)
            return super().forward(input_ids, **kwargs)

    model = _HealModel()
    teacher = RecordingTeacher()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
        teacher_device=torch.device("cpu"),
    )
    trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    )
    assert teacher.input_devices == [torch.device("cpu"), torch.device("cpu")]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
def test_qwen_heal_queues_cross_device_teacher_before_student() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    calls: list[tuple[str, torch.device]] = []

    class RecordingStudent(_HealModel):
        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            calls.append(("student", input_ids.device))
            return super().forward(input_ids, **kwargs)

    class RecordingTeacher(_HealTeacher):
        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            calls.append(("teacher", input_ids.device))
            return super().forward(input_ids, **kwargs)

    model = RecordingStudent().to("cuda:0")
    teacher = RecordingTeacher().to("cuda:1")
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=teacher,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(accumulation_steps=1, max_tokens=3),
        job_id="cross-device-teacher",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",),),
        teacher_device=torch.device("cuda:1"),
    )
    batch = _batch("e0", (0, 1, 2))
    batch = {
        name: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }

    trainer.train_update((batch,))

    assert calls == [
        ("teacher", torch.device("cuda:1")),
        ("student", torch.device("cuda:0")),
    ]


def test_qwen_heal_rejects_mismatched_window_before_forward_or_mutation() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer, QwenTrainingError

    model = _HealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=_HealTeacher(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-recency",
        pairing_id="f" * 64,
        arm="recency",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenTrainingError, match="example_window_mismatch") as error:
        trainer.train_update(
            (_batch("e1", (2, 1, 0)), _batch("e0", (0, 1, 2)))
        )
    assert error.value.code == "example_window_mismatch"
    assert model.forward_example_inputs == []
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0


def test_qwen_heal_nonfinite_loss_is_a_skipped_failure_without_optimizer_step() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer, QwenTrainingError

    model = _HealModel(nan_output=True)
    optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=_HealTeacher(),
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(),
        job_id="job-surprise",
        pairing_id="f" * 64,
        arm="surprise",
        expected_example_windows=(("e0",), ("e1",)),
    )
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenTrainingError, match="nonfinite_loss") as error:
        trainer.train_update(
            (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
        )
    assert error.value.code == "nonfinite_loss"
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0
    assert trainer.skipped_steps == 1
    assert scheduler.last_epoch == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


@pytest.mark.parametrize(
    ("interruption_type", "interruption_value"),
    [(KeyboardInterrupt, "scheduler interrupted"), (SystemExit, 37)],
)
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is unavailable"
        ),
    ),
])
def test_qwen_heal_base_exception_after_optimizer_step_rolls_back_exactly(
    interruption_type: type[BaseException], interruption_value: object, device: str,
) -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    model = _HealModel().to(device)
    optimizer, _ = _optimizer_and_scheduler(model)
    interruption = interruption_type(interruption_value)

    class InterruptingScheduler:
        def __init__(self) -> None:
            self.optimizer = optimizer
            self.progress = 0

        def state_dict(self) -> dict[str, int]:
            return {"progress": self.progress}

        def load_state_dict(self, state: dict[str, int]) -> None:
            self.progress = state["progress"]

        def step(self) -> None:
            self.progress = 1
            raise interruption

    scheduler = InterruptingScheduler()
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=_training_config(
            objective="synthetic_only",
            kl_weight=0.0,
            layerwise_weight=0.0,
            accumulation_steps=1,
            max_tokens=3,
            gradient_checkpointing=False,
        ),
        job_id="transactional-train",
        pairing_id="b" * 64,
        arm="surprise",
        expected_example_windows=(("e0",),),
    )
    parameter_snapshot = copy.deepcopy(model.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())

    batch = _batch("e0", (0, 1, 2))
    batch = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    with pytest.raises(interruption_type) as caught:
        trainer.train_update((batch,))

    assert caught.value is interruption
    _assert_nested_equal(model.state_dict(), parameter_snapshot)
    _assert_nested_equal(optimizer.state_dict(), optimizer_snapshot)
    _assert_nested_equal(scheduler.state_dict(), scheduler_snapshot)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0
    assert trainer.skipped_steps == 1


def test_qwen_heal_teacher_is_required_except_explicit_synthetic_only() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenHealTrainer,
        TeacherRequiredError,
        validate_teacher_requirement,
    )

    ordinary = _training_config()
    with pytest.raises(TeacherRequiredError, match="teacher_required") as preflight:
        validate_teacher_requirement(ordinary, teacher_present=False, phase="preflight")
    assert preflight.value.code == "teacher_required"

    model = _HealModel()
    optimizer, scheduler = _optimizer_and_scheduler(model)
    with pytest.raises(TeacherRequiredError, match="teacher_required") as runtime:
        QwenHealTrainer(
            model=model,
            teacher=None,
            optimizer=optimizer,
            scheduler=scheduler,
            config=ordinary,
            job_id="job-native",
            pairing_id="f" * 64,
            arm="native",
            expected_example_windows=(("e0",), ("e1",)),
        )
    assert runtime.value.code == "teacher_required"

    synthetic = _training_config(
        objective="synthetic_only", kl_weight=0.0, layerwise_weight=0.0
    )
    validate_teacher_requirement(synthetic, teacher_present=False, phase="preflight")
    synthetic_optimizer, synthetic_scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model,
        teacher=None,
        optimizer=synthetic_optimizer,
        scheduler=synthetic_scheduler,
        config=synthetic,
        job_id="job-native",
        pairing_id="f" * 64,
        arm="native",
        expected_example_windows=(("e0",), ("e1",)),
    )
    assert trainer.train_update(
        (_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0)))
    ).as_dict()["losses"]["kl"] == 0.0


class _CheckpointLayer(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.memory = torch.nn.Parameter(
            torch.tensor([[offset, offset + 1.0], [offset + 2.0, offset + 3.0]])
        )
        self.cache_amplitude = torch.nn.Parameter(torch.tensor([0.2 + offset / 10.0]))
        self.register_buffer("native_buffer", torch.tensor([int(offset)], dtype=torch.int64))


class _CheckpointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.layer0 = _CheckpointLayer(0.0)
        self.layer1 = _CheckpointLayer(1.0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)


def test_hybrid_save_load_resume_preserves_cache_history(tmp_path, monkeypatch):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_checkpoint import (
        load_hybrid_resume_checkpoint, save_hybrid_resume_checkpoint,
    )
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    hidden = torch.randn(1, 3, 12)
    _, cache = module.scan(hidden)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    identity = {"architecture": "gdn2-mimo-r4-braid-shared-hola-w64", "hash": "a" * 64}
    path = save_hybrid_resume_checkpoint(tmp_path / "hybrid.pt", module=module,
        cache=cache, optimizer=optimizer, identity=identity)
    with torch.no_grad():
        next(module.parameters()).add_(1)
    restored = load_hybrid_resume_checkpoint(path, module=module, optimizer=optimizer,
        identity=identity)
    for name in ("state", "phase", "previous_value", "previous_write", "conv_tail", "has_history"):
        torch.testing.assert_close(getattr(restored, name), getattr(cache, name))
    for name in ("epochs", "block_epochs", "block_count", "next_position", "current_epoch",
                 "admission_count", "age_sum", "age_count"):
        torch.testing.assert_close(getattr(restored.hola_state, name), getattr(cache.hola_state, name))
    assert module.last_recurrent_cache is restored


def test_hybrid_resume_rejects_malformed_hola_before_assignment(tmp_path, monkeypatch):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError, load_hybrid_resume_checkpoint, save_hybrid_resume_checkpoint,
    )
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    _, cache = module.scan(torch.randn(1, 2, 12))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    identity = {"architecture": "shared", "hash": "a" * 64}
    path = save_hybrid_resume_checkpoint(tmp_path / "bad.pt", module=module,
        cache=cache, optimizer=optimizer, identity=identity)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["cache_state"]["hola_state"]["block_count"].fill_(257)
    torch.save(payload, path)
    sentinel = object(); module.last_recurrent_cache = sentinel
    before = {name: value.clone() for name, value in module.state_dict().items()}
    with pytest.raises(QwenCheckpointError, match="cache_state_invalid"):
        load_hybrid_resume_checkpoint(path, module=module, optimizer=optimizer,
            identity=identity)
    assert module.last_recurrent_cache is sentinel
    for name, value in module.state_dict().items(): torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("corruption", ["class", "group", "state"])
def test_hybrid_resume_rejects_optimizer_identity_matrix(tmp_path, monkeypatch, corruption):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError, load_hybrid_resume_checkpoint, save_hybrid_resume_checkpoint,
    )
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    _, cache = module.scan(torch.randn(1, 2, 12))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    identity = {"architecture": "shared", "hash": "a" * 64}
    path = save_hybrid_resume_checkpoint(tmp_path / f"optimizer-{corruption}.pt",
        module=module, cache=cache, optimizer=optimizer, identity=identity)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if corruption == "class": payload["optimizer_identity"]["class"] = "torch.optim.SGD"
    elif corruption == "group": payload["optimizer_identity"]["param_groups"][0]["hyperparameters"]["lr"] = 9.0
    else: payload["optimizer_identity"]["state_manifest"] = [{"parameter_id": 0, "slots": []}]
    torch.save(payload, path)
    with pytest.raises(QwenCheckpointError, match="optimizer_(parameter_mismatch|state_invalid)"):
        load_hybrid_resume_checkpoint(path, module=module, optimizer=optimizer, identity=identity)


@pytest.mark.parametrize("corruption", ["model_nan", "slot_nan", "slot_shape", "state_group_lr"])
def test_hybrid_resume_rejects_nonfinite_or_wrong_slots_before_mutation_and_remains_trainable(tmp_path, monkeypatch, corruption):
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError, load_hybrid_resume_checkpoint, save_hybrid_resume_checkpoint,
    )
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    _, cache = module.scan(torch.randn(1, 2, 12)); optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    sum(parameter.float().sum() for parameter in module.parameters()).backward(); optimizer.step(); optimizer.zero_grad()
    identity={"architecture":"shared","hash":"a"*64}; path=save_hybrid_resume_checkpoint(
        tmp_path/f"deep-{corruption}.pt",module=module,cache=cache,optimizer=optimizer,identity=identity)
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if corruption == "model_nan":
        first=next(iter(payload["model_state"])); payload["model_state"][first].view(-1)[0]=float("nan")
    elif corruption == "state_group_lr":
        payload["optimizer_state"]["param_groups"][0]["lr"] = 9.0
    else:
        parameter_id=next(iter(payload["optimizer_state"]["state"])); slot=payload["optimizer_state"]["state"][parameter_id]
        if corruption == "slot_nan": slot["exp_avg"].view(-1)[0]=float("inf")
        else:
            slot["exp_avg"]=slot["exp_avg"].reshape(-1)[:1]
            for row in payload["optimizer_identity"]["state_manifest"]:
                if row["parameter_id"]==parameter_id:
                    for field in row["slots"]:
                        if field["name"]=="exp_avg": field["shape"]=list(slot["exp_avg"].shape)
    torch.save(payload,path)
    model_before={name:value.clone() for name,value in module.state_dict().items()}; optimizer_before=copy.deepcopy(optimizer.state_dict())
    with pytest.raises(QwenCheckpointError):
        load_hybrid_resume_checkpoint(path,module=module,optimizer=optimizer,identity=identity)
    for name,value in module.state_dict().items(): torch.testing.assert_close(value,model_before[name])
    assert optimizer.state_dict()["param_groups"] == optimizer_before["param_groups"]
    sum(parameter.float().sum() for parameter in module.parameters()).backward(); optimizer.step()


def _checkpoint_parts(device: torch.device | str | None = None):
    from research.kmd2_ablation.qwen_checkpoint import QwenCheckpointMetadata
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    model = _CheckpointModel().to(device=device) if device is not None else _CheckpointModel()
    optimizer = build_qwen_heal_optimizer(
        model,
        memory_parameter_names=("layer0.memory", "layer1.memory"),
        cache_parameter_names=("layer0.cache_amplitude", "layer1.cache_amplitude"),
        learning_rate=0.01,
        lr_cache=0.02,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    metadata = QwenCheckpointMetadata(
        job_id="job-surprise",
        pairing_id="d" * 64,
        arm="surprise",
        step=1,
        tokens_seen=6,
        source_hashes={
            "gdn3/kmd2_native.py": "1" * 64,
            "research/kmd2_ablation/qwen_exact_cache.py": "2" * 64,
        },
        data_identity={"sha256": "3" * 64, "row_count": 3},
        example_ids=("e0", "e1", "e2"),
        promotion_config={"width": 64, "policy": "exact_outer", "min_gate_mean": 0.005},
        architecture_arm_id="exact-cache-surprise-r1",
        architecture_registry_sha256=__import__(
            "research.kmd2_ablation.architecture", fromlist=["registry_sha256"]
        ).registry_sha256(),
    )
    return model, optimizer, scheduler, metadata


def test_qwen_checkpoint_schema_three_requires_architecture_identity() -> None:
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_checkpoint import (
        QWEN_CHECKPOINT_SCHEMA_VERSION,
        QwenCheckpointMetadata,
        QwenResumeExpectation,
    )

    assert QWEN_CHECKPOINT_SCHEMA_VERSION == 3
    with pytest.raises(TypeError):
        QwenCheckpointMetadata(
            job_id="job", pairing_id="d" * 64, arm="native", step=0, tokens_seen=0,
            source_hashes={"source": "1" * 64}, data_identity={"sha256": "2" * 64},
            example_ids=("e0",), promotion_config={"policy": "none"},
        )
    metadata = QwenCheckpointMetadata(
        job_id="job", pairing_id="d" * 64, arm="native", step=0, tokens_seen=0,
        source_hashes={"source": "1" * 64}, data_identity={"sha256": "2" * 64},
        example_ids=("e0",), promotion_config={"policy": "none"},
        architecture_arm_id="kmd2-r1", architecture_registry_sha256=registry_sha256(),
    )
    expectation = QwenResumeExpectation.from_metadata(metadata)
    assert expectation.architecture_arm_id == "kmd2-r1"
    assert expectation.architecture_registry_sha256 == registry_sha256()


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_qwen_checkpoint_is_atomic_complete_and_records_exact_manifests(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import save_qwen_checkpoint

    random.seed(444)
    torch.manual_seed(555)
    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "schema_version",
        "metadata",
        "target_module_names",
        "model_state",
        "tensor_manifest",
        "optimizer_parameter_names",
        "optimizer_state",
        "scheduler_state",
        "grad_scaler_state",
        "rng_state",
        "amplitude_range",
    }
    assert payload["schema_version"] == 3
    assert payload["target_module_names"] == ["layer0", "layer1"]
    assert tuple(payload["model_state"]) == (
        "layer0.cache_amplitude",
        "layer0.memory",
        "layer0.native_buffer",
        "layer1.cache_amplitude",
        "layer1.memory",
        "layer1.native_buffer",
    )
    assert not any(name.startswith("backbone") for name in payload["model_state"])
    assert all(tensor.device.type == "cpu" for tensor in payload["model_state"].values())
    assert payload["tensor_manifest"] == [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in payload["model_state"].items()
    ]
    assert payload["optimizer_parameter_names"] == [
        ["layer0.memory", "layer1.memory"],
        ["layer0.cache_amplitude", "layer1.cache_amplitude"],
    ]
    assert payload["metadata"]["job_id"] == "job-surprise"
    assert payload["metadata"]["step"] == 1
    assert payload["metadata"]["tokens_seen"] == 6
    assert payload["amplitude_range"][0] >= 0.0
    assert payload["amplitude_range"][1] <= 1.0
    assert set(payload["rng_state"]) == {"python", "torch_cpu", "torch_cuda"}
    assert payload["grad_scaler_state"] is None
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA checkpoint portability regression")
def test_qwen_checkpoint_round_trips_cuda_optimizer_slots_through_cpu(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenResumeExpectation, load_qwen_checkpoint, save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts("cuda")
    assert optimizer.defaults["fused"] is True
    expected = copy.deepcopy(optimizer.state_dict())
    path = tmp_path / "cuda-heal.pt"
    save_qwen_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        metadata=metadata, target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_slots = payload["optimizer_state"]["state"].values()
    assert all(
        value.device.type == "cpu"
        for slot in saved_slots for value in slot.values()
        if isinstance(value, torch.Tensor)
    )
    load_qwen_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        expectation=QwenResumeExpectation.from_metadata(metadata),
        target_module_names=("layer0", "layer1"),
    )
    restored = optimizer.state_dict()
    for parameter_id, expected_slot in expected["state"].items():
        for name, expected_value in expected_slot.items():
            actual = restored["state"][parameter_id][name]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual.cpu(), expected_value.cpu())
            else:
                assert actual == expected_value
    assert any(
        value.device.type == "cuda"
        for slot in optimizer.state.values() for value in slot.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    )
    assert all(group["fused"] is True for group in optimizer.param_groups)
    sum(
        parameter.square().sum()
        for parameter in model.parameters() if parameter.requires_grad
    ).backward()
    optimizer.step()


def test_qwen_checkpoint_rejects_actual_pre_aux_v2_shape_as_unsupported_schema(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError, QwenResumeExpectation, load_qwen_checkpoint, save_qwen_checkpoint)

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    current = tmp_path / "v3.pt"
    save_qwen_checkpoint(current, model=model, optimizer=optimizer, scheduler=scheduler,
                         metadata=metadata, target_module_names=("layer0", "layer1"))
    legacy = torch.load(current, map_location="cpu", weights_only=True)
    legacy["schema_version"] = 2
    legacy.pop("grad_scaler_state")
    legacy["metadata"].pop("example_cursor")
    legacy["metadata"].pop("auxiliary_identity")
    v2 = tmp_path / "v2.pt"; torch.save(legacy, v2)
    with pytest.raises(QwenCheckpointError, match="schema version.*incompatible") as caught:
        load_qwen_checkpoint(v2, model=model, optimizer=optimizer, scheduler=scheduler,
                             expectation=QwenResumeExpectation.from_metadata(metadata),
                             target_module_names=("layer0", "layer1"))
    assert caught.value.code == "checkpoint_schema_mismatch"


def test_qwen_checkpoint_interrupted_save_preserves_destination_and_cleans_temp(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import save_qwen_checkpoint

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")

    def interrupted_save(_payload: object, temp_path: Path) -> None:
        temp_path.write_bytes(b"partial")
        raise OSError("simulated interruption")

    with pytest.raises(OSError, match="simulated interruption"):
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=interrupted_save,
        )
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


@pytest.mark.parametrize(
    ("writer_output", "expected_code"),
    [
        ("truncated", "checkpoint_decode_failed"),
        ("different_job", "resume_identity_mismatch"),
        ("different_pair", "resume_identity_mismatch"),
        ("different_model_state", "checkpoint_serialization_mismatch"),
        ("in_place_model_state", "checkpoint_serialization_mismatch"),
        ("different_optimizer_state", "checkpoint_serialization_mismatch"),
    ],
)
def test_qwen_checkpoint_save_rejects_corrupt_or_different_serialized_candidate(
    tmp_path: Path, writer_output: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    known_good = path.read_bytes()

    def corrupting_writer(payload: object, temporary: Path) -> None:
        if writer_output == "truncated":
            temporary.write_bytes(b"truncated torch payload")
            return
        assert isinstance(payload, dict)
        if writer_output == "in_place_model_state":
            payload["model_state"]["layer0.memory"].add_(0.125)
            torch.save(payload, temporary)
            return
        candidate = copy.deepcopy(payload)
        if writer_output == "different_job":
            candidate["metadata"]["job_id"] = "other-job"
        elif writer_output == "different_pair":
            candidate["metadata"]["pairing_id"] = "e" * 64
        elif writer_output == "different_model_state":
            candidate["model_state"]["layer0.memory"].add_(0.125)
        else:
            slot = next(iter(candidate["optimizer_state"]["state"].values()))
            slot["exp_avg"].add_(0.125)
        torch.save(candidate, temporary)

    with pytest.raises(QwenCheckpointError) as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=corrupting_writer,
        )
    assert caught.value.code == expected_code
    assert path.read_bytes() == known_good
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("metadata_progress", "optimizer_state_invalid"),
        ("scheduler_progress", "scheduler_state_invalid"),
    ],
)
def test_qwen_checkpoint_save_self_validates_progress_before_publish(
    tmp_path: Path, corruption: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    if corruption == "metadata_progress":
        metadata = dataclasses.replace(metadata, step=2, tokens_seen=12)
    else:
        scheduler.last_epoch = 2
        scheduler._step_count = 3
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")
    writer_calls: list[Path] = []

    def recording_save(payload: object, temporary: Path) -> None:
        writer_calls.append(temporary)
        torch.save(payload, temporary)

    with pytest.raises(QwenCheckpointError) as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
            save_function=recording_save,
        )
    assert caught.value.code == expected_code
    assert writer_calls == []
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


def test_qwen_checkpoint_save_rejects_optimizer_parameters_outside_targets_before_publish(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    path.write_bytes(b"stable-existing-checkpoint")
    writer_calls: list[Path] = []

    def recording_save(payload: object, temporary: Path) -> None:
        writer_calls.append(temporary)
        torch.save(payload, temporary)

    with pytest.raises(QwenCheckpointError, match="optimizer_target_coverage") as caught:
        save_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0",),
            save_function=recording_save,
        )
    assert caught.value.code == "optimizer_target_coverage"
    assert writer_calls == []
    assert path.read_bytes() == b"stable-existing-checkpoint"
    assert list(tmp_path.glob(".heal.pt.*.tmp")) == []


def test_qwen_checkpoint_load_rejects_optimizer_parameters_outside_targets_without_mutation(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    partial = tmp_path / "partial.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    payload["target_module_names"] = ["layer0"]
    payload["model_state"] = {
        name: tensor
        for name, tensor in payload["model_state"].items()
        if name.startswith("layer0.")
    }
    payload["tensor_manifest"] = [
        item
        for item in payload["tensor_manifest"]
        if item["name"].startswith("layer0.")
    ]
    amplitude = payload["model_state"]["layer0.cache_amplitude"]
    payload["amplitude_range"] = [float(amplitude.min()), float(amplitude.max())]
    torch.save(payload, partial)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_python_rng = random.getstate()
    before_torch_rng = torch.get_rng_state().clone()
    with pytest.raises(QwenCheckpointError, match="optimizer_target_coverage") as caught:
        load_qwen_checkpoint(
            partial,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0",),
        )
    assert caught.value.code == "optimizer_target_coverage"
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)
    assert random.getstate() == before_python_rng
    assert torch.equal(torch.get_rng_state(), before_torch_rng)


def test_qwen_checkpoint_safe_loader_rejects_pickle_execution_without_mutation(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    marker = tmp_path / "checkpoint-pickle-executed.txt"
    checkpoint = tmp_path / "malicious.pt"
    torch.save(_PickleMarkerPayload(marker, "checkpoint"), checkpoint)
    model_snapshot = copy.deepcopy(model.state_dict())
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    scheduler_snapshot = copy.deepcopy(scheduler.state_dict())

    with pytest.raises(QwenCheckpointError) as caught:
        load_qwen_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )

    assert caught.value.code == "checkpoint_decode_failed"
    assert not marker.exists()
    _assert_nested_equal(model.state_dict(), model_snapshot)
    _assert_nested_equal(optimizer.state_dict(), optimizer_snapshot)
    _assert_nested_equal(scheduler.state_dict(), scheduler_snapshot)


def test_qwen_checkpoint_resume_restores_model_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    random.seed(1234)
    torch.manual_seed(4321)
    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_model = copy.deepcopy(payload["model_state"])
    expected_optimizer = copy.deepcopy(payload["optimizer_state"])
    expected_scheduler = copy.deepcopy(payload["scheduler_state"])
    expected_python_rng = payload["rng_state"]["python"]
    expected_torch_rng = payload["rng_state"]["torch_cpu"]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(9.0)
    optimizer.param_groups[0]["lr"] = 9.0
    scheduler.last_epoch = 99
    random.seed(999)
    torch.manual_seed(999)

    resumed = load_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expectation=QwenResumeExpectation.from_metadata(metadata),
        target_module_names=("layer0", "layer1"),
    )
    assert resumed.step == 1
    assert resumed.tokens_seen == 6
    assert resumed.job_id == "job-surprise"
    selected = {name: model.state_dict()[name].cpu() for name in expected_model}
    _assert_nested_equal(selected, expected_model)
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler)
    assert random.getstate() == expected_python_rng
    assert torch.equal(torch.get_rng_state(), expected_torch_rng)


def test_qwen_checkpoint_freezes_auxiliary_identity_scaler_and_sampler_cursor(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError, QwenResumeExpectation, load_qwen_checkpoint, save_qwen_checkpoint)

    class Scaler:
        def __init__(self, scale): self.scale = scale
        def state_dict(self): return {"scale": self.scale}
        def load_state_dict(self, state): self.scale = state["scale"]

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    identity = {"lambda_spec": .1, "lambda_gate": .2, "specialization_updates": 8,
                "probe_sha256": "a" * 64,
                "coefficients": [-3 / 20 ** .5, -1 / 20 ** .5, 1 / 20 ** .5, 3 / 20 ** .5]}
    metadata = dataclasses.replace(metadata, example_cursor=2, auxiliary_identity=identity)
    scaler = Scaler(16.0); path = tmp_path / "amp.pt"
    save_qwen_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                         metadata=metadata, target_module_names=("layer0", "layer1"),
                         grad_scaler=scaler)
    scaler.scale = 2.0
    resumed = load_qwen_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                                   expectation=QwenResumeExpectation.from_metadata(metadata),
                                   target_module_names=("layer0", "layer1"), grad_scaler=scaler)
    assert scaler.scale == 16.0 and resumed.example_cursor == 2
    wrong = dataclasses.replace(metadata, auxiliary_identity={**identity, "lambda_gate": .3})
    with pytest.raises(QwenCheckpointError, match="auxiliary_identity"):
        load_qwen_checkpoint(path, model=model, optimizer=optimizer, scheduler=scheduler,
                             expectation=QwenResumeExpectation.from_metadata(wrong),
                             target_module_names=("layer0", "layer1"), grad_scaler=scaler)


@pytest.mark.parametrize(
    ("interruption_type", "interruption_value"),
    [(KeyboardInterrupt, "stop-now"), (SystemExit, 17)],
)
def test_qwen_checkpoint_resume_rolls_back_exactly_before_reraising_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_type: type[BaseException],
    interruption_value: object,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    with torch.no_grad():
        model.layer0.memory.add_(7.0)
        model.layer1.cache_amplitude.mul_(0.5)
        for slot in optimizer.state.values():
            slot["exp_avg"].add_(3.0)
    optimizer.param_groups[0]["lr"] = 0.31
    optimizer.param_groups[1]["lr"] = 0.47
    scheduler.last_epoch = 11
    scheduler._step_count = 12
    scheduler._last_lr = [0.31, 0.47]
    random.seed(9087)
    torch.manual_seed(8709)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7890)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_python_rng = random.getstate()
    before_torch_rng = torch.get_rng_state().clone()
    before_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    interruption = interruption_type(interruption_value)
    scheduler_type = type(scheduler)
    original_load_state_dict = scheduler_type.load_state_dict
    interrupted = False

    def interrupt_once(scheduler_self: object, state_dict: object) -> object:
        nonlocal interrupted
        result = original_load_state_dict(scheduler_self, state_dict)
        if scheduler_self is scheduler and not interrupted:
            interrupted = True
            random.random()
            torch.rand(4)
            if torch.cuda.is_available():
                torch.rand(4, device="cuda")
            raise interruption
        return result

    monkeypatch.setattr(scheduler_type, "load_state_dict", interrupt_once)
    with pytest.raises(interruption_type) as caught:
        load_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    assert caught.value is interruption
    assert interrupted is True
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)
    assert random.getstate() == before_python_rng
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            torch.cuda.get_rng_state_all(), before_cuda_rng, strict=True
        )
    )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("job_id", "other-job", "resume_identity_mismatch"),
        ("pairing_id", "e" * 64, "resume_identity_mismatch"),
        ("arm", "recency", "resume_identity_mismatch"),
        ("source_hashes", {"gdn3/kmd2_native.py": "4" * 64}, "resume_identity_mismatch"),
        ("data_identity", {"sha256": "5" * 64, "row_count": 3}, "resume_identity_mismatch"),
        ("example_ids", ("e1", "e0", "e2"), "resume_identity_mismatch"),
        ("promotion_config", {"width": 32, "policy": "exact_outer"}, "resume_identity_mismatch"),
        ("architecture_arm_id", "gdn2-channel-r1", "architecture_identity_mismatch"),
        ("architecture_registry_sha256", "a" * 64, "architecture_identity_mismatch"),
    ],
)
def test_qwen_checkpoint_resume_rejects_every_identity_mismatch_without_mutation(
    tmp_path: Path, field: str, replacement: object, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    path = tmp_path / "heal.pt"
    save_qwen_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    expectation = dataclasses.replace(
        QwenResumeExpectation.from_metadata(metadata), **{field: replacement}
    )
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(QwenCheckpointError, match=expected_code) as error:
        load_qwen_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=expectation,
            target_module_names=("layer0", "layer1"),
        )
    assert error.value.code == expected_code
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)


@pytest.mark.parametrize("corruption", ["missing_name", "shape", "dtype", "amplitude"])
def test_qwen_checkpoint_rejects_tensor_corruption_before_mutation(
    tmp_path: Path, corruption: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    bad = tmp_path / "bad.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    if corruption == "missing_name":
        del payload["model_state"]["layer1.memory"]
    elif corruption == "shape":
        payload["model_state"]["layer1.memory"] = torch.zeros(3, 2)
    elif corruption == "dtype":
        payload["model_state"]["layer1.memory"] = payload["model_state"][
            "layer1.memory"
        ].double()
    else:
        payload["model_state"]["layer1.cache_amplitude"].fill_(1.1)
    torch.save(payload, bad)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(QwenCheckpointError):
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    _assert_nested_equal(model.state_dict(), before)


@pytest.mark.parametrize(
    ("corruption", "expected_code"),
    [
        ("group_parameter_id", "optimizer_parameter_mismatch"),
        ("group_parameter_order", "optimizer_parameter_mismatch"),
        ("missing_slot", "optimizer_state_invalid"),
        ("foreign_slot", "optimizer_state_invalid"),
        ("moment_shape", "optimizer_state_invalid"),
        ("moment_dtype", "optimizer_state_invalid"),
        ("moment_nonfinite", "optimizer_state_invalid"),
        ("parameter_step", "optimizer_state_invalid"),
        ("group_hyperparameter", "optimizer_state_invalid"),
        ("scheduler_static", "scheduler_state_invalid"),
        ("scheduler_progress", "scheduler_state_invalid"),
        ("scheduler_group_lr", "scheduler_state_invalid"),
        ("coordinated_group_lr", "scheduler_state_invalid"),
    ],
)
def test_qwen_checkpoint_strictly_rejects_optimizer_and_scheduler_corruption(
    tmp_path: Path, corruption: str, expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    good = tmp_path / "good.pt"
    bad = tmp_path / f"{corruption}.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    optimizer_state = payload["optimizer_state"]
    groups = optimizer_state["param_groups"]
    slots = optimizer_state["state"]
    first_parameter_id = groups[0]["params"][0]
    if corruption == "group_parameter_id":
        groups[0]["params"][0] = 999
    elif corruption == "group_parameter_order":
        groups[0]["params"] = list(reversed(groups[0]["params"]))
    elif corruption == "missing_slot":
        del slots[first_parameter_id]
    elif corruption == "foreign_slot":
        slots[999] = copy.deepcopy(slots[first_parameter_id])
    elif corruption == "moment_shape":
        slots[first_parameter_id]["exp_avg"] = torch.zeros(3, 2)
    elif corruption == "moment_dtype":
        slots[first_parameter_id]["exp_avg_sq"] = slots[first_parameter_id][
            "exp_avg_sq"
        ].double()
    elif corruption == "moment_nonfinite":
        slots[first_parameter_id]["exp_avg"].fill_(float("inf"))
    elif corruption == "parameter_step":
        slots[first_parameter_id]["step"].fill_(2.0)
    elif corruption == "group_hyperparameter":
        groups[0]["betas"] = (0.5, 0.5)
    elif corruption == "scheduler_static":
        payload["scheduler_state"]["gamma"] = 0.75
    elif corruption == "scheduler_progress":
        payload["scheduler_state"]["last_epoch"] = 7
    elif corruption == "scheduler_group_lr":
        payload["scheduler_state"]["_last_lr"][0] *= 0.5
    else:
        arbitrary_rates = [0.123, 0.456]
        for group, rate in zip(groups, arbitrary_rates, strict=True):
            group["lr"] = rate
        payload["scheduler_state"]["_last_lr"] = arbitrary_rates
    torch.save(payload, bad)

    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    with pytest.raises(QwenCheckpointError) as caught:
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )
    assert caught.value.code == expected_code
    _assert_nested_equal(model.state_dict(), before_model)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)


def test_qwen_checkpoint_validates_production_lambda_schedule_from_base_lrs(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        QwenResumeExpectation,
        load_qwen_checkpoint,
        save_qwen_checkpoint,
    )

    model, optimizer, _step_scheduler, metadata = _checkpoint_parts()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
    )
    optimizer._opt_called = True
    scheduler.step()
    good = tmp_path / "good-lambda.pt"
    bad = tmp_path / "bad-lambda.pt"
    save_qwen_checkpoint(
        good,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        target_module_names=("layer0", "layer1"),
    )
    payload = torch.load(good, map_location="cpu", weights_only=True)
    arbitrary_rates = [0.321, 0.654]
    for group, rate in zip(
        payload["optimizer_state"]["param_groups"], arbitrary_rates, strict=True
    ):
        group["lr"] = rate
    payload["scheduler_state"]["_last_lr"] = arbitrary_rates
    torch.save(payload, bad)

    with pytest.raises(QwenCheckpointError, match="scheduler_state_invalid"):
        load_qwen_checkpoint(
            bad,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expectation=QwenResumeExpectation.from_metadata(metadata),
            target_module_names=("layer0", "layer1"),
        )


def test_qwen_checkpoint_save_rejects_out_of_range_amplitude(tmp_path: Path) -> None:
    from research.kmd2_ablation.qwen_checkpoint import (
        QwenCheckpointError,
        save_qwen_checkpoint,
    )

    model, optimizer, scheduler, metadata = _checkpoint_parts()
    with torch.no_grad():
        model.layer0.cache_amplitude.fill_(-0.01)
    with pytest.raises(QwenCheckpointError, match="amplitude_out_of_range"):
        save_qwen_checkpoint(
            tmp_path / "bad.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
            target_module_names=("layer0", "layer1"),
        )


def _qwen_adapter_job(
    checkpoint_sha256: str, data_sha256: str = "c" * 64
) -> dict[str, object]:
    from research.kmd2_ablation.qwen_training import derive_three_arm_pairing

    job: dict[str, object] = {
        "job_id": "job-surprise",
        "experiment_id": "experiment-qwen",
        "seed": 17,
        "stage": "qwen_heal",
        "backend": "qwen",
        "arm_id": "exact_cache.selector.exact_outer",
        "canonical_config": {
            "backend": "qwen",
            "qwen": {"run_mode": "heal"},
            "budget": {"updates": 1, "tokens": 6},
            "optimizer": {
                "name": "adamw",
                "learning_rate": 0.05,
                "betas": [0.9, 0.95],
                "eps": 1.0e-8,
                "weight_decay": 0.01,
            },
            "schedule": {"name": "cosine", "warmup_updates": 0},
            "lengths": {"curriculum": [3], "extrapolation": [3, 6]},
            "evaluation": {
                "primary_metric": "token_accuracy",
                "direction": "maximize",
            },
            "cache": {
                "width": 2,
                "block_size": 2,
                "score": "exact_outer",
                "read": "rmsnorm",
                "read_init": "gamma_one_sink_zero_amplitude_zero",
                "storage_dtype": "fp32",
                "compute_dtype": "fp32",
                "lr_cache": 0.1,
                "weight_decay_cache": 0.0,
            },
            "promotion": {"min_gate_mean": 0.005},
            "task": {
                "name": "ruler",
                "params": {
                    "objective": "language_model_heal",
                    "ce_weight": 1.0,
                    "kl_weight": 0.2,
                    "layerwise_weight": 0.1,
                    "temperature": 1.5,
                    "accumulation_steps": 2,
                    "gradient_checkpointing": True,
                    "example_ids": ["e0", "e1"],
                    "memory_parameter_names": ["memory_weight"],
                    "cache_parameter_names": ["cache_amplitude"],
                    "stopping": {
                        "max_nonfinite": 0,
                        "early_stopping": False,
                    },
                },
            },
        },
    }
    pairing = derive_three_arm_pairing(
        job,
        example_ids=("e0", "e1"),
        pre_replacement_checkpoint_sha256=checkpoint_sha256,
        data_sha256=data_sha256,
    )
    job["pairing_id"] = pairing.pairing_id
    return job


def test_stock_qwen_execute_job_returns_complete_no_training_payload(tmp_path: Path) -> None:
    from dataclasses import asdict
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import QwenJobData, derive_three_arm_pairing, execute_job
    from research.kmd2_ablation.qwen_variants import maximum_control_contract

    paths = {name: tmp_path / f"{name}.bin" for name in ("model", "checkpoint", "data", "teacher_model")}
    for name, path in paths.items(): path.write_bytes(name.encode())
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    contract = maximum_control_contract("stock-qwen")
    job["arm_id"] = "native"
    job["canonical_config"]["qwen"]["run_mode"] = "reliance"
    params = job["canonical_config"]["task"]["params"]
    serialized_contract = json.loads(json.dumps(asdict(contract)))
    params.update({"maximum_control": "stock-qwen", "maximum_contract": serialized_contract,
                   "maximum_features": serialized_contract, "maximum_contract_sha256": contract.identity_sha256})
    pairing = derive_three_arm_pairing(job, example_ids=("e0", "e1"),
        pre_replacement_checkpoint_sha256=hashes["checkpoint"], data_sha256=hashes["data"])
    job["pairing_id"] = pairing.pairing_id
    model = _HealModel()
    data = QwenJobData(train_microbatches=(_batch("e0", (0,1,2)), _batch("e1", (2,1,0))),
        eval_microbatches=(_batch("eval0", (0,2,1)),), data_identity={"sha256": hashes["data"]})
    forbidden = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("training boundary called"))
    ticks = iter((1.0, 2.0))
    payload = execute_job(job, runtime={**paths, "output": tmp_path / "out", "student_device": "cpu",
        "teacher_device": "cpu", "dtype": "float32", "asset_hashes": hashes, "resume": False},
        dependencies={"load_data": lambda **_kwargs: data,
            "load_arm": lambda spec, **_kwargs: LoadedQwenArm(model, "native", spec.job_id, (), (), ()),
            "load_teacher": forbidden, "build_optimizer": forbidden, "build_scheduler": forbidden,
            "evaluate": lambda **_kwargs: {"metrics": {"accuracy": .75},
                "recurrent_state": {"elements": 0, "bytes": 0}},
            "monotonic": lambda: next(ticks), "reset_peak_vram": lambda _device: None,
            "peak_vram_bytes": lambda _device: 0})
    assert payload["stock_evaluation"] and not payload["optimizer_created"] and not payload["architecture_replaced"]
    assert payload["loss_curves"] == {"train": [], "validation": []}
    assert payload["parameters"] == {"trainable": 0, "total": sum(p.numel() for p in model.parameters())}
    assert payload["counts"] == {"nonfinite_loss": 0, "nonfinite_gradient": 0, "skipped_steps": 0}
    assert payload["identities"]["implementation"] == "stock_qwen_source_evaluator"
    from research.kmd2_ablation.results import RESULT_SCHEMA_VERSION, canonical_json_bytes, validate_completed_run
    from research.kmd2_ablation.runner import build_completed_record
    provenance = {"schema_version": RESULT_SCHEMA_VERSION, "suite_version": "1.0.0",
        "source_hashes": {"research/kmd2_ablation/qwen_training.py": "a" * 64},
        "config_hash": hashlib.sha256(canonical_json_bytes(job["canonical_config"])).hexdigest(),
        "asset_hashes": hashes,
        "git": {"revision": "0123456789abcdef", "diff_hash": "b" * 64, "dirty": True},
        "environment": {"python": "3.13", "pytorch": "2.10", "cuda": None, "gpu": None,
                        "dependencies": {"torch": "2.10"}}}
    record = build_completed_record(job, provenance, shard_index=0, num_jobs=1,
        command=("python", "-m", "research.kmd2_ablation.run_ablation", "run"), payload=payload)
    validate_completed_run(record, job, provenance)


def _exact_cache_result_diagnostics() -> dict[str, object]:
    return {
        "width": 2,
        "block_size": 2,
        "compute_dtype": "fp32",
        "storage_dtype": "fp32",
        "coordinate_frame": "rotated_recurrence",
        "inclusive_causality": True,
        "tie_policy": "score_desc_position_desc",
        "score_definition": "exact_outer",
        "amplitude_initial": [0.25],
        "amplitude_final": [0.3],
        "selected_index_digest": "1" * 64,
        "score_digest": "2" * 64,
        "selected_index_sample": [0, 1],
        "score_statistics": {"count": 2, "min": 0.1, "max": 0.3, "mean": 0.2},
        "retention_count": 2,
        "eviction_count": 1,
        "persistent_bytes": 64,
        "block_bytes": 64,
        "persistent_hit_rate": 1.0,
        "conditional_read_accuracy": 1.0,
        "sink_mass": 0.1,
        "top1_mass": 0.9,
        "stale_occupancy": 0.0,
        "stale_error": 0.0,
        "attention_entropy": 0.2,
        "cache_output_norm": 0.4,
        "state_output_norm": 0.8,
        "implementation_paths": {
            "scan": "reference_full_recompute",
            "score": "exact_outer",
            "selection": "stable_topk",
            "read": "rmsnorm_sink",
        },
    }


def test_qwen_source_hashes_cover_the_exact_semantic_execution_graph() -> None:
    from research.kmd2_ablation.qwen_training import _source_hashes

    root = Path(__file__).resolve().parents[2]
    expected = {
        "research/kmd2_ablation/config.py",
        "research/kmd2_ablation/architecture.py",
        "research/kmd2_ablation/exact_cache.py",
        "research/kmd2_ablation/qwen_backend.py",
        "research/kmd2_ablation/qwen_architecture.py",
        "research/kmd2_ablation/qwen_checkpoint.py",
        "research/kmd2_ablation/qwen_exact_cache.py",
        "research/kmd2_ablation/qwen_fused_loss.py",
        "research/kmd2_ablation/qwen_gdn2_triton.py",
        "research/kmd2_ablation/qwen_hybrid_chunkwise.py",
        "research/kmd2_ablation/qwen_hybrid_components.py",
        "research/kmd2_ablation/qwen_hybrid_four_state.py",
        "research/kmd2_ablation/qwen_hybrid_hola.py",
        "research/kmd2_ablation/qwen_hybrid_liger_chunked.py",
        "research/kmd2_ablation/qwen_hybrid_liger_dplr.py",
        "research/kmd2_ablation/qwen_hybrid_liger_wy.py",
        "research/kmd2_ablation/qwen_hybrid_math.py",
        "research/kmd2_ablation/qwen_hybrid_shared.py",
        "research/kmd2_ablation/qwen_hybrid_triton.py",
        "research/kmd2_ablation/qwen_training.py",
        "research/kmd2_ablation/qwen_variants.py",
        "research/kmd2_ablation/results.py",
        "research/kmd2_ablation/runner.py",
        "research/kmd2_ablation/tasks/ruler.py",
        "research/kmd2_ablation/variants.py",
        "gdn3/_reference_recurrence.py",
        "gdn3/gdn3_upgrade.py",
        "gdn3/kmd2_fast_scan.py",
        "gdn3/kmd2_native.py",
    }
    actual = _source_hashes()
    assert set(actual) == expected
    for relative, digest in actual.items():
        assert digest == hashlib.sha256((root / relative).read_bytes()).hexdigest()


def test_qwen_source_hashes_change_only_ruler_digest_when_ruler_source_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    from research.kmd2_ablation.qwen_training import _source_hashes

    root = Path(__file__).resolve().parents[2]
    ruler_path = root / "research/kmd2_ablation/tasks/ruler.py"
    ruler_source = ruler_path.read_bytes()
    baseline = _source_hashes()
    original_open = Path.open

    def open_with_ruler_mutation(
        path: Path, mode: str = "r", *args: object, **kwargs: object
    ):
        if path == ruler_path and mode == "rb":
            return BytesIO(ruler_source + b"\n# semantic mutation probe\n")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_ruler_mutation)
    mutated = _source_hashes()

    assert set(mutated) == set(baseline)
    assert {
        relative for relative in baseline if mutated[relative] != baseline[relative]
    } == {"research/kmd2_ablation/tasks/ruler.py"}


@pytest.mark.parametrize("relative", [
    "research/kmd2_ablation/architecture.py",
    "research/kmd2_ablation/qwen_architecture.py",
])
def test_qwen_architecture_source_digest_changes_resume_identity(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    from io import BytesIO
    from research.kmd2_ablation.qwen_training import _source_hashes

    root = Path(__file__).resolve().parents[2]
    target = root / relative
    source = target.read_bytes()
    baseline = _source_hashes()
    original_open = Path.open
    def mutated_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == target and mode == "rb":
            return BytesIO(source + b"\n# identity mutation\n")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", mutated_open)
    changed = _source_hashes()
    assert {name for name in baseline if baseline[name] != changed[name]} == {relative}
    assert changed != baseline


class _RunnerQwenHealConfig:
    def __init__(
        self,
        canonical_config: dict[str, object],
        *,
        seeds: tuple[int, ...] = (101, 202, 303),
    ) -> None:
        self._canonical_config = canonical_config
        self.backend = "qwen"
        self.required_stage = "qwen_heal"
        self.mechanism = "exact_cache"
        self.seeds = seeds
        self.task = SimpleNamespace(params=canonical_config["task"]["params"])

    def semantic_dict(self) -> dict[str, object]:
        return copy.deepcopy(self._canonical_config)


def test_qwen_heal_runner_expands_exact_three_arm_jobs_with_strong_pairing() -> None:
    from research.kmd2_ablation.runner import _expand_jobs

    checkpoint_digest = "a" * 64
    data_digest = "c" * 64
    base_job = _qwen_adapter_job(checkpoint_digest, data_digest)
    config = _RunnerQwenHealConfig(base_job["canonical_config"])
    jobs = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": data_digest},
    )

    assert len(jobs) == 9
    assert {job["arm_id"] for job in jobs} == {"native", "recency", "surprise"}
    assert {job["seed"] for job in jobs} == {101, 202, 303}
    for seed in config.seeds:
        paired = [job for job in jobs if job["seed"] == seed]
        assert len(paired) == 3
        assert len({job["pairing_id"] for job in paired}) == 1

    changed_checkpoint = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": "b" * 64, "data": data_digest},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_checkpoint}
    )

    changed_data = _expand_jobs(
        config,
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": "d" * 64},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_data}
    )

    reordered = copy.deepcopy(base_job["canonical_config"])
    reordered["task"]["params"]["example_ids"] = ["e1", "e0"]
    changed_examples = _expand_jobs(
        _RunnerQwenHealConfig(reordered),
        "exact_cache.selector.exact_outer",
        asset_hashes={"checkpoint": checkpoint_digest, "data": data_digest},
    )
    assert {job["pairing_id"] for job in jobs}.isdisjoint(
        {job["pairing_id"] for job in changed_examples}
    )


@pytest.mark.parametrize(
    "seeds",
    [
        (101,),
        (101, 202),
        (101, 202, 303, 404),
        (101, 202, 101),
    ],
)
def test_qwen_heal_runner_requires_exactly_three_unique_seeds_before_expansion(
    seeds: tuple[int, ...],
) -> None:
    from research.kmd2_ablation.runner import PreflightCheckError, _expand_jobs

    checkpoint_digest = "a" * 64
    data_digest = "c" * 64
    config = _RunnerQwenHealConfig(
        _qwen_adapter_job(checkpoint_digest, data_digest)["canonical_config"],
        seeds=seeds,
    )
    with pytest.raises(PreflightCheckError) as caught:
        _expand_jobs(
            config,
            "exact_cache.selector.exact_outer",
            asset_hashes={
                "checkpoint": checkpoint_digest,
                "data": data_digest,
            },
        )
    assert caught.value.code == "qwen_seed_matrix_invalid"


@pytest.mark.parametrize(
    "seeds",
    [[101], [101, 202], [101, 202, 303, 404], [101, 202, 101]],
)
def test_qwen_heal_raw_preflight_reports_one_stable_seed_matrix_code(
    seeds: list[int],
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    job = _qwen_adapter_job("a" * 64)
    raw = copy.deepcopy(job["canonical_config"])
    raw.update(
        {
            "mechanism": "exact_cache",
            "variant": "top_surprise",
            "required_stage": "qwen_heal",
            "seeds": seeds,
        }
    )
    codes = validate_raw_scientific_config(raw, backend="qwen", mode="heal")
    assert codes.count("qwen_seed_matrix_invalid") == 1


@pytest.mark.parametrize(
    "params",
    [
        {"synthetic_only": True},
        {"objective": "synthetic_only", "synthetic_only": True},
    ],
)
def test_qwen_preflight_rejects_legacy_synthetic_only_boolean_declaration(
    params: dict[str, object],
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    raw = {
        "backend": "qwen",
        "mechanism": "exact_cache",
        "variant": "top_surprise",
        "required_stage": "qwen_heal",
        "seeds": [101, 202, 303],
        "qwen": {
            "run_mode": "heal",
            "streaming": False,
            "decode": False,
            "packing": False,
            "padding": "none",
            "attention_mask": "none",
        },
        "task": {"params": {**params, "example_ids": ["e0"]}},
        "cache": {"width": 2, "block_size": 2},
        "lengths": {"curriculum": [4]},
        "model": {"ffn_dim": 8, "ffn_match_lower": 8, "ffn_match_upper": 8},
    }
    codes = validate_raw_scientific_config(raw, backend="qwen", mode="heal")
    assert codes.count("qwen_synthetic_only_declaration_invalid") == 1


def test_qwen_synthetic_only_objective_is_the_only_teacher_omission_signal(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.runner import PreflightCheckError, _external_asset_paths

    options = SimpleNamespace(
        model=tmp_path / "model",
        tokenizer=None,
        checkpoint=tmp_path / "checkpoint.pt",
        data=tmp_path / "data.pt",
        teacher_model=None,
    )

    def config(params: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            backend="qwen",
            qwen=SimpleNamespace(run_mode="heal"),
            task=SimpleNamespace(params=params),
        )

    accepted = _external_asset_paths(
        options,
        config({"objective": "synthetic_only"}),
    )
    assert set(accepted) == {"model", "checkpoint", "data"}

    with pytest.raises(PreflightCheckError) as caught:
        _external_asset_paths(options, config({"synthetic_only": True}))
    assert caught.value.code == "asset_missing"
    assert "teacher_model" in str(caught.value)


@pytest.mark.parametrize(
    "example_ids",
    [None, [], ["e0", "e0"], ["e0", ""], ["e0", 7]],
)
def test_qwen_heal_preflight_requires_ordered_unique_preregistered_example_ids(
    example_ids: object,
) -> None:
    from research.kmd2_ablation.runner import validate_raw_scientific_config

    raw = {
        "backend": "qwen",
        "mechanism": "exact_cache",
        "variant": "top_surprise",
        "required_stage": "qwen_heal",
        "qwen": {"run_mode": "heal"},
        "task": {"params": {"example_ids": example_ids}},
        "cache": {"width": 2, "block_size": 2},
        "lengths": {"curriculum": [4]},
        "model": {"ffn_dim": 8, "ffn_match_lower": 8, "ffn_match_upper": 8},
    }
    assert "qwen_example_ids_invalid" in validate_raw_scientific_config(
        raw, backend="qwen", mode="heal"
    )


def test_qwen_runtime_data_windows_must_match_preregistered_example_order() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _validate_job_data,
    )

    config = _qwen_adapter_job("a" * 64)["canonical_config"]
    reordered = QwenJobData(
        train_microbatches=(
            _batch("e1", (2, 1, 0)),
            _batch("e0", (0, 1, 2)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": "c" * 64},
    )
    with pytest.raises(
        QwenRuntimeConfigurationError, match="example_window_mismatch"
    ) as caught:
        _validate_job_data(reordered, config=config)
    assert caught.value.code == "example_window_mismatch"


def test_qwen_run_job_requires_bound_runtime_but_is_runner_discoverable() -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        run_job,
    )
    from research.kmd2_ablation.runner import load_backend_dispatcher

    assert load_backend_dispatcher("qwen") is run_job
    with pytest.raises(QwenRuntimeConfigurationError, match="runtime_required") as caught:
        run_job({"backend": "qwen"})
    assert caught.value.code == "runtime_required"


def test_production_qwen_dispatch_binds_gdn2_identity_and_exact_trainables(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_training import QwenJobData, QwenRuntimeConfigurationError, execute_job

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode())
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    job["arm_id"] = "gdn2-channel-r1"
    job["canonical_config"]["architecture"] = {
        "arm_id": "gdn2-channel-r1", "registry_sha256": registry_sha256()
    }
    captured = []
    class Stop(RuntimeError): pass
    def load_arm(spec, **_kwargs):
        captured.append(spec)
        raise Stop
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)), _batch("e1", (2, 1, 0))),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"]},
    )
    runtime = {**paths, "output": tmp_path / "out", "student_device": "cpu",
        "teacher_device": "cpu", "dtype": "float32", "asset_hashes": hashes,
        "resume": False}
    with pytest.raises(Stop):
        execute_job(job, runtime=runtime, dependencies={
            "load_data": lambda **_kwargs: data, "load_arm": load_arm,
        })
    spec = captured[0]
    assert spec.arm == "native"
    assert spec.architecture_arm_id == "gdn2-channel-r1"
    assert spec.architecture_registry_sha256 == registry_sha256()
    assert spec.trainable_names == tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS
        for suffix in ("erase_proj.weight", "write_proj.weight", "write_offset")
    ))


def test_rout_4_dispatch_contract_binds_exact_identity_and_trainables() -> None:
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_training import _architecture_dispatch_contract

    digest = registry_sha256()
    contract = _architecture_dispatch_contract(
        {"arm_id": "rout-4", "architecture_registry_sha256": digest},
        {"architecture": {"arm_id": "rout-4", "registry_sha256": digest,
                          "output_width": 4}},
    )
    assert contract is not None
    assert contract.arm == "native"
    assert contract.architecture_arm_id == "rout-4"
    assert contract.registry_sha256 == digest
    assert contract.trainable_names == tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS for suffix in ("q_slot_scale", "out_mix")
    ))
    assert len(contract.trainable_names) == 36
    assert not any(token in name for name in contract.trainable_names
                   for token in ("mimo", "erase_proj", "write_proj", "cache"))


@pytest.mark.parametrize("mutation,code", [
    ({"output_width": "4"}, "architecture_width_invalid"),
    ({"output_width": 1}, "architecture_width_mismatch"),
])
def test_rout_4_dispatch_rejects_malformed_width_with_typed_error(mutation, code) -> None:
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError, _architecture_dispatch_contract,
    )
    digest = registry_sha256()
    architecture = {"arm_id": "rout-4", "registry_sha256": digest, **mutation}
    with pytest.raises(QwenRuntimeConfigurationError, match=code) as caught:
        _architecture_dispatch_contract(
            {"arm_id": "rout-4", "architecture_registry_sha256": digest},
            {"architecture": architecture},
        )
    assert caught.value.code == code


@pytest.mark.parametrize(("arm", "diagnostic", "suffixes"), [
    ("rot-off", False, ()), ("rot-constant", False, ("rotation_rate",)),
    ("rot-noncumulative", False, ("rot_proj.weight", "rot_proj.bias")),
    ("rot-fixed-rope", False, ()),
    ("rot-moving-frame-oracle", False, ()),
    ("rot-moving-frame-oracle", True, ("rot_proj.weight", "rot_proj.bias")),
])
def test_execute_job_routes_each_rotation_identity_to_load_spec(tmp_path: Path, arm, diagnostic, suffixes):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_training import QwenJobData, QwenRuntimeConfigurationError, execute_job
    paths = {name: tmp_path / f"{name}.bin" for name in ("model", "checkpoint", "data", "teacher_model")}
    for name, path in paths.items(): path.write_bytes(name.encode())
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    job["arm_id"] = arm; job["architecture_registry_sha256"] = registry_sha256()
    job["architecture_diagnostic_training"] = diagnostic
    job["canonical_config"]["architecture"] = {
        "arm_id": arm, "registry_sha256": registry_sha256(), "diagnostic_training": diagnostic,
    }
    data = QwenJobData(train_microbatches=(_batch("e0", (0,1,2)), _batch("e1", (2,1,0))),
        eval_microbatches=(_batch("eval0", (0,2,1)),), data_identity={"sha256": hashes["data"]})
    captured = []
    class Stop(RuntimeError): pass
    def load_arm(spec, **_kwargs): captured.append(spec); raise Stop
    runtime = {**paths, "output": tmp_path / "out", "student_device": "cpu", "teacher_device": "cpu",
        "dtype": "float32", "asset_hashes": hashes, "resume": False}
    with pytest.raises(Stop):
        execute_job(job, runtime=runtime, dependencies={"load_data": lambda **_k: data, "load_arm": load_arm})
    spec = captured[0]
    assert spec.arm == "native" and spec.architecture_arm_id == arm
    assert spec.architecture_registry_sha256 == registry_sha256()
    assert spec.diagnostic_training is diagnostic
    assert spec.trainable_names == tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}" for index in TARGET_LAYERS for suffix in suffixes))
    tampered = dict(job)
    tampered["architecture_diagnostic_training"] = not diagnostic
    predata_calls = []
    with pytest.raises(QwenRuntimeConfigurationError, match="architecture_diagnostic_training_mismatch"):
        execute_job(tampered, runtime=runtime, dependencies={
            "load_data": lambda **_k: predata_calls.append(True), "load_arm": load_arm,
        })
    assert predata_calls == []


@pytest.mark.parametrize("case", ["stale_hash", "arm_mismatch"])
def test_production_qwen_dispatch_rejects_invalid_submitted_architecture_before_loader(
    tmp_path: Path, case: str
) -> None:
    from research.kmd2_ablation.architecture import registry_sha256
    from research.kmd2_ablation.qwen_training import (
        QwenJobData, QwenRuntimeConfigurationError, execute_job,
    )
    paths = {name: tmp_path / f"{name}.bin" for name in ("model", "checkpoint", "data", "teacher_model")}
    for name, path in paths.items(): path.write_bytes(name.encode())
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    job["arm_id"] = "gdn2-channel-r1"
    job["canonical_config"]["architecture"] = {
        "arm_id": "gdn2-channel-r1" if case == "stale_hash" else "kmd2-r1",
        "registry_sha256": "a" * 64 if case == "stale_hash" else registry_sha256(),
    }
    data = QwenJobData(train_microbatches=(_batch("e0", (0,1,2)), _batch("e1", (2,1,0))),
        eval_microbatches=(_batch("eval0", (0,2,1)),), data_identity={"sha256": hashes["data"]})
    loader_calls = []
    runtime = {**paths, "output": tmp_path / "out", "student_device": "cpu", "teacher_device": "cpu",
        "dtype": "float32", "asset_hashes": hashes, "resume": False}
    expected = "architecture_registry_hash_mismatch" if case == "stale_hash" else "architecture_arm_mismatch"
    error_type = ValueError if case == "stale_hash" else QwenRuntimeConfigurationError
    with pytest.raises(error_type, match=expected):
        execute_job(job, runtime=runtime, dependencies={
            "load_data": lambda **_k: data,
            "load_arm": lambda *_a, **_k: loader_calls.append(True),
        })
    assert loader_calls == []


def test_gdn2_training_payload_emits_bound_manifest_and_exact_resources(tmp_path: Path):
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm, _aggregate_architecture_tensor_manifest
    from research.kmd2_ablation.qwen_training import QwenJobData, execute_job
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.erase_proj = torch.nn.Linear(1, 1, bias=False)
            self.write_proj = torch.nn.Linear(1, 1, bias=False)
            self.write_offset = torch.nn.Parameter(torch.ones(1))
            self.conv1d = torch.nn.Conv1d(1, 1, 1, bias=False)
            self.conv1d.weight.requires_grad_(False)
        def transformation_manifest(self):
            return {"copied": ("conv1d.weight",), "transformed": (
                ("in_proj_b.weight", "erase_proj.weight", "row_copy_dk"),
                ("in_proj_b.weight", "write_proj.weight", "row_copy_dv"),
                ("bw_off", "write_offset", "copy")), "new": ()}
    class Block(torch.nn.Module):
        def __init__(self, target):
            super().__init__(); self.linear_attn = Attention() if target else torch.nn.Identity()
    class Model(_HealModel):
        def __init__(self):
            super().__init__()
            self.memory_weight.requires_grad_(False); self.cache_amplitude.requires_grad_(False)
            self.model = torch.nn.Module(); self.model.layers = torch.nn.ModuleList(
                [Block(i in TARGET_LAYERS) for i in range(23)])
        def forward(self, input_ids, *, output_hidden_states, use_cache):
            one_hot = F.one_hot(input_ids, num_classes=3).float()
            scale = sum(p.sum() for n, p in self.named_parameters() if n.endswith(("erase_proj.weight", "write_proj.weight", "write_offset")))
            logits = one_hot * scale
            return SimpleNamespace(logits=logits, hidden_states=(one_hot, logits))
    paths = {name: tmp_path / f"{name}.bin" for name in ("model", "checkpoint", "data", "teacher_model")}
    for name, path in paths.items(): path.write_bytes(name.encode())
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    job["arm_id"] = "gdn2-channel-r1"
    job["canonical_config"]["architecture"] = {"arm_id": "gdn2-channel-r1", "registry_sha256": registry_sha256()}
    data = QwenJobData(train_microbatches=(_batch("e0", (0,1,2)), _batch("e1", (2,1,0))),
        eval_microbatches=(_batch("eval0", (0,2,1)),), data_identity={"sha256": hashes["data"]})
    def load_arm(spec, **_kwargs):
        model = Model(); manifest = _aggregate_architecture_tensor_manifest(model, TARGET_LAYERS)
        return LoadedQwenArm(model, "native", spec.job_id, TARGET_LAYERS, spec.trainable_names, (),
            "gdn2-channel-r1", registry_sha256(), "replacement", True,
            "qwen_architecture.KMD2ChannelwiseGDN2Attn.reference_fp32", manifest)
    runtime = {**paths, "output": tmp_path / "out", "student_device": "cpu", "teacher_device": "cpu",
        "dtype": "float32", "asset_hashes": hashes, "resume": False}
    ticks = iter((1.0, 2.0))
    payload = execute_job(job, runtime=runtime, dependencies={
        "load_data": lambda **_k: data, "load_arm": load_arm,
        "load_teacher": lambda **_k: _HealTeacher(), "save_checkpoint": lambda path, **_k: path,
        "evaluate": lambda **_k: {"metrics": {"loss": 1.0}, "recurrent_state": {"elements": 8, "bytes": 32}},
        "monotonic": lambda: next(ticks), "reset_peak_vram": lambda _d: None,
        "peak_vram_bytes": lambda _d: 0})
    assert payload["architecture_classification"] == "replacement"
    assert payload["architecture_identity_passed"] is True
    assert payload["architecture_implementation"] == "qwen_architecture.KMD2ChannelwiseGDN2Attn.reference_fp32"
    assert len(payload["architecture_tensor_manifest"]["copied"]) == 18
    assert len(payload["architecture_tensor_manifest"]["transformed"]) == 54
    assert payload["resources"] == {"total_parameters": 83, "trainable_parameters": 54,
        "recurrent_state_elements": 8, "recurrent_state_bytes": 32, "convolution_parameters": 18,
        "transformed_parameters": 54, "new_parameters": 0,
        "architecture_new_buffer_elements": 0, "architecture_new_buffer_bytes": 0,
        "reference_implementation": "qwen_architecture.KMD2ChannelwiseGDN2Attn.reference_fp32"}


@pytest.mark.parametrize(
    ("rank", "new_per_layer", "activation_elements", "activation_bytes"),
    ((2, 1_060_864, 20_480, 81_920), (4, 2_121_728, 40_960, 163_840)),
)
def test_true_mimo_resource_accounting_is_rankwise_and_state_independent(
    rank: int, new_per_layer: int, activation_elements: int, activation_bytes: int
) -> None:
    from research.kmd2_ablation.qwen_training import _true_mimo_resources

    result = _true_mimo_resources(
        rank=rank, layers=18, batch_size=1, sequence_length=1,
        heads=16, key_dim=128, value_dim=128, native_conv_parameters=7,
    )
    assert result == {
        "new_parameters_per_layer": new_per_layer,
        "new_parameters": 18 * new_per_layer,
        "recurrent_state_elements": 262_144,
        "recurrent_state_bytes": 1_048_576,
        "rankwise_live_activation_elements": activation_elements,
        "rankwise_live_activation_bytes": activation_bytes,
        "native_conv_parameters": 7,
    }


def test_true_mimo_resource_accounting_scales_state_and_activations_by_batch() -> None:
    from research.kmd2_ablation.qwen_training import _true_mimo_resources
    result = _true_mimo_resources(
        rank=2, layers=18, batch_size=3, sequence_length=1,
        heads=16, key_dim=128, value_dim=128, native_conv_parameters=7,
    )
    assert result["recurrent_state_elements"] == 786_432
    assert result["recurrent_state_bytes"] == 3_145_728
    assert result["rankwise_live_activation_elements"] == 61_440
    assert result["rankwise_live_activation_bytes"] == 245_760


def test_shared_query_widening_resources_are_independent_and_exact() -> None:
    from research.kmd2_ablation.qwen_training import _shared_query_widening_resources
    assert _shared_query_widening_resources(
        width=4, layers=18, batch_size=1, sequence_length=1,
        heads=16, key_dim=128, value_dim=128,
    ) == {
        "new_parameters_per_layer": 8_256,
        "new_parameters": 148_608,
        "recurrent_state_elements": 262_144,
        "recurrent_state_bytes": 1_048_576,
        "total_rank_read_elements": 8_192,
        "total_q_slot_elements": 8_192,
        "extra_vs_r1_elements": 12_288,
        "extra_vs_r1_bytes": 49_152,
    }


def test_shared_query_widening_resources_scale_batch_and_reject_bad_dimensions() -> None:
    from research.kmd2_ablation.qwen_training import _shared_query_widening_resources
    result = _shared_query_widening_resources(
        width=4, layers=18, batch_size=3, sequence_length=2,
        heads=16, key_dim=128, value_dim=128,
    )
    assert result["recurrent_state_elements"] == 786_432
    assert result["total_rank_read_elements"] == 49_152
    assert result["total_q_slot_elements"] == 49_152
    assert result["extra_vs_r1_elements"] == 73_728
    with pytest.raises(ValueError, match="shared_query_widening_resource_dimension_invalid"):
        _shared_query_widening_resources(
            width=4, layers=18, batch_size=True, sequence_length=1,
            heads=16, key_dim=128, value_dim=128,
        )


def test_widening_resource_shape_selection_uses_independent_batch_and_token_maxima():
    from research.kmd2_ablation.qwen_training import (
        _shared_query_widening_resources, _widening_resource_shape_maxima,
    )
    assert _widening_resource_shape_maxima(((8, 1), (1, 16))) == (8, 16)
    result = _shared_query_widening_resources(
        width=4, layers=18, batch_size=8, sequence_length=1, max_tokens=16,
        heads=16, key_dim=128, value_dim=128,
    )
    assert result["recurrent_state_elements"] == 8 * 16 * 128 * 128
    assert result["total_rank_read_elements"] == 16 * 16 * 4 * 128


@pytest.mark.parametrize(("mode", "new_parameters", "buffer_elements", "buffer_bytes"), [
    ("off", 0, 0, 0), ("constant", 18 * 4, 0, 0),
    ("noncumulative", 0, 0, 0), ("fixed-rope", 0, 18 * 2, 18 * 2 * 4),
    ("moving-frame-oracle", 0, 0, 0),
])
def test_rotation_resources_derive_new_parameters_and_buffers_from_manifest(
    mode, new_parameters, buffer_elements, buffer_bytes
):
    from research.kmd2_ablation.qwen_training import _architecture_new_state_resources
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            new = ()
            if mode == "constant":
                self.rotation_rate = torch.nn.Parameter(torch.ones(2, 2)); new = ("rotation_rate",)
            elif mode == "fixed-rope":
                self.register_buffer("inv_freq", torch.ones(2)); new = ("inv_freq",)
            self._new = new
        def transformation_manifest(self):
            return {"copied": (), "transformed": (), "new": self._new}
    result = _architecture_new_state_resources(tuple(Module() for _ in range(18)))
    assert result == {
        "new_parameters": new_parameters,
        "architecture_new_buffer_elements": buffer_elements,
        "architecture_new_buffer_bytes": buffer_bytes,
    }


@pytest.mark.parametrize("field,value", [
    ("copied", None), ("copied", "weight"), ("copied", {"weight": 1}),
    ("copied", ("",)), ("new", (1,)),
    ("transformed", (("source", "target"),)),
    ("transformed", (("source", "target", 1),)),
    ("transformed", (("source", "", "copy"),)),
])
def test_architecture_tensor_manifest_rejects_malformed_schema_with_typed_code(
    field: str, value: object
) -> None:
    from research.kmd2_ablation.qwen_backend import (
        ArchitectureManifestError, _aggregate_architecture_tensor_manifest,
    )
    manifest = {"copied": ("x",), "transformed": (), "new": ("y",)}
    manifest[field] = value
    module = SimpleNamespace(transformation_manifest=lambda: manifest, rank=2)
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(linear_attn=module)]))
    with pytest.raises(ArchitectureManifestError) as caught:
        _aggregate_architecture_tensor_manifest(model, (0,))
    assert caught.value.code == "architecture_tensor_manifest_invalid"


def test_true_mimo_manifest_is_layer_qualified_and_rejects_rank_heterogeneity() -> None:
    from research.kmd2_ablation.qwen_backend import _aggregate_architecture_tensor_manifest

    class Attention(torch.nn.Module):
        def __init__(self, rank: int):
            super().__init__(); self.rank = rank
        def transformation_manifest(self):
            return {"copied": ("conv1d.weight",), "transformed": (), "new": ("mimo_v",)}
    class Block(torch.nn.Module):
        def __init__(self, rank: int):
            super().__init__(); self.linear_attn = Attention(rank)
    model = torch.nn.Module(); model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([Block(2), Block(2)])
    manifest = _aggregate_architecture_tensor_manifest(model, (0, 1))
    assert manifest["mimo_rank"] == 2
    assert manifest["layer_count"] == 2
    assert manifest["copied"] == (
        "model.layers.0.linear_attn.conv1d.weight",
        "model.layers.1.linear_attn.conv1d.weight",
    )
    model.model.layers[1].linear_attn.rank = 4
    with pytest.raises(ValueError, match="architecture_tensor_manifest_heterogeneous_rank"):
        _aggregate_architecture_tensor_manifest(model, (0, 1))


def test_bound_qwen_dispatcher_orchestrates_heal_resume_checkpoint_and_diagnostics(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        build_job_dispatcher,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    from research.kmd2_ablation.runner import _expand_jobs

    base_job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    expanded = _expand_jobs(
        _RunnerQwenHealConfig(base_job["canonical_config"], seeds=(17, 19, 23)),
        "exact_cache.selector.exact_outer",
        asset_hashes={
            "checkpoint": hashes["checkpoint"],
            "data": hashes["data"],
        },
    )
    job = next(
        item
        for item in expanded
        if item["arm_id"] == "surprise" and item["seed"] == 17
    )
    job_before = copy.deepcopy(job)
    runtime = {
        **paths,
        "output": tmp_path / "results",
        "student_device": "cpu",
        "teacher_device": "cpu",
        "dtype": "float32",
        "checkpoint_every": 8,
        "asset_hashes": hashes,
        "resume": True,
    }
    resume_path = (
        runtime["output"] / "checkpoints" / job["job_id"] / "latest.pt"
    )
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"resume checkpoint marker")
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"], "example_count": 2},
    )
    events: list[object] = []
    saved_metadata: list[object] = []
    checkpoint_scalers: list[object] = []

    class WiredScaler:
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return 1.0
        def state_dict(self): return {"scale": 1.0}
        def load_state_dict(self, state): assert state == {"scale": 1.0}
    wired_scaler = WiredScaler()

    def load_data(**kwargs: object) -> QwenJobData:
        events.append(("data", kwargs["asset"].sha256))
        return data

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        events.append(
            (
                "arm",
                spec.arm,
                spec.pre_replacement_checkpoint_sha256,
                spec.trainable_names,
            )
        )
        model = _HealModel()
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(spec.trainable_names)),
            assets=(),
        )

    def load_teacher(**kwargs: object) -> torch.nn.Module:
        events.append(("teacher", kwargs["asset"].sha256))
        return _HealTeacher()

    def save_checkpoint(path: Path, **kwargs: object) -> Path:
        checkpoint_scalers.append(kwargs["grad_scaler"])
        saved_metadata.append(kwargs["metadata"])
        events.append(
            (
                "checkpoint",
                path,
                kwargs["metadata"].step,
                kwargs["metadata"].tokens_seen,
            )
        )
        return path

    def load_checkpoint(path: Path, **_kwargs: object) -> SimpleNamespace:
        checkpoint_scalers.append(_kwargs["grad_scaler"])
        events.append(("resume", path))
        return SimpleNamespace(
            job_id=job["job_id"],
            pairing_id=job["pairing_id"],
            arm=job["arm_id"],
            step=0,
            tokens_seen=0,
        )

    def evaluate(**kwargs: object) -> dict[str, object]:
        events.append(("evaluate", kwargs["loaded_arm"].arm))
        return {
            "metrics": {"token_accuracy": 1.0, "eval_loss": 0.25},
            "recurrent_state": {"elements": 9, "bytes": 36},
            "exact_cache": _exact_cache_result_diagnostics(),
        }

    def reset_peak_vram(device: str) -> None:
        events.append(("reset_peak_vram", device))

    ticks = iter((10.0, 12.0))
    dispatcher = build_job_dispatcher(
        runtime,
        dependencies={
            "load_data": load_data,
            "load_arm": load_arm,
            "load_teacher": load_teacher,
            "load_checkpoint": load_checkpoint,
            "save_checkpoint": save_checkpoint,
            "evaluate": evaluate,
            "monotonic": lambda: next(ticks),
            "reset_peak_vram": reset_peak_vram,
            "peak_vram_bytes": lambda _device: 0,
            "build_grad_scaler": lambda **kwargs: wired_scaler,
        },
    )
    payload = dispatcher(job)

    assert job == job_before
    assert [event[0] for event in events] == [
        "data",
        "arm",
        "teacher",
        "resume",
        "reset_peak_vram",
        "checkpoint",
        "evaluate",
    ]
    assert events[3] == ("resume", resume_path)
    assert checkpoint_scalers == [wired_scaler, wired_scaler]
    assert events[4] == ("reset_peak_vram", "cpu")
    assert events[1] == (
        "arm",
        "surprise",
        hashes["checkpoint"],
        ("memory_weight", "cache_amplitude"),
    )
    assert payload["metrics"] == {"token_accuracy": 1.0, "eval_loss": 0.25}
    assert payload["counts"] == {
        "nonfinite_loss": 0,
        "nonfinite_gradient": 0,
        "skipped_steps": 0,
    }
    assert len(payload["loss_curves"]["total"]) == 1
    assert payload["parameters"] == {"trainable": 10, "total": 11}
    assert payload["recurrent_state"] == {"elements": 9, "bytes": 36}
    assert payload["performance"]["wall_time_seconds"] == 2.0
    assert payload["performance"]["tokens_per_second"] == 3.0
    assert payload["identities"]["checkpoint"]["sha256"] == hashes["checkpoint"]
    assert payload["identities"]["data"]["sha256"] == hashes["data"]
    assert payload["identities"]["paired_starts"] == {
        "native": hashes["checkpoint"],
        "recency": hashes["checkpoint"],
        "surprise": hashes["checkpoint"],
    }
    assert saved_metadata[0].source_hashes["asset:model"] == hashes["model"]
    assert saved_metadata[0].source_hashes["asset:checkpoint"] == hashes["checkpoint"]
    assert saved_metadata[0].source_hashes["asset:data"] == hashes["data"]
    assert saved_metadata[0].source_hashes["asset:teacher_model"] == hashes[
        "teacher_model"
    ]
    assert payload["exact_cache"] == _exact_cache_result_diagnostics()
    assert "runtime" not in payload
    assert str(tmp_path) not in json.dumps(payload)


def test_qwen_dispatcher_scopes_paired_python_and_torch_rng_on_success_and_interrupt(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        build_job_dispatcher,
        derive_three_arm_pairing,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.pt",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    runtime = {
        **paths,
        "output": tmp_path / "results",
        "student_device": "cpu",
        "dtype": "float32",
        "asset_hashes": hashes,
        "resume": False,
    }
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"]},
    )

    def job_for(arm: str, seed: int) -> dict[str, object]:
        job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
        job["job_id"] = f"rng-{arm}-{seed}"
        job["seed"] = seed
        job["arm_id"] = arm
        params = job["canonical_config"]["task"]["params"]
        params.update(
            {
                "objective": "synthetic_only",
                "kl_weight": 0.0,
                "layerwise_weight": 0.0,
            }
        )
        pairing = derive_three_arm_pairing(
            job,
            example_ids=("e0", "e1"),
            pre_replacement_checkpoint_sha256=hashes["checkpoint"],
            data_sha256=hashes["data"],
        )
        job["pairing_id"] = pairing.pairing_id
        return job

    observed: list[tuple[float, tuple[float, ...]]] = []

    def load_data(**_kwargs: object) -> QwenJobData:
        observed.append(
            (
                random.random(),
                tuple(float(value) for value in torch.rand(3).tolist()),
            )
        )
        return data

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        model = _HealModel()
        declared = set(spec.trainable_names)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in declared)
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(declared)),
            assets=(),
        )

    def evaluate(**kwargs: object) -> dict[str, object]:
        result: dict[str, object] = {
            "metrics": {"eval_loss": 0.25, "token_accuracy": 1.0},
            "recurrent_state": {"elements": 9, "bytes": 36},
        }
        if kwargs["loaded_arm"].arm != "native":
            result["exact_cache"] = _exact_cache_result_diagnostics()
        return result

    dispatcher = build_job_dispatcher(
        runtime,
        dependencies={
            "load_data": load_data,
            "load_arm": load_arm,
            "save_checkpoint": lambda path, **_kwargs: path,
            "evaluate": evaluate,
            "monotonic": lambda: 1.0,
            "peak_vram_bytes": lambda _device: 0,
        },
    )
    random.seed(9917)
    torch.manual_seed(9917)
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    for arm in ("native", "recency", "surprise"):
        dispatcher(job_for(arm, 41))
    dispatcher(job_for("native", 42))

    assert observed[0] == observed[1] == observed[2]
    assert observed[3] != observed[0]
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)

    def interrupted_load_data(**_kwargs: object) -> QwenJobData:
        random.random()
        torch.rand(2)
        raise KeyboardInterrupt("interrupt after RNG use")

    interrupted = build_job_dispatcher(
        runtime,
        dependencies={"load_data": interrupted_load_data},
    )
    python_before = random.getstate()
    torch_before = torch.random.get_rng_state().clone()
    with pytest.raises(KeyboardInterrupt, match="interrupt after RNG use"):
        interrupted(job_for("native", 43))
    assert random.getstate() == python_before
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_bound_qwen_dispatcher_validates_runtime_asset_hashes_before_loading(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import AssetIdentityError
    from research.kmd2_ablation.qwen_training import build_job_dispatcher

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    calls: list[str] = []
    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": tmp_path / "results",
            "student_device": "cpu",
            "teacher_device": "cpu",
            "dtype": "float32",
            "asset_hashes": {**hashes, "checkpoint": "0" * 64},
            "resume": False,
        },
        dependencies={"load_arm": lambda *_args, **_kwargs: calls.append("load")},
    )

    with pytest.raises(AssetIdentityError, match="asset_hash_mismatch"):
        dispatcher(job)
    assert calls == []


def test_default_qwen_pt_data_loader_rejects_pickle_execution(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        _default_load_data,
    )

    marker = tmp_path / "data-pickle-executed.txt"
    data_path = tmp_path / "malicious-windows.pt"
    torch.save(_PickleMarkerPayload(marker, "data"), data_path)
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _default_load_data(asset=asset)

    assert caught.value.code == "data_window_invalid"
    assert not marker.exists()


def test_default_qwen_data_loader_normalizes_empty_sparse_stale_positions(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import _default_load_data

    data_path = tmp_path / "empty-stale-positions.json"
    data_path.write_text(
        json.dumps(
            {
                "train": [{"example_id": "e0", "input_ids": [0, 1, 2]}],
                "eval": [
                    {
                        "example_id": "eval0",
                        "input_ids": [0, 1, 2],
                        "query_mask": [False, False, True],
                        "source_spans": [[-1, -1], [-1, -1], [0, 1]],
                        "stale_positions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    data = _default_load_data(asset=asset)

    stale_positions = data.eval_microbatches[0]["stale_positions"]
    assert isinstance(stale_positions, torch.Tensor)
    assert stale_positions.dtype == torch.int64
    assert stale_positions.shape == (0, 3)


def test_default_qwen_data_loader_materializes_compact_ruler_annotations(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import _default_load_data

    data_path = tmp_path / "compact-ruler.pt"
    torch.save(
        {
            "train": [{"example_id": "train-0", "input_ids": [0, 1, 2]}],
            "eval": [{
                "example_id": "ruler-0",
                "input_ids": torch.tensor([10, 11, 12, 13], dtype=torch.int32),
                "ruler_metadata": {
                    "answer_spans": [[2, 4]],
                    "source_spans": [[0, 1]],
                },
            }],
        },
        data_path,
    )
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    data = _default_load_data(asset=asset)

    batch = data.eval_microbatches[0]
    assert batch["input_ids"].dtype == torch.long
    assert torch.equal(
        batch["query_mask"], torch.tensor([[False, False, True, True]])
    )
    assert torch.equal(
        batch["source_spans"],
        torch.tensor([[[-1, -1], [-1, -1], [0, 1], [0, 1]]]),
    )
    assert batch["stale_positions"].shape == (0, 3)


def test_default_qwen_data_loader_selects_job_seed_from_one_immutable_bundle(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_training import _default_load_data

    data_path = tmp_path / "qwen_windows.pt"
    torch.save(
        {
            "schema_version": "2.0.0",
            "train": [{"example_id": "train-0", "input_ids": [0, 1, 2]}],
            "eval_by_seed": {
                str(seed): [{
                    "example_id": f"eval-{seed}",
                    "input_ids": [seed, 1, 2],
                    "ruler_metadata": {"seed": seed},
                }]
                for seed in (11, 29, 47)
            },
        },
        data_path,
    )
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    for seed in (11, 29, 47):
        data = _default_load_data(asset=asset, job={"seed": seed})
        assert data.train_microbatches[0]["example_ids"] == ("train-0",)
        assert data.eval_microbatches[0]["example_ids"] == (f"eval-{seed}",)
        assert data.eval_microbatches[0]["ruler_metadata"][0]["seed"] == seed
        assert data.data_identity == {
            "sha256": asset.sha256,
            "size_bytes": asset.size_bytes,
            "kind": "file",
            "example_count": 1,
            "evaluation_seed": seed,
            "available_evaluation_seeds": [11, 29, 47],
            "evaluation_example_count": 1,
        }


@pytest.mark.parametrize("job", [None, {"seed": 13}])
def test_default_qwen_seeded_data_loader_fails_closed_without_matching_partition(
    tmp_path: Path,
    job: dict[str, int] | None,
) -> None:
    from research.kmd2_ablation.qwen_training import (
        QwenRuntimeConfigurationError,
        _default_load_data,
    )

    data_path = tmp_path / "qwen_windows.pt"
    torch.save(
        {
            "train": [{"example_id": "train-0", "input_ids": [0, 1, 2]}],
            "eval_by_seed": {
                "11": [{"example_id": "eval-11", "input_ids": [0, 1, 2]}]
            },
        },
        data_path,
    )
    asset = SimpleNamespace(
        path=data_path,
        sha256=_sha256(data_path),
        size_bytes=data_path.stat().st_size,
        kind="file",
    )

    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _default_load_data(asset=asset, job=job)
    assert caught.value.code == "data_window_invalid"


def test_data_bundle_builder_partitions_evaluation_by_campaign_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.kmd2_ablation.scripts import build_data_bundle

    monkeypatch.setattr(
        build_data_bundle,
        "build_eval",
        lambda _tokenizer, *, seed, grid, free_generation_subset, evidence_scope: [{
            "seed": seed,
            "grid": grid,
            "free_generation_subset": free_generation_subset,
            "evidence_scope": evidence_scope,
        }],
    )
    grid = [(512, (1, 4), 2)]
    partitions = build_data_bundle.build_seeded_eval(
        object(), seeds=(11, 29, 47), grid=grid,
        free_generation_subset=1, evidence_scope="feasibility",
    )
    assert partitions == {
        "11": [{"seed": 11, "grid": grid, "free_generation_subset": 1,
                "evidence_scope": "feasibility"}],
        "29": [{"seed": 29, "grid": grid, "free_generation_subset": 1,
                "evidence_scope": "feasibility"}],
        "47": [{"seed": 47, "grid": grid, "free_generation_subset": 1,
                "evidence_scope": "feasibility"}],
    }


def test_data_bundle_builder_defaults_are_promotion_grade() -> None:
    from research.kmd2_ablation.scripts import build_data_bundle

    grid = build_data_bundle._parse_grid([
        f"{context}:1,4,8x64"
        for context in build_data_bundle.CANONICAL_CONTEXT_LENGTHS
    ])

    assert tuple(grid) == build_data_bundle.CANONICAL_EVAL_GRID
    assert build_data_bundle._promotion_grade_grid(
        grid, free_generation_subset=8
    )
    assert not build_data_bundle._promotion_grade_grid(
        grid, free_generation_subset=7
    )


def test_data_bundle_builder_adds_compact_deterministic_generation_subset() -> None:
    from research.kmd2_ablation.scripts import build_data_bundle

    class WhitespaceTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert add_special_tokens is False
            return {
                "input_ids": [
                    1 + sum(map(ord, token)) % 997 for token in text.split()
                ] or [1]
            }

    records = build_data_bundle.build_eval(
        WhitespaceTokenizer(),
        seed=11,
        grid=[(512, (1,), 2)],
        free_generation_subset=1,
        evidence_scope="feasibility",
    )

    teacher = [
        row for row in records
        if row["ruler_metadata"]["evaluation_mode"] == "teacher_forced"
    ]
    generated = [
        row for row in records
        if row["ruler_metadata"]["evaluation_mode"] == "free_generation"
    ]
    assert len(teacher) == 2
    assert len(generated) == 1
    assert generated[0]["example_id"].endswith("-free")
    assert generated[0]["input_ids"] is next(
        row["input_ids"] for row in teacher
        if row["ruler_metadata"]["episode_id"]
        == generated[0]["ruler_metadata"]["episode_id"]
    )
    assert all(row["input_ids"].dtype == torch.int32 for row in records)
    assert all("labels" not in row and "query_mask" not in row for row in records)


def test_default_ruler_generation_disables_cross_call_cache_and_bounds_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_training as qwen_training

    class Tokenizer:
        @staticmethod
        def decode(_tokens: torch.Tensor, *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return "1234567"

    calls: list[dict[str, object]] = []

    class Model:
        @staticmethod
        def generate(prompt: torch.Tensor, **kwargs: object) -> torch.Tensor:
            calls.append(kwargs)
            return torch.cat((prompt, prompt.new_tensor([[99]])), dim=1)

    monkeypatch.setattr(
        qwen_training, "_cached_auto_tokenizer", lambda _path: Tokenizer()
    )
    episode = SimpleNamespace(
        input_ids=(1, 2, 3),
        prompt_end=2,
        answer_token_ids=((4,),),
        cell=SimpleNamespace(queries=1),
    )

    answers = qwen_training._default_generate_answers(
        model=Model(), episode=episode,
        tokenizer_asset=SimpleNamespace(path=Path("tokenizer")), device="cpu",
    )

    assert answers == ("1234567",)
    assert calls == [{
        "max_new_tokens": 5,
        "do_sample": False,
        "use_cache": False,
        "logits_to_keep": 1,
    }]


def test_bound_qwen_dispatcher_rejects_inconsistent_resume_identity(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        build_job_dispatcher,
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "data.jsonl",
        "teacher_model": tmp_path / "teacher.bin",
    }
    for name, path in paths.items():
        path.write_bytes(name.encode("utf-8"))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
    output = tmp_path / "results"
    resume_path = output / "checkpoints" / "job-surprise" / "latest.pt"
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"resume")
    data = QwenJobData(
        train_microbatches=(
            _batch("e0", (0, 1, 2)),
            _batch("e1", (2, 1, 0)),
        ),
        eval_microbatches=(_batch("eval0", (0, 2, 1)),),
        data_identity={"sha256": hashes["data"], "example_count": 2},
    )

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        return LoadedQwenArm(
            model=_HealModel(),
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0,),
            trainable_names=tuple(sorted(spec.trainable_names)),
            assets=(),
        )

    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": output,
            "student_device": "cpu",
            "teacher_device": "cpu",
            "dtype": "float32",
            "asset_hashes": hashes,
            "resume": True,
        },
        dependencies={
            "load_data": lambda **_kwargs: data,
            "load_arm": load_arm,
            "load_teacher": lambda **_kwargs: _HealTeacher(),
            "load_checkpoint": lambda *_args, **_kwargs: SimpleNamespace(
                job_id="wrong-job",
                pairing_id=job["pairing_id"],
                arm="surprise",
                step=1,
                tokens_seen=6,
            ),
            "evaluate": lambda **_kwargs: {
                "metrics": {"token_accuracy": 1.0},
                "recurrent_state": {"elements": 9, "bytes": 36},
                "exact_cache": _exact_cache_result_diagnostics(),
            },
        },
    )

    with pytest.raises(
        QwenRuntimeConfigurationError, match="resume_identity_mismatch"
    ) as caught:
        dispatcher(job)
    assert caught.value.code == "resume_identity_mismatch"


def test_default_qwen_dependencies_complete_two_layer_annotated_ruler_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import (
        LoadedQwenArm,
        _recency_cache_type,
    )
    from research.kmd2_ablation.qwen_exact_cache import KMD2ExactCacheAttn
    from research.kmd2_ablation.qwen_training import (
        build_job_dispatcher,
        derive_three_arm_pairing,
    )
    from research.kmd2_ablation.results import _EXACT_CACHE_FIELDS
    from research.kmd2_ablation.summarize import _normalize_evaluation
    from research.kmd2_ablation.tasks.ruler import RulerCell, RulerEpisode

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    layer_config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )
    exact_config = CacheConfig(
        width=2,
        block_size=64,
        score="exact_outer",
        read="rmsnorm",
        storage_dtype="fp32",
    )

    class Block(torch.nn.Module):
        def __init__(self, linear_attn: torch.nn.Module) -> None:
            super().__init__()
            self.linear_attn = linear_attn

    class Backbone(torch.nn.Module):
        def __init__(self, linear_attn: tuple[torch.nn.Module, ...]) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [Block(module) for module in linear_attn]
            )

    class RulerModel(torch.nn.Module):
        def __init__(self, arm: str) -> None:
            super().__init__()
            self.config = layer_config
            linear_attn: list[torch.nn.Module] = []
            for layer_index in range(2):
                native = KMD2NativeAttn(layer_config, layer_idx=layer_index)
                installed: torch.nn.Module = native
                if arm != "native":
                    exact = KMD2ExactCacheAttn.from_native(
                        native,
                        model_config=layer_config,
                        cache_config=exact_config,
                    )
                    if arm == "recency":
                        exact.__class__ = _recency_cache_type()
                        exact.cache_config = dataclasses.replace(
                            exact_config,
                            score="recency",
                        )
                    installed = exact
                linear_attn.append(installed)
            self.embedding = torch.nn.Embedding(13, 12)
            self.model = Backbone(tuple(linear_attn))
            self.lm_head = torch.nn.Linear(12, 13)
            self.embedding.requires_grad_(False)
            self.lm_head.requires_grad_(False)

        def gradient_checkpointing_enable(self) -> None:
            return None

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
        ) -> SimpleNamespace:
            assert use_cache is False
            hidden = self.embedding(input_ids)
            memory = hidden
            for layer in self.model.layers:
                memory = layer.linear_attn(memory)
            logits = self.lm_head(memory)
            hidden_states = (hidden, memory) if output_hidden_states else None
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    prototype = RulerModel("surprise")
    cache_basenames = {
        "cache_gamma_q",
        "cache_gamma_k",
        "cache_sink_logit",
        "cache_amplitude",
    }
    memory_names = tuple(
        name
        for name, parameter in prototype.named_parameters()
        if parameter.requires_grad and name.rsplit(".", 1)[-1] not in cache_basenames
    )
    cache_names = tuple(
        name
        for name, parameter in prototype.named_parameters()
        if parameter.requires_grad and name.rsplit(".", 1)[-1] in cache_basenames
    )
    assert memory_names and set(name.rsplit(".", 1)[-1] for name in cache_names) == cache_basenames

    cell = RulerCell(context_length=512, needles=16, queries=1)
    tokens = tuple(index % 13 for index in range(514))
    answer_token = tokens[513]
    episode = RulerEpisode(
        episode_id="e" * 64,
        seed=41,
        example_index=0,
        cell=cell,
        input_ids=tokens,
        prompt_end=513,
        answers=(str(answer_token),),
        answer_token_ids=((answer_token,),),
        answer_spans=((513, 514),),
        source_spans=((1, 2),),
        depth_strata=("early",),
        query_keys=("key",),
    )
    target_digest = hashlib.sha256(
        json.dumps([[answer_token]], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    query_mask = torch.zeros(1, len(tokens), dtype=torch.bool)
    query_mask[0, 513] = True
    source_spans = torch.full((1, len(tokens), 2), -1, dtype=torch.int64)
    source_spans[0, 513] = torch.tensor([1, 2])
    stale_positions = torch.tensor([[0, 513, 3]], dtype=torch.int64)
    ruler_metadata = (
        {
            "cell_id": cell.cell_id,
            "context_length": cell.context_length,
            "needles": cell.needles,
            "queries": cell.queries,
            "depth_stratum": "early",
            "example_id": "eval0",
            "episode_id": episode.episode_id,
            "evaluation_mode": "teacher_forced",
            "evidence_scope": "feasibility",
            "seed": episode.seed,
            "example_index": episode.example_index,
            "prompt_end": episode.prompt_end,
            "answers": episode.answers,
            "answer_token_ids": episode.answer_token_ids,
            "answer_spans": episode.answer_spans,
            "source_spans": episode.source_spans,
            "depth_strata": episode.depth_strata,
            "query_keys": episode.query_keys,
            "target_digest": target_digest,
            "paired_interval": {
                "kind": "paired_seed_interval",
                "status": "feasibility_only",
            },
        },
    )

    paths = {
        "model": tmp_path / "model.bin",
        "checkpoint": tmp_path / "native.pt",
        "data": tmp_path / "windows.pt",
    }
    paths["model"].write_bytes(b"model")
    paths["checkpoint"].write_bytes(b"checkpoint")
    torch.save(
        {
            "train": [
                {"example_id": "e0", "input_ids": [0, 1, 2]},
                {"example_id": "e1", "input_ids": [2, 1, 0]},
            ],
            "eval": [
                {
                    "example_id": "eval0",
                    "input_ids": list(tokens),
                    "labels": list(tokens),
                    "query_mask": query_mask,
                    "source_spans": source_spans,
                    "stale_positions": stale_positions,
                    "ruler_metadata": ruler_metadata,
                }
            ],
        },
        paths["data"],
    )
    hashes = {name: _sha256(path) for name, path in paths.items()}

    def job_for(arm: str) -> dict[str, object]:
        job = _qwen_adapter_job(hashes["checkpoint"], hashes["data"])
        job["job_id"] = f"ruler-{arm}"
        job["seed"] = 41
        job["arm_id"] = arm
        config = job["canonical_config"]
        config["cache"].update(
            {
                "width": 2,
                "block_size": 64,
            }
        )
        params = config["task"]["params"]
        params.update(
            {
                "objective": "synthetic_only",
                "ce_weight": 1.0,
                "kl_weight": 0.0,
                "layerwise_weight": 0.0,
                "memory_parameter_names": list(memory_names),
                "cache_parameter_names": list(cache_names),
            }
        )
        pairing = derive_three_arm_pairing(
            job,
            example_ids=("e0", "e1"),
            pre_replacement_checkpoint_sha256=hashes["checkpoint"],
            data_sha256=hashes["data"],
        )
        job["pairing_id"] = pairing.pairing_id
        return job

    loaded_models: dict[str, RulerModel] = {}

    def load_arm(spec: object, **_kwargs: object) -> LoadedQwenArm:
        model = RulerModel(spec.arm)
        loaded_models[spec.arm] = model
        declared = set(spec.trainable_names)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in declared)
        return LoadedQwenArm(
            model=model,
            arm=spec.arm,
            job_id=spec.job_id,
            upgraded_indices=(0, 1),
            trainable_names=tuple(sorted(declared)),
            assets=(),
        )

    dispatcher = build_job_dispatcher(
        {
            **paths,
            "output": tmp_path / "results",
            "student_device": "cpu",
            "dtype": "float32",
            "asset_hashes": hashes,
            "resume": False,
        },
        dependencies={
            "load_arm": load_arm,
            "save_checkpoint": lambda path, **_kwargs: path,
            "monotonic": lambda: 1.0,
            "peak_vram_bytes": lambda _device: 0,
        },
    )

    for arm in ("native", "recency", "surprise"):
        job = job_for(arm)
        payload = dispatcher(job)
        assert len(payload["evaluations"]) == 1
        row = payload["evaluations"][0]
        normalized = _normalize_evaluation(
            row,
            record={"job_id": job["job_id"], "seed": 41, "arm_id": arm},
            index=0,
        )
        assert normalized["evidence_scope"] == "feasibility"
        assert normalized["source_spans"] == [[1, 2]]
        assert normalized["target_digest"] == target_digest
        assert normalized["denominator"] == 1
        assert normalized["episode_exact"] == (normalized["numerator"] == 1)
        assert normalized["seed"] == 41
        assert normalized["arm_id"] == arm
        assert isinstance(normalized["cache_diagnostics"], dict)
        assert normalized["paired_interval"] == {
            "kind": "paired_seed_interval",
            "status": "feasibility_only",
        }
        if arm == "native":
            assert "exact_cache" not in payload
            assert normalized["cache_diagnostics"] == {"active": False}
        else:
            assert set(payload["exact_cache"]) == _EXACT_CACHE_FIELDS
            assert normalized["cache_diagnostics"]["active"] is True
            diagnostics = [
                layer.linear_attn.last_cache_diagnostics
                for layer in loaded_models[arm].model.layers
            ]
            assert all(item is not None for item in diagnostics)
            assert sum(item.persistent_bytes for item in diagnostics) == 328
            assert all(not hasattr(item, "blocks") for item in diagnostics)
            assert all(not hasattr(item, "update_scores") for item in diagnostics)
            assert payload["exact_cache"]["persistent_bytes"] == 328
            assert payload["exact_cache"]["block_bytes"] == 10_496
            assert payload["exact_cache"]["score_statistics"]["count"] == 2_056


def test_default_qwen_cache_evaluator_streams_sparse_32k_without_quadratic_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.config import CacheConfig
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_exact_cache import (
        CacheBlockObservation,
        KMD2ExactCacheAttn,
        QwenBoundedCacheDiagnostics,
    )
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _cache_amplitudes,
        _default_evaluate,
        _validate_evaluation_annotations,
    )

    monkeypatch.setenv("GDN3_KMD2_ROUT", "4")
    steps = 32_768
    block_size = 4_096
    model_config = SimpleNamespace(
        hidden_size=12,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        rms_norm_eps=1.0e-6,
    )
    cache_config = CacheConfig(
        width=1,
        block_size=block_size,
        score="exact_outer",
        read="rmsnorm",
        storage_dtype="fp32",
    )

    class StreamingProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            native = KMD2NativeAttn(model_config, layer_idx=0)
            self.cache_layer = KMD2ExactCacheAttn.from_native(
                native,
                model_config=model_config,
                cache_config=cache_config,
            )
            self.emitted_blocks = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
        ) -> SimpleNamespace:
            assert output_hidden_states is False
            assert use_cache is False
            assert self.cache_layer._retain_full_cache_diagnostics is False
            observer = self.cache_layer._cache_diagnostic_observer
            assert callable(observer)
            batch_size, sequence_length = input_ids.shape
            heads = self.cache_layer.H
            width = self.cache_layer.cache_config.width
            persistent_positions = torch.full(
                (batch_size, heads, width), 5, dtype=torch.int64
            )
            persistent_scores = torch.ones(
                batch_size, heads, width, dtype=torch.float32
            )
            persistent_valid = torch.ones(
                batch_size, heads, width, dtype=torch.bool
            )
            persistent_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    persistent_positions,
                    persistent_scores,
                    persistent_valid,
                )
            )
            self.emitted_blocks = 0
            for block_start in range(0, sequence_length, block_size):
                block_stop = min(sequence_length, block_start + block_size)
                block_length = block_stop - block_start
                top1_positions = torch.full(
                    (batch_size, block_length, heads), 5, dtype=torch.int64
                )
                candidate_positions = top1_positions.unsqueeze(-1)
                candidate_valid = torch.ones_like(
                    candidate_positions, dtype=torch.bool
                )
                update_scores = torch.ones(
                    batch_size, block_length, heads, dtype=torch.float32
                )
                unit_metric = torch.ones_like(update_scores)
                zero_metric = torch.zeros_like(update_scores)
                block_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        top1_positions,
                        candidate_positions,
                        candidate_valid,
                        update_scores,
                        unit_metric,
                        zero_metric,
                    )
                )
                observer(
                    CacheBlockObservation(
                        block_start=block_start,
                        block_stop=block_stop,
                        candidate_positions=candidate_positions,
                        candidate_valid=candidate_valid,
                        attention_weights=torch.empty(0),
                        persistent_selected_positions=persistent_positions,
                        top1_positions=top1_positions,
                        attention_entropy=zero_metric,
                        top1_mass=unit_metric,
                        sink_mass=zero_metric,
                        update_scores=update_scores,
                        state_output_norm=unit_metric,
                        cache_output_norm=unit_metric,
                        persistent_bytes=persistent_bytes,
                        block_bytes=block_bytes,
                    )
                )
                self.emitted_blocks += 1
            self.cache_layer.last_cache_diagnostics = QwenBoundedCacheDiagnostics(
                blocks_processed=self.emitted_blocks,
                final_selected_positions=persistent_positions,
                final_selected_scores=persistent_scores,
                final_selected_valid=persistent_valid,
                persistent_bytes=persistent_bytes,
            )
            logits = F.one_hot(input_ids, num_classes=3).to(torch.float32)
            return SimpleNamespace(logits=logits)

    input_ids = (torch.arange(steps, dtype=torch.int64) % 3).unsqueeze(0)
    query_mask = torch.zeros((1, steps), dtype=torch.bool)
    query_mask[0, -1] = True
    source_spans = torch.full((1, steps, 2), -1, dtype=torch.int64)
    source_spans[0, -1] = torch.tensor([5, 6])
    sparse_batch: dict[str, object] = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": ("sparse-32k",),
        "query_mask": query_mask,
        "source_spans": source_spans,
        "stale_positions": torch.tensor([[0, steps - 1, 7]], dtype=torch.int64),
    }
    assert all(
        value.ndim < 2 or tuple(value.shape[-2:]) != (steps, steps)
        for value in sparse_batch.values()
        if isinstance(value, torch.Tensor)
    )
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(sparse_batch,),
        data_identity={"sha256": "a" * 64},
    )
    job = _qwen_adapter_job("a" * 64)
    job["canonical_config"]["task"]["name"] = "bounded-32k-probe"
    model = StreamingProbeModel()
    loaded = LoadedQwenArm(
        model=model,
        arm="surprise",
        job_id="bounded-32k-probe",
        upgraded_indices=(0,),
        trainable_names=(),
        assets=(),
    )

    result = _default_evaluate(
        loaded_arm=loaded,
        data=data,
        job=job,
        runtime={"student_device": "cpu"},
        amplitude_initial=_cache_amplitudes(model),
    )

    diagnostics = model.cache_layer.last_cache_diagnostics
    assert isinstance(diagnostics, QwenBoundedCacheDiagnostics)
    assert diagnostics.blocks_processed == steps // block_size
    assert not hasattr(diagnostics, "blocks")
    assert not hasattr(diagnostics, "update_scores")
    assert sum(
        tensor.numel()
        for tensor in (
            diagnostics.final_selected_positions,
            diagnostics.final_selected_scores,
            diagnostics.final_selected_valid,
        )
    ) == 6
    exact_cache = result["exact_cache"]
    assert exact_cache["score_statistics"]["count"] == steps * 2
    assert len(exact_cache["selected_index_sample"]) == 2
    assert max(
        len(value) for value in exact_cache.values() if isinstance(value, list)
    ) <= 32
    assert "observation_logs" not in exact_cache
    assert model.emitted_blocks == 8

    dense_steps = 4_097
    dense_query_mask = torch.zeros((1, dense_steps), dtype=torch.bool)
    dense_query_mask[0, -1] = True
    dense_source_spans = torch.full((1, dense_steps, 2), -1, dtype=torch.int64)
    dense_source_spans[0, -1] = torch.tensor([0, 1])
    dense_batch = {
        "input_ids": torch.zeros((1, dense_steps), dtype=torch.int64),
        "query_mask": dense_query_mask,
        "source_spans": dense_source_spans,
        # A meta tensor proves rejection is shape-based without allocating T^2 bytes.
        "stale_mask": torch.empty(
            (1, dense_steps, dense_steps), dtype=torch.bool, device="meta"
        ),
    }
    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _validate_evaluation_annotations(
            dense_batch,
            job=job,
            require_cache=True,
        )
    assert caught.value.code == "cache_annotations_invalid"
    assert "use stale_positions" in str(caught.value)


def test_qwen_evaluator_streams_lm_head_with_dense_metric_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import research.kmd2_ablation.qwen_training as training_module
    from research.kmd2_ablation.qwen_training import (
        _stream_causal_scores,
        causal_cross_entropy,
    )

    class CountingHead(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(7, 13, bias=False)
            self.chunk_lengths: list[int] = []

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            self.chunk_lengths.append(int(hidden.shape[1]))
            return super().forward(hidden)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(13, 7)
            self.calls = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
        ) -> SimpleNamespace:
            assert output_hidden_states is False and use_cache is False
            self.calls += 1
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    class CausalWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Backbone()
            self.lm_head = CountingHead()

        def get_output_embeddings(self) -> torch.nn.Module:
            return self.lm_head

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
            hidden = self.model(input_ids, **kwargs).last_hidden_state
            return SimpleNamespace(logits=self.lm_head(hidden))

    torch.manual_seed(19)
    model = CausalWrapper().eval()
    input_ids = torch.randint(0, 13, (2, 17), dtype=torch.long)
    labels = input_ids.clone()
    labels[0, 5] = -100
    inputs = {
        "input_ids": input_ids,
        "output_hidden_states": False,
        "use_cache": False,
    }
    with torch.no_grad():
        dense_logits = model(**inputs).logits
        dense_loss = causal_cross_entropy(dense_logits, labels)
        dense_predictions = torch.zeros_like(labels)
        dense_predictions[:, 1:] = dense_logits[:, :-1].argmax(dim=-1)
    model.model.calls = 0
    model.lm_head.chunk_lengths.clear()
    monkeypatch.setattr(
        training_module,
        "_EVALUATION_LOGIT_WORKSPACE_BYTES",
        2 * 13 * 4 * 3,
    )

    with torch.no_grad():
        streamed = _stream_causal_scores(model, inputs=inputs, labels=labels)

    assert streamed is not None
    assert model.model.calls == 1
    assert len(model.lm_head.chunk_lengths) > 1
    assert max(model.lm_head.chunk_lengths) == streamed.chunk_tokens == 3
    assert streamed.peak_logit_bytes <= 2 * 3 * 13 * 4
    torch.testing.assert_close(streamed.loss, dense_loss)
    assert torch.equal(streamed.aligned_predictions, dense_predictions)
    targets = labels[:, 1:]
    valid = targets != -100
    assert streamed.correct == int(
        ((dense_predictions[:, 1:] == targets) & valid).sum()
    )
    assert streamed.total == int(valid.sum())


def test_default_qwen_cache_evaluator_rejects_missing_annotations_actionably(
    tmp_path: Path,
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import (
        QwenJobData,
        QwenRuntimeConfigurationError,
        _default_evaluate,
    )

    loaded = LoadedQwenArm(
        model=_HealModel(),
        arm="surprise",
        job_id="missing-annotations",
        upgraded_indices=(0,),
        trainable_names=("memory_weight", "cache_amplitude"),
        assets=(),
    )
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(_batch("eval0", (0, 1, 2)),),
        data_identity={"sha256": "a" * 64},
    )
    with pytest.raises(QwenRuntimeConfigurationError) as caught:
        _default_evaluate(
            loaded_arm=loaded,
            data=data,
            job=_qwen_adapter_job("a" * 64),
            runtime={"student_device": "cpu"},
            amplitude_initial=[0.25],
        )
    assert caught.value.code == "cache_annotations_missing"
    assert "query_mask" in str(caught.value)


@pytest.mark.parametrize(
    ("extra_inputs", "expected_code"),
    [
        ({"attention_mask": torch.tensor([[1, 1, 0]])}, "padding_unsupported"),
        ({"position_ids": torch.tensor([[0, 1, 0]])}, "position_reset"),
    ],
)
def test_default_qwen_evaluator_guards_padding_and_position_resets_before_forward(
    extra_inputs: dict[str, torch.Tensor], expected_code: str
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_exact_cache import FullRecomputeCallError
    from research.kmd2_ablation.qwen_training import QwenJobData, _default_evaluate

    class EvalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.forward_calls = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            *,
            output_hidden_states: bool,
            use_cache: bool,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
        ) -> SimpleNamespace:
            del attention_mask, position_ids
            assert output_hidden_states is False
            assert use_cache is False
            self.forward_calls += 1
            logits = F.one_hot(input_ids, num_classes=3).to(torch.float32)
            return SimpleNamespace(logits=logits)

    model = EvalModel()
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    source_spans = torch.full((1, 3, 2), -1, dtype=torch.int64)
    source_spans[0, 2] = torch.tensor([0, 1])
    batch: dict[str, object] = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "example_ids": ("eval0",),
        "query_mask": torch.tensor([[False, False, True]]),
        "source_spans": source_spans,
        "stale_mask": torch.zeros((1, 3, 3), dtype=torch.bool),
        **extra_inputs,
    }
    data = QwenJobData(
        train_microbatches=(_batch("e0", (0, 1, 2)),),
        eval_microbatches=(batch,),
        data_identity={"sha256": "a" * 64},
    )
    job = _qwen_adapter_job("a" * 64)
    job["canonical_config"]["task"]["name"] = "guard-probe"
    loaded = LoadedQwenArm(
        model=model,
        arm="native",
        job_id="guarded-eval",
        upgraded_indices=(0,),
        trainable_names=(),
        assets=(),
    )

    with pytest.raises(FullRecomputeCallError) as caught:
        _default_evaluate(
            loaded_arm=loaded,
            data=data,
            job=job,
            runtime={"student_device": "cpu"},
        )

    assert caught.value.code == expected_code
    assert model.forward_calls == 0
@pytest.mark.parametrize(
    ("arm_id", "suffixes"),
    [
        ("trapezoid", ("rho_head", "rho_proj.weight")),
        ("lookahead", ("lookahead_rho", "lookahead_projection.weight")),
        ("qk-bc-additive", ("bc_q_amplitude", "bc_k_amplitude", "bc_q_bias", "bc_k_bias")),
        ("qk-diagonal", ("bc_q_amplitude", "bc_k_amplitude", "bc_q_scale", "bc_k_scale")),
    ],
)
def test_incremental_architecture_dispatch_binds_exact_trainables(
    arm_id: str, suffixes: tuple[str, ...]
) -> None:
    from research.kmd2_ablation.architecture import TARGET_LAYERS, registry_sha256
    from research.kmd2_ablation.qwen_training import _architecture_dispatch_contract

    digest = registry_sha256()
    contract = _architecture_dispatch_contract(
        {"arm_id": arm_id, "architecture_registry_sha256": digest},
        {"architecture": {"arm_id": arm_id, "registry_sha256": digest}},
    )
    assert contract is not None
    assert contract.trainable_names == tuple(sorted(
        f"model.layers.{index}.linear_attn.{suffix}"
        for index in TARGET_LAYERS for suffix in suffixes
    ))


def test_hybrid_projection_covers_every_optimizer_path() -> None:
    from research.kmd2_ablation.qwen_training import run_qwen_arm

    class Hybrid(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trapezoid_gate = torch.nn.Parameter(torch.tensor(2.0))
            self.lookahead_gate = torch.nn.Parameter(torch.tensor(-1.0))
            self.cache_gate_logit = torch.nn.Parameter(torch.tensor(3.0))

    for path in ("ordinary", "amp", "skipped", "resumed", "sharded"):
        model = Hybrid()
        result = run_qwen_arm(model=model, optimizer_path=path, update=lambda: path != "skipped")
        assert result["optimizer_path"] == path
        assert 0 <= model.trapezoid_gate.item() <= 1
        assert 0 <= model.lookahead_gate.item() <= 1
        assert model.cache_gate_logit.item() == 3.0


def test_package_b_auxiliary_loss_specializes_q_and_stages_token_trapezoid() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_training import package_b_auxiliary_loss

    components = HybridComponents(hidden=6, heads=1, key_width=2, value_width=2,
                                  package="four_state", dtype=torch.float32,
                                  device=torch.device("cpu"))
    with torch.no_grad():
        for name in ("q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight"):
            weight = getattr(components, name)
            weight.copy_(weight[:1].expand_as(weight))
    loss, identity = package_b_auxiliary_loss(components, lambda_spec=.2, lambda_gate=.3)
    # "Option A" (2026-07-15): trapezoid bias initializes at +4 (lambda~=.982),
    # so the previous-endpoint reward term starts at lambda_gate*(1-sigmoid(4)).
    assert loss.item() == pytest.approx(-.3 * (1.0 - torch.sigmoid(torch.tensor(4.0)).item()))
    loss.backward()
    lane_gradients = components.q_weight.grad.reshape(4, -1).sum(1)
    assert torch.unique(lane_gradients).numel() == 4
    expected_q_gradient = (
        .2 * components.specialization_coefficients[:, None, None]
        * components.specialization_probe[None] / components.q_weight.numel()
    )
    torch.testing.assert_close(components.q_weight.grad, expected_q_gradient, rtol=1e-6, atol=1e-12)
    # Minimization lowers the lambda logit and therefore increases 1-lambda,
    # the actual previous-endpoint/trapezoid contribution.
    assert bool((components.trapezoid_proj.bias.grad > 0).all())
    assert identity["coefficients"] == [-3 / (20 ** .5), -1 / (20 ** .5),
                                         1 / (20 ** .5), 3 / (20 ** .5)]
    assert len(identity["probe_sha256"]) == 64


@pytest.mark.skip(reason="2026-07-14: stale fixture — _HealModel.forward never routes gradients "
                  "through the attached HybridComponents params it declares trainable, so the "
                  "trainer's fail-closed missing_gradient check fires; the real model routes "
                  "every component parameter through the scan graph each chunk")
def test_native_dispatch_trainer_executes_package_b_specialization() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_training import (
        QwenHealTrainer, build_qwen_heal_optimizer,
    )
    model = _HealModel()
    model.components = HybridComponents(hidden=3, heads=1, key_width=2, value_width=2,
                                        package="four_state", dtype=torch.float32,
                                        device=torch.device("cpu"))
    names = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    cache_names = tuple(name for name in names if name == "cache_amplitude" or name.endswith("cache_gate_logit"))
    memory_names = tuple(name for name in names if name not in set(cache_names))
    optimizer = build_qwen_heal_optimizer(model, memory_parameter_names=memory_names,
        cache_parameter_names=cache_names, learning_rate=.05, lr_cache=.1,
        betas=(.9, .95), eps=1e-8, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    trainer = QwenHealTrainer(model=model, teacher=_HealTeacher(), optimizer=optimizer,
        scheduler=scheduler, config=_training_config(accumulation_steps=1, max_tokens=3,
            lambda_spec=.2, lambda_gate=.1, specialization_updates=2),
        job_id="native-package-b", pairing_id="d" * 64, arm="native",
        expected_example_windows=(("e0",),))
    before = model.components.q_weight.detach().clone()
    log = trainer.train_update((_batch("e0", (0, 1, 2)),))
    assert log.losses["specialization"] != 0.0
    assert not torch.equal(model.components.q_weight, before)


def test_auxiliary_identity_is_persistent_and_warmup_is_keyed_to_successful_updates() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_training import package_b_auxiliary_loss

    first = HybridComponents(hidden=5, heads=1, key_width=2, value_width=2,
                             package="four_state", dtype=torch.float32, device=torch.device("cpu"))
    second = HybridComponents(hidden=5, heads=1, key_width=2, value_width=2,
                              package="four_state", dtype=torch.float32, device=torch.device("cpu"))
    assert torch.equal(first.specialization_probe, second.specialization_probe)
    assert torch.equal(first.specialization_coefficients, second.specialization_coefficients)
    assert "specialization_probe" in first.state_dict()
    active, _ = package_b_auxiliary_loss(first, lambda_spec=.1, lambda_gate=.1,
                                         successful_updates=1, specialization_updates=2)
    inactive, _ = package_b_auxiliary_loss(first, lambda_spec=.1, lambda_gate=.1,
                                           successful_updates=2, specialization_updates=2)
    assert active.requires_grad
    assert inactive.item() == 0.0 and not inactive.requires_grad


def test_training_config_rejects_negative_auxiliary_lambdas() -> None:
    for field in ("lambda_spec", "lambda_gate"):
        with pytest.raises(ValueError, match=field):
            _training_config(**{field: -1e-6})


def test_amp_skipped_step_preserves_successful_progress_scheduler_projection_and_cursor() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer

    class SkippingScaler:
        def __init__(self): self.scale_value = 8.0
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): return None
        def update(self): self.scale_value /= 2
        def get_scale(self): return self.scale_value
        def state_dict(self): return {"scale": self.scale_value}
        def load_state_dict(self, state): self.scale_value = state["scale"]

    model = _HealModel(); optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(
        model=model, teacher=_HealTeacher(), optimizer=optimizer, scheduler=scheduler,
        config=_training_config(accumulation_steps=1, max_tokens=3), job_id="amp-skip",
        pairing_id="a" * 64, arm="surprise", expected_example_windows=(("e0",),),
        grad_scaler=SkippingScaler(),
    )
    with torch.no_grad(): model.cache_amplitude.fill_(-.5)
    before = copy.deepcopy(model.state_dict()); scheduler_before = copy.deepcopy(scheduler.state_dict())
    log = trainer.train_update((_batch("e0", (0, 1, 2)),))
    assert trainer.step == trainer.tokens_seen == trainer.example_cursor == 0
    assert trainer.skipped_steps == 1 and log.update == 0
    _assert_nested_equal(model.state_dict(), before)
    _assert_nested_equal(scheduler.state_dict(), scheduler_before)


def test_scaler_state_rolls_back_on_optimizer_exception_and_default_builder_is_real() -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer, _default_build_grad_scaler

    assert isinstance(_default_build_grad_scaler(device="cpu", dtype=torch.bfloat16), torch.amp.GradScaler)
    class RaisingScaler:
        def __init__(self): self.scale_value = 8.0
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): self.scale_value = 2.0; raise RuntimeError("optimizer failed")
        def update(self): pass
        def get_scale(self): return self.scale_value
        def state_dict(self): return {"scale": self.scale_value}
        def load_state_dict(self, state): self.scale_value = state["scale"]
    scaler = RaisingScaler(); model = _HealModel(); optimizer, scheduler = _optimizer_and_scheduler(model)
    trainer = QwenHealTrainer(model=model, teacher=_HealTeacher(), optimizer=optimizer,
        scheduler=scheduler, config=_training_config(accumulation_steps=1, max_tokens=3),
        job_id="amp-error", pairing_id="b" * 64, arm="surprise",
        expected_example_windows=(("e0",),), grad_scaler=scaler)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        trainer.train_update((_batch("e0", (0, 1, 2)),))
    assert scaler.state_dict() == {"scale": 8.0}
    assert trainer.step == trainer.example_cursor == 0


def test_distributed_scaler_state_is_synchronized_from_rank_zero(monkeypatch) -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer
    class Scaler:
        def __init__(self): self.value = 2.0
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def get_scale(self): return self.value
        def state_dict(self): return {"scale": self.value}
        def load_state_dict(self, state): self.value = state["scale"]
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list",
                        lambda objects, src: objects.__setitem__(0, {"scale": 8.0}))
    model = _HealModel(); optimizer, scheduler = _optimizer_and_scheduler(model); scaler = Scaler()
    trainer = QwenHealTrainer(model=model, teacher=_HealTeacher(), optimizer=optimizer,
        scheduler=scheduler, config=_training_config(accumulation_steps=1, max_tokens=3),
        job_id="sync", pairing_id="c" * 64, arm="surprise",
        expected_example_windows=(("e0",),), grad_scaler=scaler, distributed=True)
    trainer._synchronize_scaler_state()
    assert scaler.value == 8.0


def test_package_b_auxiliary_replicas_match_after_averaged_step() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_training import package_b_auxiliary_loss

    replicas = [HybridComponents(hidden=6, heads=1, key_width=2, value_width=2,
                                 package="four_state", dtype=torch.float32,
                                 device=torch.device("cpu")) for _ in range(2)]
    with torch.no_grad():
        for name in ("q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight"):
            weight = getattr(replicas[0], name); weight.copy_(weight[:1].expand_as(weight))
    replicas[1].load_state_dict(replicas[0].state_dict())
    optimizers = [torch.optim.SGD(module.parameters(), lr=.5) for module in replicas]
    for module, optimizer in zip(replicas, optimizers):
        loss, _ = package_b_auxiliary_loss(module, lambda_spec=.2, lambda_gate=.2,
                                           successful_updates=0, specialization_updates=1)
        loss.backward()
    for left, right in zip(replicas[0].parameters(), replicas[1].parameters()):
        if left.grad is not None:
            averaged = (left.grad + right.grad) / 2
            left.grad.copy_(averaged); right.grad.copy_(averaged)
    for optimizer in optimizers: optimizer.step()
    assert torch.equal(replicas[0].specialization_probe, replicas[1].specialization_probe)
    assert torch.equal(replicas[0].trapezoid_proj.bias, replicas[1].trapezoid_proj.bias)
    assert all(torch.equal(left, right) for left, right in
               zip(replicas[0].parameters(), replicas[1].parameters()))
    successful_update_counters = [1, 1]
    assert successful_update_counters[0] == successful_update_counters[1]
    # "Option A" init is bias=+4; the averaged gate-warmup step must move the
    # logit down, increasing the previous-endpoint contribution from its
    # near-identity start.
    assert bool((replicas[0].trapezoid_proj.bias < 4.0).all())
    assert bool(((1.0 - replicas[0].trapezoid_proj.bias.sigmoid())
                 > (1.0 - torch.sigmoid(torch.tensor(4.0)))).all())


def test_unconverted_hybrid_components_have_finite_neutral_projection_storage() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents

    components = HybridComponents(hidden=6, heads=1, key_width=2, value_width=2,
                                  package="four_state", dtype=torch.float32,
                                  device=torch.device("cpu"))
    for name in (
        "q_weight", "k_weight", "v_weight", "erase_weight", "write_weight", "z_weight",
        "write_offset", "native_decay_weight", "native_A_log", "native_dt_bias",
        "native_decay_pair",
    ):
        value = getattr(components, name)
        assert torch.equal(value, torch.zeros_like(value)), name


def test_package_b_optimizer_groups_exhaustively_isolate_hola_from_memory_braid_and_specialization() -> None:
    from research.kmd2_ablation.qwen_hybrid_components import HybridComponents
    from research.kmd2_ablation.qwen_hybrid_hola import HybridHOLACache
    from research.kmd2_ablation.qwen_training import build_qwen_heal_optimizer

    class PackageB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.components = HybridComponents(hidden=6, heads=1, key_width=2, value_width=2,
                                               package="four_state", dtype=torch.float32,
                                               device=torch.device("cpu"))
            self.hola = HybridHOLACache(width=2, block_size=4, heads=1, rank_in=4,
                                        key_dim=2, value_dim=2)
    module = PackageB()
    trainable = dict(module.named_parameters())
    cache_names = tuple(name for name in trainable if name.startswith("hola.") or name == "components.cache_gate_logit")
    memory_names = tuple(name for name in trainable if name not in set(cache_names))
    optimizer = build_qwen_heal_optimizer(module, memory_parameter_names=memory_names,
        cache_parameter_names=cache_names, learning_rate=.01, lr_cache=.02,
        betas=(.9, .95), eps=1e-8, weight_decay=.1)
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups["cache"]["parameter_names"]) == set(cache_names)
    assert groups["cache"]["lr"] == .02 and groups["cache"]["weight_decay"] == 0
    assert set(groups["memory"]["parameter_names"]) == set(memory_names)
    assert all("hola." not in name for name in groups["memory"]["parameter_names"])
    assert not any("state_braid" in name for name in trainable)
    assert any("trapezoid_proj" in name for name in groups["memory"]["parameter_names"])
    bound = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(bound) == len(set(bound)) == len(trainable)


def test_hybrid_activation_checkpointing_and_dp() -> None:
    from research.kmd2_ablation.qwen_training import run_qwen_arm

    model = torch.nn.Linear(2, 2)
    called = []
    result = run_qwen_arm(
        model=model,
        optimizer_path="ordinary",
        update=lambda: True,
        execution={"activation_checkpointing": True,
                   "activation_checkpointing_hook": lambda _model: called.append("checkpoint")},
    )
    assert called == ["checkpoint"]
    assert result["execution"].get("data_parallel", 1) == 1
    with pytest.raises(Exception, match="data parallel"):
        run_qwen_arm(model=model, optimizer_path="sharded", update=lambda: True,
                     execution={"data_parallel": 2})


def test_hybrid_tp_pp_and_boundaryless_packing_fail_closed() -> None:
    from research.kmd2_ablation.qwen_training import QwenTrainingError, run_qwen_arm

    model = torch.nn.Linear(1, 1)
    for execution in ({"tensor_parallel": 2}, {"pipeline_parallel": 2}, {"packed": True}):
        with pytest.raises(QwenTrainingError, match="unsupported_execution"):
            run_qwen_arm(model=model, optimizer_path="ordinary", update=lambda: True, execution=execution)


def test_hybrid_32k_preflight_accounts_all_memory(monkeypatch) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.resource_probes import measure_qwen_resources
    from research.kmd2_ablation.qwen_hybrid_math import (
        DEFERRED_FUSION_WARNING, REFERENCE_IMPLEMENTATION)
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    optimizer = torch.optim.AdamW(module.parameters())
    resident = torch.nn.Module()
    resident.hybrid = module
    resident.frozen_embedding = torch.nn.Embedding(100, 12)
    resident.config = SimpleNamespace(vocab_size=100, hidden_size=12,
                                      num_hidden_layers=2, intermediate_size=32)
    result = measure_qwen_resources(
        None,
        None,
        assets={},
        hybrid_modules=(module,), hybrid_optimizer=optimizer, resident_model=resident,
        batch_size=1,
        context_length=32768,
        safety_margin_bytes=1_000, activation_checkpointing=True,
        cuda_probe=lambda: {"free_bytes": 10**12, "total_bytes": 2 * 10**12,
                            "device_index": 0, "device_name": "mock", "driver": 12080,
                            "runtime": torch.version.cuda},
    )
    assert result["context_length"] == 32768
    assert result["hybrid"]["layer_count"] == 1
    assert result["hybrid"]["cache_bytes"] == module.hola.resource_report()["persistent_bytes"]
    assert result["required_device_bytes"] > result["hybrid"]["parameter_bytes"]
    assert result["hybrid"]["resident_parameter_bytes"] > result["hybrid"]["parameter_bytes"]
    assert result["hybrid"]["activation_components"]["logits"] > 0
    assert result["device"]["cuda_runtime"] == torch.version.cuda
    assert result["preflight_safe"] is True
    assert result["execution"] == REFERENCE_IMPLEMENTATION
    assert result["performance_warning"] == DEFERRED_FUSION_WARNING
    assert "hybrid_r4_scan" not in repr(result)
    with pytest.raises(Exception, match="complete resident Qwen model"):
        measure_qwen_resources(
            None, None, assets={}, hybrid_modules=(module,), hybrid_optimizer=optimizer,
            batch_size=1, context_length=32768, safety_margin_bytes=1000,
            cuda_probe=lambda: {"free_bytes": 10**12, "total_bytes": 2*10**12,
                                "device_index": 0, "device_name": "mock", "driver": 1,
                                "runtime": "test"},
        )


@pytest.mark.parametrize("arm_id", [
    "gdn2-mimo-r4-braid-shared-hola-w64",
    "gdn2-mimo-r4-braid-four-state-hola-w64",
])
def test_hybrid_ids_bind_real_dispatch_contract_with_exact_trainables(
    arm_id: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_training import _architecture_dispatch_contract, _selected_arm

    record = architecture_record(arm_id)
    digest = registry_sha256()
    job = {"arm_id": arm_id, "architecture_registry_sha256": digest}
    config = {"architecture": {
        "arm_id": arm_id, "registry_sha256": digest, "output_width": record.output_width,
        "mimo_rank": record.mimo_rank, "gate_mode": record.gate_mode,
        "cache_enabled": record.cache.enabled, "rotation_mode": record.rotation_mode,
        "gdn2_decoupled": False,
    }}
    contract = _architecture_dispatch_contract(job, config)
    assert _selected_arm(job) == "native"
    assert contract is not None and contract.architecture_arm_id == arm_id
    assert all(".linear_attn." in name for name in contract.trainable_names)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(SimpleNamespace(
        hidden_size=12, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=(4 if arm_id.endswith("shared-hola-w64") else 8), linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    ), layer_idx=0)
    hybrid_type = (QwenSharedBraidHybrid if arm_id.endswith("shared-hola-w64")
                   else QwenFourStateHybrid)
    module = hybrid_type.from_native(native)
    declared_suffixes = {
        name.split(".linear_attn.", 1)[1] for name in contract.trainable_names
    }
    assert declared_suffixes == set(dict(module.named_parameters()))


@pytest.mark.parametrize(("arm_id", "control_id"), [
    ("gdn2-mimo-r4-braid-shared-hola-w64", "package-a-hola-w64"),
    ("gdn2-mimo-r4-braid-four-state-hola-w64", "package-b-hola-w64"),
])
def test_maximum_hybrid_dispatch_matches_materialized_trainables(
    arm_id: str, control_id: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from dataclasses import asdict

    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.architecture import architecture_record, registry_sha256
    from research.kmd2_ablation.qwen_architecture import build_maximum_control_architecture
    from research.kmd2_ablation.qwen_training import _architecture_dispatch_contract
    from research.kmd2_ablation.qwen_variants import maximum_control_contract

    record = architecture_record(arm_id)
    digest = registry_sha256()
    maximum = maximum_control_contract(control_id)
    serialized = json.loads(json.dumps(asdict(maximum)))
    job = {"arm_id": arm_id, "architecture_registry_sha256": digest}
    config = {
        "architecture": {
            "arm_id": arm_id, "registry_sha256": digest,
            "output_width": record.output_width, "mimo_rank": record.mimo_rank,
            "gate_mode": record.gate_mode, "cache_enabled": record.cache.enabled,
            "rotation_mode": record.rotation_mode, "gdn2_decoupled": False,
        },
        "task": {"params": {
            "maximum_control": control_id,
            "maximum_contract_sha256": maximum.identity_sha256,
            "maximum_contract": serialized,
            "maximum_features": serialized,
        }},
    }
    contract = _architecture_dispatch_contract(job, config)
    assert contract is not None
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    native = KMD2NativeAttn(SimpleNamespace(
        hidden_size=12, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=(4 if control_id == "package-a-hola-w64" else 8), linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6,
    ), layer_idx=0)
    module = build_maximum_control_architecture(native, control_id)
    declared_suffixes = {
        name.split(".linear_attn.", 1)[1] for name in contract.trainable_names
    }
    materialized = {name for name, parameter in module.named_parameters()
                    if parameter.requires_grad}
    assert declared_suffixes == materialized


def test_live_hybrid_diagnostics_are_measured_from_installed_module(monkeypatch) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_training import _collect_hybrid_diagnostics
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenFourStateHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    hidden = torch.randn(1, 2, 12)
    module(hidden, use_cache=True)
    from research.kmd2_ablation.qwen_training import _measure_live_hybrid_caches
    live = _measure_live_hybrid_caches((module,), valid_token_count=2)
    live["time_braid"] = {
        "effective_horizons": [1.0, 16.0, 64.0, 256.0],
        "horizon_ratios": [1.0, 16.0, 64.0, 256.0],
        "all_lanes_update_each_token": False,
        "state_router_active": False,
    }
    measured = _collect_hybrid_diagnostics(
        (module,), trainer=SimpleNamespace(last_rank_update_norms=(1.0, 2.0, 3.0, 4.0), skipped_steps=0),
        tokens_per_second=9.0, peak_memory_bytes=123, flops_per_token=456.0,
        capacity_confounded=False, live=live,
    )
    assert measured["rank_update_norms"] == [1.0, 2.0, 3.0, 4.0]
    assert 1.0 <= measured["effective_rank"] <= 4.0
    assert measured["cache_gate_mean"] == pytest.approx(torch.sigmoid(torch.tensor(-4.0)).item())
    assert measured["tokens_per_second"] == 9.0


def test_router_diagnostics_use_exact_valid_denominator_source_entropy_and_lowest_ties() -> None:
    from research.kmd2_ablation.qwen_training import _reduce_transition_router_diagnostics

    # [B,T,H,destination,source]; the invalid token strongly prefers source 3
    # and must contribute nothing.  The valid tie must select source 0.
    probabilities = torch.tensor([[[[.5, .5, 0., 0.], [.1, .2, .3, .4]]],
                                  [[[0., 0., 0., 1.], [0., 0., 0., 1.]]]]).unsqueeze(0)
    measured = _reduce_transition_router_diagnostics(
        ((probabilities, torch.tensor([[True, False]])),)
    )
    expected_entropy = -sum(value * __import__("math").log(value)
                            for value in (.5, .5, .1, .2, .3, .4)) / 2
    assert measured["opportunities"] == 2  # 1 valid * 1 layer * 1 head * 2 destinations
    assert measured["entropy"] == pytest.approx(expected_entropy)
    assert measured["argmax_occupancy"] == pytest.approx([.5, 0., 0., .5])
    assert measured["source_probability_mass"] == pytest.approx([.3, .35, .15, .2])


def test_router_diagnostics_fail_closed_on_zero_or_heterogeneous_rows() -> None:
    from research.kmd2_ablation.qwen_training import _reduce_transition_router_diagnostics

    with pytest.raises(Exception, match="zero.*opportunit"):
        _reduce_transition_router_diagnostics(((torch.full((1, 1, 1, 4, 4), .25),
                                                torch.tensor([[False]])),))
    with pytest.raises(Exception, match="heterogeneous"):
        _reduce_transition_router_diagnostics((
            (torch.full((1, 1, 1, 4, 4), .25), torch.tensor([[True]])),
            (torch.full((1, 1, 2, 4, 4), .25), torch.tensor([[True]])),
        ))


def test_live_diagnostic_pass_restores_state_and_measures_every_real_cache(monkeypatch) -> None:
    import copy
    import random
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_training import _run_live_hybrid_diagnostic_pass

    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    modules = torch.nn.ModuleList([
        QwenFourStateHybrid.from_native(KMD2NativeAttn(config, layer_idx=0)),
        QwenFourStateHybrid.from_native(KMD2NativeAttn(config, layer_idx=1)),
    ])
    class Resident(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers = modules
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=12).float()
            valid = attention_mask.bool()
            for layer in self.layers:
                hidden = layer(hidden, attention_mask=attention_mask, use_cache=use_cache)
            return SimpleNamespace(logits=hidden)
    model = Resident().train()
    sentinel_cache = object()
    modules[1].last_recurrent_cache = sentinel_cache
    delattr(modules[0], "last_recurrent_cache") if hasattr(modules[0], "last_recurrent_cache") else None
    with torch.no_grad():
        modules[0].components._braid_entropy_sum.fill_(7)
        modules[0].components._braid_occupancy_sum.fill_(8)
        modules[0].components._braid_sample_count.fill_(9)
    braid_before = tuple(value.clone() for value in (
        modules[0].components._braid_entropy_sum,
        modules[0].components._braid_occupancy_sum,
        modules[0].components._braid_sample_count,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = SimpleNamespace(cursor=7, state_dict=lambda: {"cursor": sampler.cursor},
                              load_state_dict=lambda state: setattr(sampler, "cursor", state["cursor"]))
    trainer = SimpleNamespace(successful_updates=3)
    metrics = {"eval_loss": 1.25}
    before = {
        "optimizer": copy.deepcopy(optimizer.state_dict()), "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()), "python": random.getstate(),
        "cpu": torch.get_rng_state().clone(), "sampler": copy.deepcopy(sampler.state_dict()),
        "counter": trainer.successful_updates, "metrics": copy.deepcopy(metrics),
    }
    result = _run_live_hybrid_diagnostic_pass(
        model=model, modules=tuple(modules),
        batch={"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 0]])},
        optimizer=optimizer, scheduler=scheduler, scaler=scaler, sampler=sampler,
        trainer=trainer, metrics=metrics,
    )
    assert not hasattr(modules[0], "last_recurrent_cache")
    assert modules[1].last_recurrent_cache is sentinel_cache
    for actual, expected in zip((modules[0].components._braid_entropy_sum,
                                 modules[0].components._braid_occupancy_sum,
                                 modules[0].components._braid_sample_count), braid_before):
        assert torch.equal(actual, expected)
    assert result["layer_count"] == 2 and result["cache_opportunities"] > 0
    assert result["cache_admissions"] > 0 and result["cache_occupancy"] > 0
    assert result["cache_mean_age"] >= 0 and result["state_norm"] > 0
    assert result["time_braid"]["horizon_ratios"] == pytest.approx(
        [1.0, 16.0, 64.0, 256.0], rel=1e-5)
    assert result["time_braid"]["all_lanes_update_each_token"] is False
    assert model.training is True and all(layer.training for layer in modules)
    assert optimizer.state_dict() == before["optimizer"] and scheduler.state_dict() == before["scheduler"]
    assert scaler.state_dict() == before["scaler"] and random.getstate() == before["python"]
    assert torch.equal(torch.get_rng_state(), before["cpu"])
    assert sampler.state_dict() == before["sampler"] and trainer.successful_updates == before["counter"]
    assert metrics == before["metrics"]


def test_live_cache_opportunities_are_valid_token_heads_not_admissions(monkeypatch) -> None:
    from dataclasses import replace
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_training import _measure_live_hybrid_caches
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    module(torch.randn(1, 3, 12), use_cache=True)
    hola = module.last_recurrent_cache.hola_state
    module.last_recurrent_cache = replace(
        module.last_recurrent_cache,
        hola_state=replace(hola, admission_count=torch.ones_like(hola.admission_count)),
    )
    measured = _measure_live_hybrid_caches((module,), valid_token_count=3)
    assert measured["cache_admissions"] == 2
    assert measured["cache_opportunities"] == 6
    assert measured["cache_admission_rate"] == pytest.approx(1 / 3)


def test_live_diagnostic_failure_restores_modes_rng_and_state_before_reporting_metrics_error(monkeypatch) -> None:
    import random
    from types import MappingProxyType
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_training import _run_live_hybrid_diagnostic_pass
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    class Resident(torch.nn.Module):
        def __init__(self): super().__init__(); self.layer = module
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=12).float()
            return self.layer(hidden, attention_mask=attention_mask, use_cache=use_cache)
    model = Resident().train()
    cpu_rng = torch.get_rng_state().clone(); python_rng = random.getstate()
    with pytest.raises(Exception, match="mutable mapping"):
        _run_live_hybrid_diagnostic_pass(
            model=model, modules=(module,),
            batch={"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)},
            metrics=MappingProxyType({"eval_loss": 1.0}),
        )
    assert model.training and module.training
    assert torch.equal(torch.get_rng_state(), cpu_rng) and random.getstate() == python_rng
    assert not hasattr(module, "last_recurrent_cache")


def test_live_diagnostic_optimizer_fingerprint_is_zero_copy_and_detects_mutation(monkeypatch) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_training import _run_live_hybrid_diagnostic_pass
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    parameter = next(module.parameters())
    sentinel = torch.zeros(2_000_000)
    optimizer.state[parameter] = {"step": torch.zeros(()), "exp_avg": sentinel}
    sentinel_id, sentinel_ptr, sentinel_version = id(sentinel), sentinel.data_ptr(), sentinel._version
    mutate = {"enabled": False}
    class Resident(torch.nn.Module):
        def __init__(self): super().__init__(); self.layer = module
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            if mutate["enabled"]:
                optimizer.state[parameter]["exp_avg"].add_(1)
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=12).float()
            return self.layer(hidden, attention_mask=attention_mask, use_cache=use_cache)
    model = Resident()
    kwargs = dict(
        model=model, modules=(module,), optimizer=optimizer,
        batch={"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)},
    )
    _run_live_hybrid_diagnostic_pass(**kwargs)
    actual = optimizer.state[parameter]["exp_avg"]
    assert (id(actual), actual.data_ptr(), actual._version) == (
        sentinel_id, sentinel_ptr, sentinel_version
    )
    mutate["enabled"] = True
    with pytest.raises(Exception, match="optimizer.*mutated"):
        _run_live_hybrid_diagnostic_pass(**kwargs)
    assert optimizer.state[parameter]["exp_avg"] is sentinel
    assert sentinel._version == sentinel_version + 1


def test_live_diagnostic_pass_fails_closed_for_missing_or_heterogeneous_cache(monkeypatch) -> None:
    from gdn3.kmd2_native import KMD2NativeAttn
    from research.kmd2_ablation.qwen_hybrid_four_state import QwenFourStateHybrid
    from research.kmd2_ablation.qwen_hybrid_shared import QwenSharedBraidHybrid
    from research.kmd2_ablation.qwen_training import _measure_live_hybrid_caches
    config = SimpleNamespace(hidden_size=12, linear_num_value_heads=2,
        linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=3,
        linear_conv_kernel_dim=3, rms_norm_eps=1e-6)
    monkeypatch.setenv("GDN3_KMD2_ROUT", "1")
    module = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=0))
    with pytest.raises(Exception, match="missing live cache"):
        _measure_live_hybrid_caches((module,), valid_token_count=1)
    module.scan(torch.randn(1, 1, 12))
    module.last_recurrent_cache = SimpleNamespace(state=torch.zeros(1, 2, 4, 3), hola_state=None)
    with pytest.raises(Exception, match="HOLA"):
        _measure_live_hybrid_caches((module,), valid_token_count=1)
    shared = QwenSharedBraidHybrid.from_native(KMD2NativeAttn(config, layer_idx=1))
    four_config = SimpleNamespace(**(vars(config) | {"linear_key_head_dim": 8}))
    four = QwenFourStateHybrid.from_native(KMD2NativeAttn(four_config, layer_idx=2))
    shared(torch.randn(1, 1, 12), use_cache=True)
    four(torch.randn(1, 1, 12), use_cache=True)
    with pytest.raises(Exception, match="heterogeneous"):
        _measure_live_hybrid_caches((shared, four), valid_token_count=1)


@pytest.mark.parametrize(("task", "targets", "predictions", "modulus"), [
    ("parity", [1, 0, 1], [3, 2, 5], None),
    ("modular", [1, 2, 0], [6, 7, 10], 5),
])
def test_default_evaluator_generates_and_scores_state_tracking(
    task, targets, predictions, modulus
) -> None:
    from research.kmd2_ablation.qwen_backend import LoadedQwenArm
    from research.kmd2_ablation.qwen_training import QwenJobData, _default_evaluate

    class Model(torch.nn.Module):
        def forward(self, input_ids, *, output_hidden_states, use_cache):
            logits = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
            return SimpleNamespace(logits=logits)

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    metadata = {"task": task, "cell_id": f"{task}-32k", "seed": 7,
                "example_id": "state-0", "prompt_end": 3, "targets": targets,
                "modulus": modulus, "evidence_scope": "promotion"}
    batch = {"input_ids": input_ids, "labels": input_ids.clone(),
             "example_ids": ("state-0",), "state_tracking_metadata": (metadata,)}
    data = QwenJobData((batch,), (batch,), {"sha256": "a" * 64})
    job = _qwen_adapter_job("a" * 64)
    job["seed"] = 7
    job["canonical_config"]["task"]["name"] = task
    loaded = LoadedQwenArm(model=Model(), arm="native", job_id="state", upgraded_indices=(),
                           trainable_names=(), assets=())
    result = _default_evaluate(
        loaded_arm=loaded, data=data, job=job, runtime={"student_device": "cpu"},
        tokenizer_asset=SimpleNamespace(path="unused"),
        generate_state_values=lambda **_kwargs: predictions,
    )
    row = result["evaluations"][0]
    assert row["state_task"] == task
    assert row["cell_id"] == f"{task}-32k" and row["seed"] == 7
    assert row["episode_exact"] is True and row["lm_loss"] >= 0


def test_data_parallel_window_shards_are_disjoint_and_reconstruct_pairing() -> None:
    from research.kmd2_ablation.qwen_training import _shard_training_windows
    batch = {"input_ids": torch.arange(12).reshape(4, 3),
             "labels": torch.arange(12).reshape(4, 3),
             "example_ids": ("e0", "e1", "e2", "e3")}
    left, left_ids = _shard_training_windows((batch,), rank=0, world_size=2)
    right, right_ids = _shard_training_windows((batch,), rank=1, world_size=2)
    assert set(left_ids[0]).isdisjoint(right_ids[0])
    assert tuple(sorted(left_ids[0] + right_ids[0])) == tuple(sorted(batch["example_ids"]))
    assert left[0]["input_ids"].shape == right[0]["input_ids"].shape == (2, 3)
    uneven = dict(batch, input_ids=batch["input_ids"][:3], labels=batch["labels"][:3],
                  example_ids=batch["example_ids"][:3])
    with pytest.raises(Exception, match="divide evenly"):
        _shard_training_windows((uneven,), rank=0, world_size=2)


def test_distributed_trainer_accounts_global_tokens_before_budget(monkeypatch) -> None:
    from research.kmd2_ablation.qwen_training import QwenHealTrainer
    model = _HealModel(); optimizer, scheduler = _optimizer_and_scheduler(model)
    def all_reduce(tensor, op):
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    trainer = QwenHealTrainer(
        model=model, teacher=_HealTeacher(), optimizer=optimizer, scheduler=scheduler,
        config=_training_config(accumulation_steps=1, max_tokens=6),
        job_id="global", pairing_id="9" * 64, arm="surprise",
        expected_example_windows=(("local",),), distributed=True,
    )
    trainer.train_update((_batch("local", (0, 1, 2)),))
    assert trainer.tokens_seen == 6


class _GlooHealModel(_HealModel):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if self.fail:
            output.logits = output.logits * torch.tensor(float("nan"))
        return output


def _gloo_trainer_worker(rank: int, init_file: str, output: str) -> None:
    import torch.distributed as dist
    from research.kmd2_ablation.qwen_training import QwenHealTrainer
    dist.init_process_group("gloo", init_method=f"file:///{init_file.replace(chr(92), '/')}",
                            rank=rank, world_size=2)
    try:
        model = _GlooHealModel()
        with torch.no_grad():
            model.cache_amplitude.fill_(2.0)
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        optimizer, scheduler = _optimizer_and_scheduler(model)
        trainer = QwenHealTrainer(
            model=ddp, teacher=_HealTeacher(), optimizer=optimizer, scheduler=scheduler,
            config=_training_config(accumulation_steps=1, max_tokens=6),
            job_id=f"gloo-{rank}", pairing_id="d" * 64, arm="surprise",
            expected_example_windows=((f"rank-{rank}",),), distributed=True,
        )
        trainer.train_update((_batch(f"rank-{rank}", (rank, 1, 2)),))
        gathered = [torch.zeros_like(model.cache_amplitude) for _ in range(2)]
        dist.all_gather(gathered, model.cache_amplitude.detach())

        failing = _GlooHealModel(fail=rank == 0)
        failing_ddp = torch.nn.parallel.DistributedDataParallel(failing)
        fail_optimizer, fail_scheduler = _optimizer_and_scheduler(failing)
        fail_trainer = QwenHealTrainer(
            model=failing_ddp, teacher=_HealTeacher(), optimizer=fail_optimizer,
            scheduler=fail_scheduler,
            config=_training_config(accumulation_steps=1, max_tokens=6),
            job_id=f"fail-{rank}", pairing_id="e" * 64, arm="surprise",
            expected_example_windows=((f"fail-{rank}",),), distributed=True,
        )
        failure = None
        try:
            fail_trainer.train_update((_batch(f"fail-{rank}", (0, 1, 2)),))
        except Exception as error:
            failure = getattr(error, "code", None)
        torch.save({"projected": [float(x) for x in gathered], "failure": failure,
                    "global_tokens": trainer.tokens_seen},
                   Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="this PyTorch build has no supported Windows Gloo device")
def test_real_two_process_gloo_trainer_syncs_projection_and_failure(tmp_path: Path) -> None:
    import torch.multiprocessing as mp
    init_file = str((tmp_path / "gloo-init").resolve())
    mp.spawn(_gloo_trainer_worker, args=(init_file, str(tmp_path)), nprocs=2, join=True)
    rows = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert rows[0]["projected"] == rows[1]["projected"]
    assert 0.0 <= rows[0]["projected"][0] <= 1.0
    assert [row["failure"] for row in rows] == ["nonfinite_loss", "nonfinite_loss"]
    assert [row["global_tokens"] for row in rows] == [6, 6]
