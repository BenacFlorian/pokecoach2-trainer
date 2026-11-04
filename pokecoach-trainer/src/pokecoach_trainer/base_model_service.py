"""Registre local des modèles de base (parents SB3) pour le fine-tuning."""

from __future__ import annotations

import shutil
from pathlib import Path

from stable_baselines3 import PPO

from pokecoach_trainer.compatibility import build_contract, contracts_compatible
from pokecoach_trainer.errors import ContractMismatch, InvalidConfiguration, NotFound
from pokecoach_trainer.schemas import (
    BaseModelManifest,
    BaseModelSource,
    DatasetManifest,
    ModelConfig,
)
from pokecoach_trainer.storage import (
    base_model_dir,
    read_json,
    resolve_inbox_relative,
    sha256_file,
    validate_id,
    write_json_atomic,
)
from pokecoach_trainer.upstream import vgc_bench_workdir


def describe_policy(policy) -> ModelConfig:
    """Extrait l'architecture effective d'une policy VGC-Bench chargée."""

    from vgc_bench.src.policy import MaskedActorCriticPolicy

    if not isinstance(policy, MaskedActorCriticPolicy):
        raise TypeError(f"policy incompatible: {type(policy).__name__}")
    pi = policy.pi_features_extractor
    return ModelConfig(
        d_model=int(policy.action_net.in_features),
        num_layers=int(pi.embed_layers),
        num_heads=int(pi.num_heads),
        embed_len=int(pi.embed_len),
        choose_on_teampreview=bool(policy.choose_on_teampreview),
    )


def _manifest_path(base_model_id: str) -> Path:
    return base_model_dir(base_model_id) / "manifest.json"


def read_base_model_manifest(base_model_id: str) -> BaseModelManifest:
    """Lit le manifeste enregistré ou lève `NotFound`."""

    path = _manifest_path(validate_id(base_model_id))
    if not path.is_file():
        raise NotFound(f"modèle de base inconnu: {base_model_id}")
    return BaseModelManifest.model_validate(read_json(path))


def _verify_checkpoint_file(path: Path, expected_sha256: str) -> str:
    digest = sha256_file(path)
    if digest.lower() != expected_sha256.lower():
        raise InvalidConfiguration(
            "hash checkpoint divergent de la provenance de confiance",
            details={"expected": expected_sha256.lower(), "actual": digest.lower()},
        )
    return digest


def _load_ppo_from_path(path: Path, device: str) -> PPO:
    """Charge SB3 après contrôle d'existence ; erreur si classe ou format invalide."""

    with vgc_bench_workdir():
        from vgc_bench.src.policy import MaskedActorCriticPolicy

        loaded = PPO.load(str(path), device=device)
        if not isinstance(loaded.policy, MaskedActorCriticPolicy):
            raise InvalidConfiguration(
                "classe de policy inattendue",
                details={"policy_class": type(loaded.policy).__name__},
            )
    return loaded


def _state_dict_identical(left: dict, right: dict) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        if left[key].shape != right[key].shape:
            return False
        if not left[key].equal(right[key]):
            return False
    return True


def register_base_model(relative_path: str, source: BaseModelSource) -> BaseModelManifest:
    """
    Copie un checkpoint depuis l'inbox vers `base_models/` après contrôle du hash.

    Ne modifie pas le fichier source. Refuse un ID déjà enregistré.
    """

    validate_id(source.base_model_id)
    inbox_path = resolve_inbox_relative(relative_path)
    if not inbox_path.is_file():
        raise FileNotFoundError(f"checkpoint inbox introuvable: {relative_path}")
    digest = _verify_checkpoint_file(inbox_path, source.expected_sha256)

    target_dir = base_model_dir(source.base_model_id)
    if target_dir.exists():
        raise InvalidConfiguration(
            "modèle de base déjà enregistré",
            details={"base_model_id": source.base_model_id},
        )
    target_dir.mkdir(parents=True)
    dest = target_dir / "model.zip"
    shutil.copy2(inbox_path, dest)

    loaded = _load_ppo_from_path(dest, "cpu")
    architecture = describe_policy(loaded.policy)
    contract = build_contract(architecture)
    param_count = sum(p.numel() for p in loaded.policy.parameters())

    manifest = BaseModelManifest(
        base_model_id=source.base_model_id,
        source=source,
        checkpoint_sha256=digest,
        architecture=architecture,
        contract_fingerprint=contract.fingerprint,
        param_count=param_count,
        policy_class=type(loaded.policy).__name__,
    )
    write_json_atomic(_manifest_path(source.base_model_id), manifest.model_dump(mode="json"))
    del loaded
    return manifest


def load_base_model(base_model_id: str, device: str) -> PPO:
    """
    Charge le parent enregistré après vérification hash, manifeste et poids.

    Compare les tensors au fichier sur disque avant tout entraînement.
    """

    manifest = read_base_model_manifest(base_model_id)
    ckpt_path = base_model_dir(base_model_id) / manifest.checkpoint_relpath
    if not ckpt_path.is_file():
        raise NotFound(f"checkpoint parent absent: {base_model_id}")

    digest = sha256_file(ckpt_path)
    if digest.lower() != manifest.checkpoint_sha256.lower():
        raise InvalidConfiguration("hash du registre parent divergent")

    reference = _load_ppo_from_path(ckpt_path, "cpu")
    reference_sd = {k: v.detach().clone() for k, v in reference.policy.state_dict().items()}
    del reference

    loaded = _load_ppo_from_path(ckpt_path, device)
    architecture = describe_policy(loaded.policy)
    if architecture.model_dump() != manifest.architecture.model_dump():
        raise ContractMismatch(
            "architecture constatée différente du manifeste parent",
            details={
                "manifest": manifest.architecture.model_dump(),
                "loaded": architecture.model_dump(),
            },
        )
    if not _state_dict_identical(reference_sd, loaded.policy.state_dict()):
        raise InvalidConfiguration("poids chargés divergents du fichier source")

    contract = build_contract(architecture)
    if contract.fingerprint != manifest.contract_fingerprint:
        raise ContractMismatch(
            "empreinte de contrat parent obsolète",
            details={
                "manifest": manifest.contract_fingerprint,
                "computed": contract.fingerprint,
            },
        )
    return loaded


def assert_dataset_contract_compatible(
    manifest: BaseModelManifest,
    dataset: DatasetManifest,
) -> None:
    """Refuse un fine-tuning si le contrat dataset diverge du parent."""

    parent_contract = build_contract(manifest.architecture)
    if dataset.encoding_contract_fingerprint is None:
        return
    if dataset.encoding_contract_fingerprint != parent_contract.fingerprint:
        raise ContractMismatch(
            "contrat dataset incompatible avec le parent",
            details={
                "parent": parent_contract.fingerprint,
                "dataset": dataset.encoding_contract_fingerprint,
            },
        )
    if not contracts_compatible(parent_contract, build_contract(manifest.architecture)):
        raise ContractMismatch("contrat parent incohérent")


def assert_architecture_matches(model: ModelConfig, expected: ModelConfig) -> None:
    """Compare deux architectures ; dimensions égales mais champs différents ⇒ refus."""

    if model.model_dump() != expected.model_dump():
        raise ContractMismatch(
            "architecture incompatible (tables ou capacité différentes)",
            details={"expected": expected.model_dump(), "actual": model.model_dump()},
        )
