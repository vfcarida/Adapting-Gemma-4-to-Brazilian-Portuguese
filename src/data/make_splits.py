"""Criação de splits de dados com garantia de não-vazamento.

Este módulo coordena o pipeline completo de preparação de dados:
1. Carregar corpus bruto (Aurora-PT)
2. Aplicar filtros de qualidade (quality_manifest)
3. Deduplicar e clusterizar (cluster_dedup)
4. Criar splits train/val por cluster (não por documento)
5. Salvar splits finais e estatísticas

O split por clusters garante que documentos near-duplicates fiquem
no MESMO split, prevenindo vazamento de dados entre treino e validação.

Uso:
    python -m src.data.make_splits --config configs/data/aurora_pt.yaml
"""

import json
from pathlib import Path
from typing import Any

from src.utils.config_utils import load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def make_splits(config: dict[str, Any], output_dir: str = "outputs/data_splits") -> dict:
    """Pipeline completo de criação de splits.

    Args:
        config: Config de dados (configs/data/aurora_pt.yaml).
        output_dir: Diretório para salvar splits e estatísticas.

    Returns:
        Dict com estatísticas dos splits criados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = config.get("dataset", {})
    qc_cfg = config.get("quality_control", {})
    dedup_cfg = config.get("deduplication", {})

    # Passo 1: Carregar corpus
    logger.info("Passo 1/5: Carregando corpus...")
    from src.data.aurora_loader import AuroraLoader

    loader = AuroraLoader(config)
    raw_dataset = loader.load_raw()
    logger.info(f"  Corpus bruto: {len(raw_dataset)} documentos")

    # Passo 2: Construir manifesto de qualidade
    stats = {"raw_count": len(raw_dataset)}

    if qc_cfg.get("enabled", False):
        logger.info("Passo 2/5: Construindo manifesto de qualidade...")
        from src.data.quality_manifest import QualityManifest

        manifest = QualityManifest()
        manifest.build_manifest(raw_dataset)

        # Filtrar por qualidade, repassando os valores relevantes do config
        # como parâmetros de filter_by_quality (a API real do método).
        min_score = qc_cfg.get("min_quality_score", 0.3)
        language_cfg = qc_cfg.get("language_id", {})
        target_language = (
            language_cfg.get("target_language", "pt") if language_cfg.get("enabled", True) else None
        )
        toxicity_cfg = qc_cfg.get("toxicity_filter", {})
        max_toxicity = (
            toxicity_cfg.get("max_toxicity_score", 1.0)
            if toxicity_cfg.get("enabled", True)
            else 1.0
        )
        pii_cfg = qc_cfg.get("pii_filter", {})
        # Só exclui documentos com PII se o filtro estiver habilitado E a ação
        # configurada for de fato remover o documento (não apenas "flag").
        exclude_pii = pii_cfg.get("enabled", False) and pii_cfg.get("action") == "remove"

        # filtered_indices são índices no `raw_dataset` original (não posições
        # na lista filtrada) — ver QualityManifest.filter_by_quality.
        filtered_indices = manifest.filter_by_quality(
            min_score=min_score,
            language=target_language,
            max_toxicity=max_toxicity,
            exclude_pii=exclude_pii,
        )
        logger.info(f"  Após filtros de qualidade: {len(filtered_indices)} documentos")
        stats["post_quality_filter"] = len(filtered_indices)

        # Salvar manifesto
        manifest_path = qc_cfg.get("manifest_path", output_dir / "quality_manifest.json")
        manifest.save_manifest(manifest_path)
    else:
        logger.info("Passo 2/5: QC desabilitado, usando todos os documentos")
        filtered_indices = list(range(len(raw_dataset)))
        stats["post_quality_filter"] = len(filtered_indices)

    # Passo 3: Deduplicação e clusterização
    if dedup_cfg.get("cluster_split", False):
        logger.info("Passo 3/5: Deduplicação por clusters...")
        from src.data.cluster_dedup import ClusterDedup

        dedup = ClusterDedup()

        fuzzy_cfg = dedup_cfg.get("fuzzy_dedup", {})
        # `texts` é indexado localmente (posição 0..N-1 na lista), NÃO pelos
        # índices do raw_dataset. build_clusters() retorna um cluster_map
        # cujas chaves são essas posições locais.
        texts = [raw_dataset[i]["text"] for i in filtered_indices]
        local_cluster_map = dedup.build_clusters(
            texts,
            threshold=fuzzy_cfg.get("threshold", 0.8),
            num_perm=fuzzy_cfg.get("num_perm", 128),
        )
        # Traduz posições locais de volta para índices do raw_dataset, para
        # manter um único espaço de índices (raw_dataset) em todo o pipeline
        # (raw_dataset -> filtro de qualidade -> dedup/cluster split).
        cluster_map = {
            filtered_indices[local_idx]: cluster_id
            for local_idx, cluster_id in local_cluster_map.items()
        }
        dedup_stats = dedup.get_dedup_stats()
        stats["dedup"] = dedup_stats
        logger.info(f"  Clusters encontrados: {dedup_stats.get('n_clusters', 'N/A')}")

        # Passo 4: Split por clusters
        logger.info("Passo 4/5: Criando splits por cluster...")
        val_ratio = dataset_cfg.get("val_ratio", 0.005)
        test_ratio = dataset_cfg.get("test_ratio", 0.0)
        seed = dataset_cfg.get("seed", 42)
        split_map = dedup.split_by_clusters(
            cluster_map, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        logger.info("Passo 3/5: Dedup por cluster desabilitado, usando hash split...")
        logger.info("Passo 4/5: Split por hash de documento...")
        # Fallback: split por hash individual
        import hashlib

        val_ratio = dataset_cfg.get("val_ratio", 0.005)

        train_indices = []
        val_indices = []
        for i in filtered_indices:
            text = raw_dataset[i]["text"]
            h = hashlib.md5(text[:500].encode()).hexdigest()
            hash_val = int(h, 16) / (16**32)
            if hash_val < val_ratio:
                val_indices.append(i)
            else:
                train_indices.append(i)

        split_map = {"train": train_indices, "validation": val_indices, "test": []}

    stats["train_count"] = len(split_map["train"])
    stats["val_count"] = len(split_map["validation"])
    stats["test_count"] = len(split_map.get("test", []))
    stats["train_ratio"] = len(split_map["train"]) / max(len(filtered_indices), 1)

    # Passo 5: Salvar splits
    logger.info("Passo 5/5: Salvando splits...")
    splits_path = output_dir / "split_indices.json"
    with open(splits_path, "w") as f:
        json.dump(split_map, f)

    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Train: {stats['train_count']} | Val: {stats['val_count']}")
    logger.info(f"  Splits salvos em: {output_dir}")

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Criar splits de dados com QC e dedup por cluster")
    parser.add_argument("--config", type=str, default="configs/data/aurora_pt.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/data_splits")
    args = parser.parse_args()

    config = load_config(args.config)
    make_splits(config, args.output_dir)


if __name__ == "__main__":
    main()
