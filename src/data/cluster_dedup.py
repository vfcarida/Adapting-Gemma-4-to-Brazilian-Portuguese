"""
Deduplicação em nível de cluster e divisão de datasets.

Utiliza MinHash LSH para agrupar documentos similares e garante
que clusters inteiros sejam atribuídos ao mesmo split, prevenindo
vazamento de near-duplicates entre treino e validação.
"""

import hashlib
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

# Tentativa de importar datasketch para MinHash LSH
try:
    from datasketch import MinHash, MinHashLSH

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


class ClusterDedup:
    """
    Sistema de deduplicação baseado em clusters.

    Agrupa documentos similares usando MinHash LSH (ou hash exato como fallback)
    e permite divisão de datasets respeitando fronteiras de cluster para evitar
    vazamento de dados entre splits.
    """

    def __init__(self):
        """Inicializa o sistema de deduplicação."""
        self._cluster_map: Dict[int, int] = {}  # doc_index → cluster_id
        self._clusters: Dict[int, List[int]] = {}  # cluster_id → [doc_indices]
        self._n_duplicates_removed: int = 0
        self._original_size: int = 0

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """
        Extrai k-shingles (substrings de tamanho k) do texto.

        Args:
            text: Texto de entrada.
            k: Tamanho dos shingles.

        Returns:
            Conjunto de shingles únicos.
        """
        text = text.lower().strip()
        if len(text) < k:
            return {text}
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def _build_minhash(self, shingles: Set[str], num_perm: int = 128) -> "MinHash":
        """
        Constrói MinHash a partir de um conjunto de shingles.

        Args:
            shingles: Conjunto de shingles do documento.
            num_perm: Número de permutações para o MinHash.

        Returns:
            Objeto MinHash configurado.
        """
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode("utf-8"))
        return m

    def _exact_hash(self, text: str) -> str:
        """
        Calcula hash exato do texto (fallback quando datasketch indisponível).

        Args:
            text: Texto de entrada.

        Returns:
            Hash MD5 do texto normalizado.
        """
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def build_clusters(
        self,
        texts: List[str],
        threshold: float = 0.8,
        num_perm: int = 128,
    ) -> Dict[int, int]:
        """
        Constrói clusters de documentos similares.

        Se datasketch estiver disponível, usa MinHash LSH para encontrar
        documentos com similaridade de Jaccard acima do threshold.
        Caso contrário, faz fallback para deduplicação por hash exato.

        Args:
            texts: Lista de textos dos documentos.
            threshold: Limiar de similaridade para considerar duplicatas (0.0 a 1.0).
            num_perm: Número de permutações MinHash (mais = mais preciso, mais lento).

        Returns:
            Dicionário mapeando doc_index → cluster_id.
        """
        self._original_size = len(texts)
        self._cluster_map = {}
        self._clusters = {}

        if HAS_DATASKETCH:
            self._build_clusters_minhash(texts, threshold, num_perm)
        else:
            self._build_clusters_exact(texts)

        # Conta duplicatas (documentos em clusters com mais de 1 membro)
        self._n_duplicates_removed = sum(
            len(members) - 1 for members in self._clusters.values() if len(members) > 1
        )

        return self._cluster_map.copy()

    def _build_clusters_minhash(self, texts: List[str], threshold: float, num_perm: int) -> None:
        """
        Constrói clusters usando MinHash LSH.

        Args:
            texts: Lista de textos.
            threshold: Limiar de similaridade Jaccard.
            num_perm: Número de permutações.
        """
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes: List[Optional[MinHash]] = []

        # Fase 1: construir MinHashes e inserir no LSH
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                minhashes.append(None)
                continue

            shingles = self._get_shingles(text)
            mh = self._build_minhash(shingles, num_perm)
            minhashes.append(mh)

            try:
                lsh.insert(str(idx), mh)
            except ValueError:
                # Documento já inserido (hash idêntico) — ignora
                pass

        # Fase 2: consultar LSH para encontrar vizinhos
        visited: Set[int] = set()
        cluster_id = 0

        for idx, mh in enumerate(minhashes):
            if idx in visited or mh is None:
                continue

            # Consulta vizinhos no LSH
            try:
                neighbors = lsh.query(mh)
            except Exception:
                neighbors = [str(idx)]

            # Converte para índices inteiros
            neighbor_indices = []
            for n in neighbors:
                try:
                    n_idx = int(n)
                    if n_idx not in visited:
                        neighbor_indices.append(n_idx)
                except (ValueError, TypeError):
                    continue

            if not neighbor_indices:
                neighbor_indices = [idx]

            # Atribui cluster
            self._clusters[cluster_id] = neighbor_indices
            for n_idx in neighbor_indices:
                self._cluster_map[n_idx] = cluster_id
                visited.add(n_idx)

            cluster_id += 1

        # Documentos não visitados (vazios) recebem cluster individual
        for idx in range(len(texts)):
            if idx not in visited:
                self._cluster_map[idx] = cluster_id
                self._clusters[cluster_id] = [idx]
                cluster_id += 1

    def _build_clusters_exact(self, texts: List[str]) -> None:
        """
        Constrói clusters usando hash exato (fallback sem datasketch).

        Documentos com conteúdo idêntico (após normalização) são agrupados
        no mesmo cluster.

        Args:
            texts: Lista de textos.
        """
        hash_to_cluster: Dict[str, int] = {}
        cluster_id = 0

        for idx, text in enumerate(texts):
            if not text:
                text_hash = f"__empty_{idx}__"
            else:
                text_hash = self._exact_hash(text)

            if text_hash in hash_to_cluster:
                # Documento duplicado — adiciona ao cluster existente
                existing_cluster = hash_to_cluster[text_hash]
                self._cluster_map[idx] = existing_cluster
                self._clusters[existing_cluster].append(idx)
            else:
                # Novo cluster
                hash_to_cluster[text_hash] = cluster_id
                self._cluster_map[idx] = cluster_id
                self._clusters[cluster_id] = [idx]
                cluster_id += 1

    def split_by_clusters(
        self,
        cluster_map: Optional[Dict[int, int]] = None,
        val_ratio: float = 0.005,
        test_ratio: float = 0.0,
        seed: int = 42,
    ) -> Dict[str, List[int]]:
        """
        Divide documentos em splits respeitando fronteiras de cluster.

        Clusters inteiros são atribuídos ao mesmo split, prevenindo
        vazamento de near-duplicates entre treino e validação/teste.

        Args:
            cluster_map: Mapeamento doc_index → cluster_id (usa interno se None).
            val_ratio: Proporção de documentos para validação.
            test_ratio: Proporção de documentos para teste.
            seed: Semente para reprodutibilidade.

        Returns:
            Dicionário com chaves "train", "validation", "test" contendo
            listas de índices de documentos.
        """
        if cluster_map is None:
            cluster_map = self._cluster_map

        if not cluster_map:
            raise ValueError("Nenhum cluster encontrado. Execute build_clusters() primeiro.")

        # Reconstrói mapeamento cluster_id → [doc_indices]
        clusters: Dict[int, List[int]] = defaultdict(list)
        for doc_idx, c_id in cluster_map.items():
            clusters[c_id].append(doc_idx)

        # Ordena clusters por ID para reprodutibilidade
        cluster_ids = sorted(clusters.keys())
        total_docs = sum(len(clusters[c]) for c in cluster_ids)

        # Calcula número alvo de documentos por split
        n_val = max(1, int(total_docs * val_ratio)) if val_ratio > 0 else 0
        n_test = max(1, int(total_docs * test_ratio)) if test_ratio > 0 else 0

        # Embaralha clusters (não documentos individuais)
        rng = random.Random(seed)
        shuffled_clusters = cluster_ids.copy()
        rng.shuffle(shuffled_clusters)

        # Atribui clusters aos splits
        val_indices: List[int] = []
        test_indices: List[int] = []
        train_indices: List[int] = []

        val_count = 0
        test_count = 0

        for c_id in shuffled_clusters:
            members = clusters[c_id]

            if val_count < n_val:
                val_indices.extend(members)
                val_count += len(members)
            elif test_count < n_test:
                test_indices.extend(members)
                test_count += len(members)
            else:
                train_indices.extend(members)

        # Ordena índices dentro de cada split para consistência
        return {
            "train": sorted(train_indices),
            "validation": sorted(val_indices),
            "test": sorted(test_indices),
        }

    def get_dedup_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de deduplicação.

        Returns:
            Dicionário com:
            - n_clusters: número total de clusters
            - n_duplicates_removed: documentos identificados como duplicatas
            - dedup_ratio: proporção de duplicatas removidas
            - n_original: tamanho original do dataset
            - n_unique: número de documentos únicos (representantes de cluster)
            - method: método utilizado ("minhash_lsh" ou "exact_hash")
        """
        n_clusters = len(self._clusters)
        n_original = self._original_size
        n_unique = n_original - self._n_duplicates_removed

        dedup_ratio = self._n_duplicates_removed / n_original if n_original > 0 else 0.0

        method = "minhash_lsh" if HAS_DATASKETCH else "exact_hash"

        return {
            "n_clusters": n_clusters,
            "n_duplicates_removed": self._n_duplicates_removed,
            "dedup_ratio": round(dedup_ratio, 4),
            "n_original": n_original,
            "n_unique": n_unique,
            "method": method,
        }
