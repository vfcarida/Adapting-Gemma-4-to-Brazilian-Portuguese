"""
Sistema de manifesto de qualidade para datasets.

Rastreia sinais de qualidade por documento, incluindo detecção de idioma,
PII, toxicidade e score de qualidade geral.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Tentativa de importar dependências opcionais
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    from datasets import Dataset  # noqa: F401
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Stopwords comuns em português para detecção de idioma
_PT_STOPWORDS: Set[str] = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais",
    "as", "dos", "como", "mas", "foi", "ao", "ele", "das", "tem",
    "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos",
    "já", "está", "eu", "também", "só", "pelo", "pela", "até",
    "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos",
    "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão",
    "você", "tinha", "foram", "essa", "num", "nem", "suas", "meu",
    "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja",
    "qual", "será", "nós", "tenho", "lhe", "deles", "essas", "esses",
    "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos",
    "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso",
    "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes",
    "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
}

# Stopwords em inglês para comparação
_EN_STOPWORDS: Set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
}

# Palavras-chave indicativas de toxicidade (lista mínima para heurística)
# TODO: substituir por classificador treinado (e.g., detoxify, perspectiveAPI)
_TOXIC_KEYWORDS: Set[str] = {
    "idiota", "imbecil", "burro", "estúpido", "lixo", "nojento",
    "merda", "porra", "caralho", "puta", "fdp", "arrombado",
    "desgraçado", "vagabundo", "cretino", "otário", "babaca",
}

# Padrões regex para detecção de PII
_PII_PATTERNS = {
    "cpf": re.compile(r"\b\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-/]?\d{2}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "telefone": re.compile(
        r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-.\s]?\d{4}\b"
    ),
    "cartao_credito": re.compile(r"\b\d{4}[\s.-]?\d{4}[\s.-]?\d{4}[\s.-]?\d{4}\b"),
}


@dataclass
class DocumentQuality:
    """Registro de qualidade para um único documento."""

    doc_id: str
    n_chars: int
    n_words: int
    n_sentences: int
    language: str  # "pt", "en", "outro"
    domain: str
    quality_score: float  # 0.0 a 1.0
    has_pii: bool
    toxicity_score: float  # 0.0 a 1.0
    cluster_id: Optional[int] = None
    is_duplicate: bool = False


class QualityManifest:
    """
    Manifesto de qualidade que rastreia sinais por documento.

    Processa datasets HuggingFace e gera métricas de qualidade
    incluindo detecção de idioma, PII, toxicidade e score geral.
    """

    def __init__(self):
        """Inicializa o manifesto vazio."""
        self.records: List[DocumentQuality] = []
        self._stats_cache: Optional[Dict[str, Any]] = None

    def _detect_language(self, text: str) -> str:
        """
        Detecta o idioma do texto usando heurística de stopwords.

        TODO: integrar fasttext (lid.176.bin) para detecção mais robusta.
        O modelo fasttext oferece detecção de 176 idiomas com alta precisão.

        Args:
            text: Texto a ser analisado.

        Returns:
            Código do idioma: "pt", "en" ou "outro".
        """
        words = set(text.lower().split())
        if not words:
            return "outro"

        pt_count = len(words & _PT_STOPWORDS)
        en_count = len(words & _EN_STOPWORDS)

        # Proporção mínima para considerar detecção válida
        total_words = len(words)
        pt_ratio = pt_count / total_words if total_words > 0 else 0
        en_ratio = en_count / total_words if total_words > 0 else 0

        if pt_ratio > en_ratio and pt_ratio > 0.05:
            return "pt"
        elif en_ratio > pt_ratio and en_ratio > 0.05:
            return "en"
        return "outro"

    def _detect_pii(self, text: str) -> bool:
        """
        Detecta presença de PII (informações pessoais identificáveis).

        Verifica padrões de CPF, email, telefone e cartão de crédito.

        Args:
            text: Texto a ser analisado.

        Returns:
            True se PII detectado, False caso contrário.
        """
        for pattern in _PII_PATTERNS.values():
            if pattern.search(text):
                return True
        return False

    def _compute_toxicity(self, text: str) -> float:
        """
        Calcula score de toxicidade baseado em heurística de palavras-chave.

        TODO: substituir por classificador neural (e.g., detoxify ou modelo
        fine-tuned para português) para detecção mais precisa e contextual.

        Args:
            text: Texto a ser analisado.

        Returns:
            Score entre 0.0 (não tóxico) e 1.0 (muito tóxico).
        """
        words = text.lower().split()
        if not words:
            return 0.0

        toxic_count = sum(1 for w in words if w.strip(".,!?;:") in _TOXIC_KEYWORDS)
        # Normaliza pelo número de palavras, com cap em 1.0
        raw_score = toxic_count / len(words) * 10  # amplifica sinal
        return min(raw_score, 1.0)

    def _compute_quality_score(self, text: str, n_words: int, n_sentences: int) -> float:
        """
        Calcula score de qualidade baseado em múltiplas heurísticas.

        Fatores considerados:
        - Comprimento do texto (textos muito curtos ou muito longos penalizados)
        - Proporção palavras/sentenças (legibilidade)
        - Proporção de caracteres especiais
        - Proporção de letras maiúsculas

        Args:
            text: Texto original.
            n_words: Número de palavras.
            n_sentences: Número de sentenças.

        Returns:
            Score entre 0.0 e 1.0.
        """
        if not text or n_words == 0:
            return 0.0

        score = 0.0

        # Fator 1: comprimento adequado (textos entre 50 e 5000 palavras são ideais)
        if n_words < 10:
            length_score = 0.1
        elif n_words < 50:
            length_score = 0.4
        elif n_words <= 5000:
            length_score = 1.0
        else:
            length_score = 0.7  # textos muito longos podem ter ruído
        score += length_score * 0.3

        # Fator 2: proporção palavras por sentença (ideal: 10-25)
        if n_sentences > 0:
            words_per_sentence = n_words / n_sentences
            if 10 <= words_per_sentence <= 25:
                sentence_score = 1.0
            elif 5 <= words_per_sentence < 10 or 25 < words_per_sentence <= 40:
                sentence_score = 0.6
            else:
                sentence_score = 0.3
        else:
            sentence_score = 0.2
        score += sentence_score * 0.3

        # Fator 3: proporção de caracteres especiais (menos é melhor)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text) if len(text) > 0 else 0
        if special_ratio < 0.1:
            special_score = 1.0
        elif special_ratio < 0.2:
            special_score = 0.7
        elif special_ratio < 0.4:
            special_score = 0.4
        else:
            special_score = 0.1
        score += special_score * 0.2

        # Fator 4: proporção de maiúsculas (textos ALL CAPS penalizados)
        upper_count = sum(1 for c in text if c.isupper())
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count > 0:
            upper_ratio = upper_count / alpha_count
            if upper_ratio < 0.3:
                caps_score = 1.0
            elif upper_ratio < 0.5:
                caps_score = 0.5
            else:
                caps_score = 0.1
        else:
            caps_score = 0.5
        score += caps_score * 0.2

        return round(min(max(score, 0.0), 1.0), 4)

    def _count_sentences(self, text: str) -> int:
        """Conta sentenças usando pontuação final."""
        # Heurística simples: conta terminadores de sentença
        terminators = re.findall(r"[.!?]+", text)
        return max(len(terminators), 1) if text.strip() else 0

    def _generate_doc_id(self, text: str, index: int) -> str:
        """Gera ID único para documento baseado em hash do conteúdo."""
        content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return f"doc_{index:06d}_{content_hash}"

    def build_manifest(
        self,
        dataset: Any,
        output_path: Optional[str] = None,
        text_column: str = "text",
        domain_column: Optional[str] = None,
        default_domain: str = "geral",
    ) -> "QualityManifest":
        """
        Processa um dataset HuggingFace e constrói o manifesto de qualidade.

        Args:
            dataset: Dataset HuggingFace (ou qualquer iterável com campo de texto).
            output_path: Caminho para salvar o manifesto (opcional).
            text_column: Nome da coluna de texto no dataset.
            domain_column: Nome da coluna de domínio (opcional).
            default_domain: Domínio padrão quando não especificado.

        Returns:
            Self (para encadeamento de chamadas).
        """
        self.records = []
        self._stats_cache = None

        for idx, example in enumerate(dataset):
            # Extrai texto do exemplo
            if isinstance(example, dict):
                text = example.get(text_column, "")
                domain = example.get(domain_column, default_domain) if domain_column else default_domain
            elif isinstance(example, str):
                text = example
                domain = default_domain
            else:
                text = str(example)
                domain = default_domain

            if not text:
                continue

            n_chars = len(text)
            n_words = len(text.split())
            n_sentences = self._count_sentences(text)
            language = self._detect_language(text)
            has_pii = self._detect_pii(text)
            toxicity_score = self._compute_toxicity(text)
            quality_score = self._compute_quality_score(text, n_words, n_sentences)
            doc_id = self._generate_doc_id(text, idx)

            record = DocumentQuality(
                doc_id=doc_id,
                n_chars=n_chars,
                n_words=n_words,
                n_sentences=n_sentences,
                language=language,
                domain=domain,
                quality_score=quality_score,
                has_pii=has_pii,
                toxicity_score=toxicity_score,
                cluster_id=None,
                is_duplicate=False,
            )
            self.records.append(record)

        if output_path:
            self.save_manifest(output_path)

        return self

    def filter_by_quality(
        self,
        min_score: float = 0.0,
        language: Optional[str] = None,
        max_toxicity: float = 1.0,
        exclude_pii: bool = False,
        exclude_duplicates: bool = False,
    ) -> List[int]:
        """
        Filtra documentos por critérios de qualidade e retorna índices válidos.

        Args:
            min_score: Score mínimo de qualidade (0.0 a 1.0).
            language: Filtrar por idioma específico (e.g., "pt").
            max_toxicity: Score máximo de toxicidade permitido.
            exclude_pii: Se True, exclui documentos com PII.
            exclude_duplicates: Se True, exclui documentos marcados como duplicados.

        Returns:
            Lista de índices dos documentos que passaram nos filtros.
        """
        filtered_indices = []

        for idx, record in enumerate(self.records):
            # Aplica filtros
            if record.quality_score < min_score:
                continue
            if language and record.language != language:
                continue
            if record.toxicity_score > max_toxicity:
                continue
            if exclude_pii and record.has_pii:
                continue
            if exclude_duplicates and record.is_duplicate:
                continue

            filtered_indices.append(idx)

        return filtered_indices

    def save_manifest(self, path: str) -> None:
        """
        Salva o manifesto em disco (parquet se disponível, senão JSON).

        Args:
            path: Caminho do arquivo de saída.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records_dicts = [asdict(r) for r in self.records]

        if HAS_PARQUET and output_path.suffix == ".parquet":
            table = pa.Table.from_pylist(records_dicts)
            pq.write_table(table, str(output_path))
        else:
            # Fallback para JSON
            json_path = output_path.with_suffix(".json") if output_path.suffix == ".parquet" else output_path
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records_dicts, f, ensure_ascii=False, indent=2)

    def load_manifest(self, path: str) -> "QualityManifest":
        """
        Carrega um manifesto salvo anteriormente.

        Args:
            path: Caminho do arquivo do manifesto.

        Returns:
            Self (para encadeamento de chamadas).
        """
        input_path = Path(path)

        if HAS_PARQUET and input_path.suffix == ".parquet":
            table = pq.read_table(str(input_path))
            records_dicts = table.to_pylist()
        elif input_path.suffix == ".json":
            with open(input_path, "r", encoding="utf-8") as f:
                records_dicts = json.load(f)
        else:
            # Tenta JSON como fallback
            json_path = input_path.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    records_dicts = json.load(f)
            else:
                raise FileNotFoundError(
                    f"Manifesto não encontrado em {input_path} nem em {json_path}"
                )

        self.records = [DocumentQuality(**r) for r in records_dicts]
        self._stats_cache = None
        return self

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calcula estatísticas agregadas por domínio.

        Returns:
            Dicionário com estatísticas por domínio incluindo:
            - count: número de documentos
            - avg_quality: score médio de qualidade
            - avg_length: comprimento médio (em palavras)
            - pii_count: número de documentos com PII
            - avg_toxicity: toxicidade média
            - language_distribution: distribuição de idiomas
        """
        if self._stats_cache is not None:
            return self._stats_cache

        if not self.records:
            return {}

        # Agrupa por domínio
        domain_groups: Dict[str, List[DocumentQuality]] = {}
        for record in self.records:
            if record.domain not in domain_groups:
                domain_groups[record.domain] = []
            domain_groups[record.domain].append(record)

        stats: Dict[str, Any] = {}
        for domain, records in domain_groups.items():
            n = len(records)
            avg_quality = sum(r.quality_score for r in records) / n
            avg_length = sum(r.n_words for r in records) / n
            pii_count = sum(1 for r in records if r.has_pii)
            avg_toxicity = sum(r.toxicity_score for r in records) / n

            # Distribuição de idiomas
            lang_dist: Dict[str, int] = {}
            for r in records:
                lang_dist[r.language] = lang_dist.get(r.language, 0) + 1

            stats[domain] = {
                "count": n,
                "avg_quality": round(avg_quality, 4),
                "avg_length": round(avg_length, 2),
                "pii_count": pii_count,
                "avg_toxicity": round(avg_toxicity, 4),
                "language_distribution": lang_dist,
            }

        # Estatísticas globais
        total = len(self.records)
        stats["_global"] = {
            "total_documents": total,
            "avg_quality": round(
                sum(r.quality_score for r in self.records) / total, 4
            ),
            "avg_length": round(
                sum(r.n_words for r in self.records) / total, 2
            ),
            "total_pii": sum(1 for r in self.records if r.has_pii),
            "total_duplicates": sum(1 for r in self.records if r.is_duplicate),
            "domains": list(domain_groups.keys()),
        }

        self._stats_cache = stats
        return stats
