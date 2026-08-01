# Gemma 4 PT-BR no Google Colab

Este diretório contém o caminho de execução recomendado para treinar e avaliar
o modelo numa única GPU do Google Colab — o objetivo final deste projeto.

## Arquivo principal

[`Gemma4_PTBR_Colab.ipynb`](Gemma4_PTBR_Colab.ipynb) — abra no Colab
(`File > Open notebook > GitHub`, cole a URL deste repo, ou faça upload
direto do `.ipynb`) e rode célula a célula. O notebook cobre:

1. Setup do ambiente (dependências certas, sem reinstalar o PyTorch/CUDA do Colab)
2. Continued Pre-Training (CPT) piloto em QLoRA no **Gemma 4 E2B**
   (`google/gemma-4-E2B` — Apache 2.0, não-gated, o menor modelo real da
   família Gemma 4) sobre o corpus **Aurora-PT**
3. Residual merge para recuperar instruction-following sem treino adicional
4. Avaliação num subconjunto de benchmarks reais em português + retenção em
   inglês, com bootstrap/Wilson confidence intervals
5. Dashboard de resultados

## Por que E2B e não E4B/26B?

O restante do projeto (`configs/train/cpt_pilot.yaml`, `cpt_main.yaml`) usa
Gemma 4 E4B/26B-A4B, pensados para 1-4x A100 80GB (GCP). Para uma **única**
GPU de Colab — incluindo o T4 16GB gratuito — o E2B (~2.3B parâmetros
efetivos) em QLoRA cabe com folga e permite terminar um piloto de
demonstração dentro do limite de uma sessão. Para escalar depois de validar o
pipeline aqui, veja `docs/TRAINING_GUIDE.md` e `infra/gcp/` (trilha GCP
multi-GPU) ou simplesmente troque `model_config` no YAML do Colab para
`configs/model/gemma4_e4b.yaml` se sua GPU tiver VRAM sobrando (L4 24GB / A100 40GB).

## Configs específicas do Colab

- [`configs/train/cpt_colab_pilot.yaml`](../configs/train/cpt_colab_pilot.yaml)
  — CPT piloto QLoRA dimensionado para 1 GPU (ver comentários no arquivo
  para o raciocínio por trás de cada hiperparâmetro).
- [`configs/eval/benchmarks_colab.yaml`](../configs/eval/benchmarks_colab.yaml)
  — subconjunto de ~6 benchmarks reais (ENEM, BLUEX, ASSIN2-RTE, HateBR, OAB,
  MMLU-EN) que cabe numa sessão, com `use_vllm: false` e logprob scoring.
- [`configs/model/gemma4_e2b.yaml`](../configs/model/gemma4_e2b.yaml) — config
  do modelo E2B.

## Persistência entre sessões

O Colab **não mantém disco entre sessões**. O notebook empurra checkpoints
para o seu HF Hub via `output.push_to_hub` (configure `HF_TOKEN` nos Colab
Secrets) — para retomar numa nova sessão:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="seu-usuario/seu-repo-de-checkpoint",
    allow_patterns="last-checkpoint/*",
    local_dir="outputs/cpt_colab_pilot",
)
```

e então rode o mesmo comando de treino — `find_latest_checkpoint`
(`src/utils/checkpointing.py`) detecta e retoma automaticamente.

Alternativa: montar o Google Drive e copiar `outputs/`/`reports/` para lá
(célula opcional no notebook).

## Rodando sem o Jupyter/Colab UI

Todas as células do notebook são, na prática, comandos do `gemma4pt` CLI já
documentado no [`README.md`](../README.md) principal. Se preferir rodar via
script (outro provedor de GPU única, uma VM, etc.), o equivalente é:

```bash
pip install -e ".[colab]"
python -m src.cli preflight
python -m src.cli train-cpt configs/train/cpt_colab_pilot.yaml
python -m src.train.residual_merge \
    --base-model google/gemma-4-E2B --instruct-model google/gemma-4-E2B-it \
    --cpt-model outputs/cpt_colab_pilot/final --alpha 1.0 \
    --output-dir outputs/residual_merge
python -m src.cli eval --config configs/eval/benchmarks_colab.yaml
python -m src.cli report
python scripts/build_dashboard.py --format markdown
```
