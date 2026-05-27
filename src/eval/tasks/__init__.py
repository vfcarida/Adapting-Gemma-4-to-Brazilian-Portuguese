# src.eval.tasks — Individual PT-BR benchmark task definitions
#
# Each task module exposes a TaskConfig dataclass and a build_task() factory.
# The benchmark_runner imports tasks dynamically via their module path.

from src.eval.tasks.enem import build_task as build_enem
from src.eval.tasks.bluex import build_task as build_bluex
from src.eval.tasks.assin2_rte import build_task as build_assin2_rte
from src.eval.tasks.assin2_sts import build_task as build_assin2_sts
from src.eval.tasks.hatebr import build_task as build_hatebr
from src.eval.tasks.tweet_sentbr import build_task as build_tweet_sentbr
from src.eval.tasks.oab_bench import build_task as build_oab_bench
from src.eval.tasks.broverbs import build_task as build_broverbs
from src.eval.tasks.copa_pt import build_task as build_copa_pt
from src.eval.tasks.mrpc_pt import build_task as build_mrpc_pt
from src.eval.tasks.rte_pt import build_task as build_rte_pt
from src.eval.tasks.donotanswer_pt import build_task as build_donotanswer_pt
from src.eval.tasks.tuguesice_pt import build_task as build_tuguesice_pt

TASK_REGISTRY: dict[str, callable] = {
    "enem": build_enem,
    "bluex": build_bluex,
    "assin2_rte": build_assin2_rte,
    "assin2_sts": build_assin2_sts,
    "hatebr": build_hatebr,
    "tweet_sentbr": build_tweet_sentbr,
    "oab_bench": build_oab_bench,
    "broverbs": build_broverbs,
    "copa_pt": build_copa_pt,
    "mrpc_pt": build_mrpc_pt,
    "rte_pt": build_rte_pt,
    "donotanswer_pt": build_donotanswer_pt,
    "tuguesice_pt": build_tuguesice_pt,
}

__all__ = ["TASK_REGISTRY"]
