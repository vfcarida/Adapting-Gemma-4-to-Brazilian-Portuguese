"""Benchmark task implementations."""

from src.eval.tasks.base_task import BaseTask


def load_task(task_name: str) -> BaseTask:
    """Load a task by name."""
    from src.eval.tasks.assin2_rte import Assin2RTETask
    from src.eval.tasks.assin2_sts import Assin2STSTask
    from src.eval.tasks.bluex import BluexTask
    from src.eval.tasks.boolq_pt import BoolQPT
    from src.eval.tasks.broverbs import BRoverbsTask
    from src.eval.tasks.capitu import Capitu
    from src.eval.tasks.copa_pt import CopaPTTask
    from src.eval.tasks.donotanswer_pt import DoNotAnswerPTTask
    from src.eval.tasks.enem import EnemTask
    from src.eval.tasks.hatebr import HateBRTask
    from src.eval.tasks.legalbench_br import LegalBenchBR
    from src.eval.tasks.lener_br import LeNERBr
    from src.eval.tasks.math_pt import MathPT
    from src.eval.tasks.mrpc_pt import MRPCPTTask
    from src.eval.tasks.oab_bench import OABBenchTask
    from src.eval.tasks.publichearing_br import PublicHearingBR
    from src.eval.tasks.retention_en import ARCEn, HellaSwagEn, MMLUEn
    from src.eval.tasks.rte_pt import RTEPTTask
    from src.eval.tasks.tuguesice_pt import TuguesicePTTask
    from src.eval.tasks.tweet_sentbr import TweetSentBRTask
    from src.eval.tasks.xlsum_pt import XLSumPTTask

    TASK_REGISTRY = {
        "enem": EnemTask,
        "bluex": BluexTask,
        "assin2_rte": Assin2RTETask,
        "assin2_sts": Assin2STSTask,
        "hatebr": HateBRTask,
        "tweet_sentbr": TweetSentBRTask,
        "oab_bench": OABBenchTask,
        "broverbs": BRoverbsTask,
        "copa_pt": CopaPTTask,
        "mrpc_pt": MRPCPTTask,
        "rte_pt": RTEPTTask,
        "donotanswer_pt": DoNotAnswerPTTask,
        "tuguesice_pt": TuguesicePTTask,
        "xlsum_pt": XLSumPTTask,
        "boolq_pt": BoolQPT,
        "lener_br": LeNERBr,
        "legalbench_br": LegalBenchBR,
        "capitu": Capitu,
        "math_pt": MathPT,
        "publichearing_br": PublicHearingBR,
        "mmlu_en": MMLUEn,
        "hellaswag_en": HellaSwagEn,
        "arc_en": ARCEn,
    }

    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")

    return TASK_REGISTRY[task_name]()
