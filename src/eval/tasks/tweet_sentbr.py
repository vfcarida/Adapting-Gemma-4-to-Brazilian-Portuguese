"""TweetSentBR task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class TweetSentBRTask(BaseTask):
    """TweetSentBR sentiment analysis."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/TweetSentBR was fabricated. Replaced with
        # eduagarcia/tweetsentbr_fewshot (real; train=75 few-shot pool, test=2010).
        hub_id = config.get("hub_id", "eduagarcia/tweetsentbr_fewshot")
        data = self._load_from_hub(hub_id, split="test")

        # Real schema: id, sentence, label — where label is a STRING
        # ("Positive"/"Negative"/"Neutral"), not an int. The previous
        # int-keyed label_map with a silent str(label_val) fallback always
        # missed (since real labels are strings), producing garbage labels
        # like "Positive" (untranslated) instead of "positivo".
        label_map = {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}
        examples = []
        for item in data:
            label_val = item.get("label", item.get("sentiment", ""))
            label_key = str(label_val).strip().lower()
            example = {
                "text": item.get("sentence", item.get("text", item.get("tweet_text", ""))),
                "label": label_map.get(label_key, str(label_val)),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "positiv" in text:
            return "positivo"
        if "negativ" in text:
            return "negativo"
        if "neutr" in text:
            return "neutro"
        return "neutro"
