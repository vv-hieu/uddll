from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
from pyspark.sql.types import *

from sklearn.metrics import ndcg_score


class RMSEEvaluator:
    def __init__(self, labelCol: str, predictionCol: str) -> None:
        self._evaluator: RegressionEvaluator = RegressionEvaluator(labelCol=labelCol, predictionCol=predictionCol, metricName='rmse')

    def evaluate(self, df: DataFrame) -> float:
        return self._evaluator.evaluate(df)


class NDCGEvaluator:
    def __init__(self, labelCol: str, predictionCol: str, idCol: str, k: int|None=None) -> None:
        self._labelCol     = labelCol
        self._preditionCol = predictionCol
        self._idCol        = idCol
        self._k            = k

    def evaluate(self, df: DataFrame) -> float:
        grouped_df = df.groupBy(self._idCol).agg(
            collect_list(col(self._labelCol)).alias(self._labelCol),
            collect_list(col(self._preditionCol)).alias(self._preditionCol)
        )

        grouped_df = grouped_df.select(self._labelCol, self._preditionCol).toPandas()
        return grouped_df.apply(lambda r: self._ndcg(r[self._labelCol], r[self._preditionCol]), axis=1).mean()

    def _ndcg(self, label, prediction) -> float:
        if len(label) <= 1:
            return 1.0
        return ndcg_score([label], [prediction], k=self._k)
        