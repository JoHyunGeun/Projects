from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from .EvaluationData import EvaluationData
from .RecommenderMetrics import RecommenderMetrics
import heapq
from collections import defaultdict
from operator import itemgetter


class SimpleCF:

    def __init__(self, df, user_based=False):
        self.df = df
        self.user_based = user_based

        reader = Reader(line_format='user item rating')
        data = Dataset.load_from_df(df=self.df, reader=reader)
        self.eval_data = EvaluationData(data)

        sim_options = {'name': 'cosine', 'user_based': self.user_based}
        self.model = KNNBasic(sim_options=sim_options)


    def item_based_cf(self, k=10, eval=False):
        topN = defaultdict(list)
        testSet = self.eval_data.GetLOOCVTestSet()

        if eval == False:
            trainSet = self.eval_data.GetFullTrainSet()
        else:
            trainSet = self.eval_data.GetLOOCVTrainSet()

        self.model.fit(trainSet)
        simsMatrix = self.model.compute_similarities()

        for uiid in range(trainSet.n_users()):

            testUserRatings = trainSet.ur[uiid]
            KNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t:t[1])

            candidates = defaultdict(float)
            for itemID, rating in KNeighbors:
                similarityRow = simsMatrix[itemID]
                for item_innerID, item_score in enumerate(similarityRow):
                    candidates[item_innerID] += item_score * (rating / 5.0)

            watched = {}
            for itemID, rating in trainSet.ur[uiid]:
                watched[itemID] = 1

            pos = 0
            for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
                if not itemID in watched:
                    topN[trainSet.to_raw_uid(uiid)].append((trainSet.to_raw_iid(itemID), ratingSum))
                    pos += 1
                    if pos > 10:
                        break
        if eval == False:
            return topN
        else:
            return RecommenderMetrics.HitRate(topN, testSet)


    def user_based_cf(self, k=10, eval=True):
        topN = defaultdict(list)
        testSet = self.eval_data.GetLOOCVTestSet()

        if eval == False:
            trainSet = self.eval_data.GetFullTrainSet()
        else:
            trainSet = self.eval_data.GetLOOCVTrainSet()

        self.model.fit(trainSet)
        simsMatrix = self.model.compute_similarities()

        for uiid in range(trainSet.n_users):

            similarityRow = simsMatrix[uiid]
            similarUsers = []
            for innerID, score in enumerate(similarityRow):
                if (innerID != uiid):
                    similarUsers.append((innerID, score))

            KNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
            candidates = defaultdict(float)
            for similarUser in KNeighbors:
                innerID = similarUser[0]
                userSimilarityScore = similarUser[1]
                theirRatings = trainSet.ur[innerID]
                for rating in theirRatings:
                    candidates[rating[0]] += userSimilarityScore * (rating[1] / 5.0)

            watched = {}
            for itemID, rating in trainSet.ur[uiid]:
                watched[itemID] = 1

            pos = 0
            for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
                if not itemID in watched:
                    topN[trainSet.to_raw_uid(uiid)].append((trainSet.to_raw_iid(itemID), ratingSum))
                    pos += 1
                    if pos > 10:
                        break
        if eval == False:
            return topN
        else:
            return RecommenderMetrics.HitRate(topN, testSet)




