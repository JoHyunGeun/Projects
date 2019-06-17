from surprise import accuracy
from collections import defaultdict


class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def GetTopN(predictions, n=10, minimumRating=0):
        topN = defaultdict(list)
        for ruid, riid, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[ruid].append((riid, estimatedRating))

        for ruid, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[ruid] = ratings[:n]

        return topN

    def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutItemID = leftOut[1]
            hit = False
            for itemID, predictedRating in topNPredicted[userID]:
                if leftOutItemID == itemID:
                    hit = True
                    break
            if hit:
                hits += 1

            total += 1
        return hits/total

    def RankBar(AllPredicted, TestSet):
        sum_rating_rank = 0
        sum_rating = 0

        for ruid, riid, rating in TestSet:
            sum_rating += rating
            anti_num = len(AllPredicted[ruid])
            rank = 0
            for i, item_rating in enumerate(AllPredicted[ruid]):
                if riid == item_rating[0]:
                    rank = i
                    break
            rank_ui = rank / anti_num
            sum_rating_rank += (rating * rank_ui)
        return sum_rating_rank / sum_rating