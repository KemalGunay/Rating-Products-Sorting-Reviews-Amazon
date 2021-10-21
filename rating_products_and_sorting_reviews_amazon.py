import pandas as pd
import math
import scipy.stats as st


df = pd.read_csv("amazon_review.csv")
df.head()

# TASK_1
df["overall"].mean()

def time_based_weighted_average(dataframe, w1=30, w2=28, w3= 24, w4=22):
    return df.loc[df["day_diff"] <= 30, "overall"].mean() * w1/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * w2/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * w3/100 + \
    df.loc[df["day_diff"] > 180, "overall"].mean() * w4/100

time_based_weighted_average(df)

# overall 4.58 ortalamaya sahipken, yukarıdaki ağırlaklandırmayla 4.88'e çıkmıştır



# TASK_2
df["helpful_no"]= df["total_vote"]- df["helpful_yes"]
df.sort_values("helpful_no", ascending=False).head(20)

# kullanacağımız tanımlı fonksiyonlar
def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(helpful_yes,helpful_no,confidence=0.95):
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1-(1-confidence)/2)
    phat = 1.0 * helpful_yes/ n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

liste = ["reviewerName","overall","helpful_yes","helpful_no","total_vote","score_pos_neg_diff","score_average_rating","wilson_lower_bound"]
df[liste].sort_values(by="wilson_lower_bound",ascending=False).head(20)
