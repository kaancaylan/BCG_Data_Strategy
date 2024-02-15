import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def dataloader(path: Path) -> pd.DataFrame:
    """Loads the data located in the path.

    Parameters
    -------
    path_ : Path
            Path of the data.

    Returns
    -------
    data : pd.DataFrame
            Data as a dataframe.
    """
    df = pd.read_csv(path, sep=";")
    df = df.dropna()
    one_hot = pd.get_dummies(df["order_channel"], dtype=int)
    df = df.drop("order_channel", axis=1)
    df = df.join(one_hot)
    return df


def daily_preprocessing(
    df: pd.DataFrame, drop_time: int, churn_time: int
) -> Tuple[pd.DataFrame]:
    """Groups the data per day and user, creating two dfs, one for computing user features, other for calculating churn.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe which we will group by day and user.
    drop_time : int
        Number of most recent days we will drop from the dataset.
    churn_time : int
        Number of days we will use to create the churn df(to calculate churn).

    Returns:
    --------
    daily_agg : pd.DataFrame
        Dataframe grouped by day and client which will be used to compute client variables.
    churn_daily_agg : pd.DataFrame
        Dataframe grouped by day and client which will be used to compute churn.
    """
    daily_agg = df.groupby(["date_order", "client_id"], as_index=False).agg(
        day_purchases=pd.NamedAgg(column="sales_net", aggfunc="count"),
        day_sales=pd.NamedAgg(column="sales_net", aggfunc="sum"),
        day_products=pd.NamedAgg(column="product_id", aggfunc="nunique"),
        day_quantity=pd.NamedAgg(column="quantity", aggfunc="sum"),
        day_branches=pd.NamedAgg(column="branch_id", aggfunc="count"),
        day_online=pd.NamedAgg(column="online", aggfunc="sum"),
        day_store=pd.NamedAgg(column="at the store", aggfunc="sum"),
        day_sales_rep=pd.NamedAgg(
            column="during the visit of a sales rep", aggfunc="sum"
        ),
        day_phone=pd.NamedAgg(column="by phone", aggfunc="sum"),
        day_other=pd.NamedAgg(column="other", aggfunc="sum"),
    )

    daily_agg["last_date_order"] = daily_agg.groupby("client_id")["date_order"].shift()
    daily_agg["last_date_order"] = np.where(
        daily_agg["last_date_order"].isna(),
        daily_agg["date_order"],
        daily_agg["last_date_order"],
    )
    daily_agg["last_date_order"] = pd.to_datetime(daily_agg["last_date_order"])
    daily_agg["date_order"] = pd.to_datetime(daily_agg["date_order"])
    daily_agg["Day_Difference"] = (
        daily_agg["date_order"] - daily_agg["last_date_order"]
    ).dt.days

    max_date = daily_agg["date_order"].max()
    daily_agg = daily_agg[(max_date - daily_agg["date_order"]).dt.days >= drop_time]
    max_date = daily_agg["date_order"].max()
    churn_daily_agg = daily_agg[
        (max_date - daily_agg["date_order"]).dt.days < churn_time
    ]
    daily_agg = daily_agg[(max_date - daily_agg["date_order"]).dt.days >= churn_time]

    return daily_agg, churn_daily_agg


def client_preprocessing(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Groups the daily data by client.

    Parameters:
    -----------
    daily_df : pd.DataFrame
        Dataframe grouped by day and user

    Returns:
    --------
    client_agg : pd.DataFrame
        Dataframe grouped by client.
    """
    client_agg = daily_df.groupby("client_id").agg(
        client_total_days=pd.NamedAgg(column="day_purchases", aggfunc="count"),
        client_avg_purchases=pd.NamedAgg(column="day_purchases", aggfunc="mean"),
        client_avg_sales=pd.NamedAgg(column="day_sales", aggfunc="mean"),
        client_avg_products=pd.NamedAgg(column="day_products", aggfunc="mean"),
        client_avg_branches=pd.NamedAgg(column="day_quantity", aggfunc="mean"),
        client_online=pd.NamedAgg(column="day_online", aggfunc="sum"),
        client_store=pd.NamedAgg(column="day_store", aggfunc="sum"),
        client_sales_rep=pd.NamedAgg(column="day_sales_rep", aggfunc="sum"),
        client_phone=pd.NamedAgg(column="day_phone", aggfunc="sum"),
        client_other=pd.NamedAgg(column="day_other", aggfunc="sum"),
        client_avg_day_diff=pd.NamedAgg(column="Day_Difference", aggfunc="mean"),
        client_first_buy=pd.NamedAgg(column="date_order", aggfunc="min"),
        client_last_buy=pd.NamedAgg(column="date_order", aggfunc="max"),
    )
    client_agg["total_purchases"] = (
        client_agg["client_online"]
        + client_agg["client_other"]
        + client_agg["client_phone"]
        + client_agg["client_sales_rep"]
        + client_agg["client_store"]
    )
    client_agg["client_online"] = (
        client_agg["client_online"] / client_agg["total_purchases"]
    )
    client_agg["client_other"] = (
        client_agg["client_other"] / client_agg["total_purchases"]
    )
    client_agg["client_phone"] = (
        client_agg["client_phone"] / client_agg["total_purchases"]
    )
    client_agg["client_sales_rep"] = (
        client_agg["client_sales_rep"] / client_agg["total_purchases"]
    )
    client_agg["client_store"] = (
        client_agg["client_store"] / client_agg["total_purchases"]
    )
    client_agg = client_agg.drop(["total_purchases"], axis=1)
    return client_agg


def recent_client_preprocessing(daily_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Does client aggregation for only a recent segment of the data.

    Parameters:
    -----------
    daily_df: pd.DataFrame
        The dataframe for which we will perform client aggregation.
    threshold: int
        Amount of days we want to use for our recency analysis.

    Returns:
    --------
    recent_client_df: pd.DataFrame
        Recent dataframe grouped by client.
    """
    max_date = daily_df["date_order"].max()
    recent_df = daily_df[(max_date - daily_df["date_order"]).dt.days <= threshold]
    recent_client_df = client_preprocessing(recent_df)
    recent_client_df = recent_client_df[
        [
            "client_avg_purchases",
            "client_avg_sales",
            "client_avg_products",
            "client_avg_branches",
            "client_avg_day_diff",
        ]
    ]
    recent_client_df = recent_client_df.rename(
        {
            "client_avg_purchases": "recent_avg_purchases",
            "client_avg_sales": "recent_avg_sales",
            "client_avg_products": "recent_avg_products",
            "client_avg_branches": "recent_avg_branches",
            "client_avg_day_diff": "recent_avg_day_diff",
        },
        axis="columns",
    )
    return recent_client_df


def join_client_dfs(client_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    """Joins the recent and total client aggregated dataframes.

    Parameters:
    -----------
    client_df: pd.DataFrame
        Dataframe grouped by client.
    recent_df: pd.DataFrame
        Recent dataframe grouped by client.

    Returns:
    --------
    joined_df: pd.DataFrame
        Joined dataframe.
    """
    joined_df = client_df.join(recent_df)
    joined_df["recent_avg_purchases"] = np.where(
        joined_df["recent_avg_purchases"].isna(),
        joined_df["client_avg_purchases"],
        joined_df["recent_avg_purchases"],
    )
    joined_df["recent_avg_sales"] = np.where(
        joined_df["recent_avg_sales"].isna(),
        joined_df["client_avg_sales"],
        joined_df["recent_avg_sales"],
    )
    joined_df["recent_avg_products"] = np.where(
        joined_df["recent_avg_products"].isna(),
        joined_df["client_avg_products"],
        joined_df["recent_avg_products"],
    )
    joined_df["recent_avg_branches"] = np.where(
        joined_df["recent_avg_branches"].isna(),
        joined_df["client_avg_branches"],
        joined_df["recent_avg_branches"],
    )
    joined_df["recent_avg_day_diff"] = np.where(
        joined_df["recent_avg_day_diff"].isna(),
        joined_df["client_avg_day_diff"],
        joined_df["recent_avg_day_diff"],
    )
    return joined_df


def filtering_joined_df(
    joined_df: pd.DataFrame, day_diff_to: int, total_days_to: int
) -> pd.DataFrame:
    """Filters the joined dataframe according to how often and how many times a client has shopped.

    Parameters:
    -----------
    joined_df: pd.DataFrame
        Dataframe we will be filtering.
    day_diff_to: int
        Threshold we use to verify if user has churned or not.
    total_days_to: int
        Threshold we use to verify if user is regular client or not.

    Returns:
    --------
    joined_df:
        Filtered dataframe.
    """
    max_date = joined_df["client_last_buy"].max()
    joined_df = joined_df[
        (max_date - joined_df["client_last_buy"]).dt.days
        <= day_diff_to * joined_df["client_avg_day_diff"]
    ]
    joined_df = joined_df[joined_df["client_total_days"] >= total_days_to]
    joined_df["loyalty_time"] = (max_date - joined_df["client_first_buy"]).dt.days
    joined_df = joined_df.drop(["client_first_buy", "client_last_buy"], axis=1)
    return joined_df


def max_dif_df(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Computes a dataframe which gives for each client, the maximum amount of time he has gone wthout buying.

    Parameters:
    -----------
    daily_df: pd.DataFrame
        Dataframe aggregated by day and client.

    Returns:
    --------
    client_df: pd.DataFrame
        Client dataframe with inofrmation about maximum time without shopping.
    """
    min_date = daily_df["date_order"].min()
    max_date = daily_df["date_order"].max()
    client_df = daily_df.groupby("client_id").agg(
        client_max_day_diff=pd.NamedAgg(column="Day_Difference", aggfunc="max"),
        client_first_buy=pd.NamedAgg(column="date_order", aggfunc="min"),
        client_last_buy=pd.NamedAgg(column="date_order", aggfunc="max"),
    )
    client_df["distace_to_min_date"] = (
        client_df["client_first_buy"] - min_date
    ).dt.days
    client_df["distance_to_max_date"] = (
        max_date - client_df["client_last_buy"]
    ).dt.days
    client_df = client_df.drop(["client_first_buy", "client_last_buy"], axis=1)
    client_df["client_max_day_diff"] = client_df.max(axis=1)
    client_df = client_df[["client_max_day_diff"]]
    return client_df


def final_preprocessing(
    data_path: Path,
    drop_time: int,
    churn_time: int,
    recent_to: int,
    day_diff_to: int,
    total_days_to: int,
) -> pd.DataFrame:
    """Computes the entire preprocessing pipeline.

    Parameters:
    -----------
    path_ : Path
            Path of the data.
    drop_time : int
        Number of most recent days we will drop from the dataset.
    churn_time : int
        Number of days we will use to create the churn df(to calculate churn).
    recent_to: int
        Amount of days we want to use for our recency analysis.
    day_diff_to: int
        Threshold we use to verify if user has churned or not.
    total_days_to: int
        Threshold we use to verify if user is regular client or not.

    Returns:
    --------
    final_df: pd.DataFrame
        Final preprocessed dataframe.
    """
    data = dataloader(data_path)
    daily_agg, churn_daily_agg = daily_preprocessing(data, drop_time, churn_time)
    client_agg = client_preprocessing(daily_agg)
    recent_client_agg = recent_client_preprocessing(daily_agg, recent_to)
    joined_client_agg = join_client_dfs(client_agg, recent_client_agg)
    filtered_joined_df = filtering_joined_df(
        joined_client_agg, day_diff_to, total_days_to
    )
    client_max_dif_df = max_dif_df(churn_daily_agg)
    final_df = filtered_joined_df.join(client_max_dif_df)
    final_df["client_max_day_diff"] = np.where(
        final_df["client_max_day_diff"].isna(),
        churn_time,
        final_df["client_max_day_diff"],
    )
    final_df["churn"] = np.where(
        final_df["client_max_day_diff"]
        >= day_diff_to * final_df["client_avg_day_diff"],
        1,
        0,
    )
    final_df = final_df.drop(["client_max_day_diff"], axis=1)
    return final_df
