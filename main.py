#Task 1
import pandas as pd
import numpy
transactions = pd.read_excel("QVI_transaction_data.xlsx")
customers = pd.read_csv("QVI_purchase_behaviour (1).csv")
print(transactions.head())
print(customers.head())
print(transactions.info())
print(customers.info())
print(transactions.columns)
print(customers.columns)
merged_data = pd.merge(transactions, customers, on="LYLTY_CARD_NBR")
print(merged_data.shape)
print(merged_data.info())
print(merged_data.isnull().sum())
print(merged_data.duplicated().sum())
print(merged_data.describe())
merged_data= merged_data.drop_duplicates()
print(merged_data["PROD_QTY"].describe())
print(merged_data["TOT_SALES"].describe())
print(merged_data.sort_values(by="PROD_QTY", ascending=False).head(10))
print(merged_data.sort_values(by="TOT_SALES", ascending=False).head())
print(
    merged_data[["PROD_QTY", "TOT_SALES"]]
    .sort_values(by="PROD_QTY", ascending=False)
    .head(10))
merged_data = merged_data[merged_data["PROD_QTY"] != 200]
print(merged_data["PROD_QTY"].max())
print(
    merged_data[["PROD_QTY", "TOT_SALES"]]
    .sort_values(by="TOT_SALES", ascending=False)
    .head(10))
print(merged_data["LIFESTAGE"].value_counts())
print(merged_data["PREMIUM_CUSTOMER"].value_counts())
print(merged_data.groupby("LIFESTAGE")["TOT_SALES"].sum().sort_values(ascending=False))
print(merged_data.groupby("LIFESTAGE")["TOT_SALES"].mean().sort_values(ascending=False))
print(merged_data.groupby("PREMIUM_CUSTOMER")["TOT_SALES"].sum().sort_values(ascending=False))
print(merged_data["PROD_NAME"].value_counts().head(10))
print(merged_data.groupby("PROD_NAME")["TOT_SALES"].sum().sort_values(ascending=False).head(10))
print(merged_data.groupby("LIFESTAGE")["TOT_SALES"].mean())

# Correct Brand Name extraction

def get_brand(product):

    product = product.lower().strip()

    if product.startswith(("natural chip", "ncc")):
        return "Natural Chip Co"

    elif product.startswith("ccs"):
        return "CCs"

    elif product.startswith(("smiths", "smith ")):
        return "Smiths"

    elif product.startswith("kettle"):
        return "Kettle"

    elif product.startswith("old el paso"):
        return "Old El Paso"

    elif product.startswith(("grain waves", "grnwves")):
        return "Grain Waves"

    elif product.startswith(("doritos", "dorito ")):
        return "Doritos"

    elif product.startswith("twisties"):
        return "Twisties"

    elif product.startswith(("ww ", "woolworths")):
        return "Woolworths"

    elif product.startswith("thins"):
        return "Thins"

    elif product.startswith("burger rings"):
        return "Burger Rings"

    elif product.startswith("cheezels"):
        return "Cheezels"

    elif product.startswith(("infuzions", "infzns")):
        return "Infuzions"

    elif product.startswith(("red rock deli", "rrd")):
        return "Red Rock Deli"

    elif product.startswith("pringles"):
        return "Pringles"

    elif product.startswith("cobs"):
        return "Cobs"

    elif product.startswith("french fries"):
        return "French Fries"

    elif product.startswith("tyrrells"):
        return "Tyrrells"

    elif product.startswith("tostitos"):
        return "Tostitos"

    elif product.startswith("cheetos"):
        return "Cheetos"

    elif product.startswith(("sunbites", "snbts")):
        return "Sunbites"

    else:
        return "Other"


merged_data["Brand Name"] = merged_data["PROD_NAME"].apply(get_brand)

merged_data["Pack Size"] = (
    merged_data["PROD_NAME"]
    .str.extract(r"(\d+)")
    .astype(int)
)

import matplotlib.pyplot as plt
sales_customer = merged_data.groupby("PREMIUM_CUSTOMER")["TOT_SALES"].sum()
sales_customer.plot(kind="bar")
plt.title("Total Sales by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

sales_life = (merged_data.groupby("LIFESTAGE")["TOT_SALES"].sum().sort_values(ascending=False))
sales_life.plot(kind="bar")
plt.title("Total Sales by Lifestage")
plt.xlabel("Lifestage")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

top_stores = (merged_data.groupby("STORE_NBR")["TOT_SALES"].sum().sort_values(ascending=False).head(10))
top_stores.plot(kind="bar")
plt.title("Top 10 Stores by Total Sales")
plt.xlabel("Store")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Task 2

merged_data['DATE'] =pd.to_datetime(merged_data["DATE"], origin='1899-12-30', unit='D')
merged_data['YEAR_MONTH'] = merged_data['DATE'].dt.to_period('M')
print((merged_data[['DATE','YEAR_MONTH']].head()))

monthly_matrics = (
                    merged_data
                    .groupby(["STORE_NBR", "YEAR_MONTH"])
                     .agg(
                           TOTAL_SALES=('TOT_SALES', "sum"),
                           TOTAL_CUSTOMERS=('LYLTY_CARD_NBR', "nunique"),
                           TOTAL_TRANSACTIONS=('DATE', "count")).reset_index())
monthly_matrics["AVG_TXN_PER_CUSTOMER"] = (
    monthly_matrics["TOTAL_TRANSACTIONS"] / monthly_matrics["TOTAL_CUSTOMERS"]
)
print(monthly_matrics.head())

pre_trial = monthly_matrics[
    monthly_matrics['YEAR_MONTH'] < '2019-02'
]
print(pre_trial['YEAR_MONTH'].unique())
import numpy as np

def find_control_store(trial_store, metric='TOTAL_SALES'):

    trial = (
        pre_trial[pre_trial['STORE_NBR'] == trial_store]
        .set_index('YEAR_MONTH')[metric]
    )

    scores = []

    for store in pre_trial['STORE_NBR'].unique():

        if store == trial_store:
            continue

        control = (
            pre_trial[pre_trial['STORE_NBR'] == store]
            .set_index('YEAR_MONTH')[metric]
        )

        common_months = trial.index.intersection(control.index)

        if len(common_months) > 5:

            trial_values = trial.loc[common_months]
            control_values = control.loc[common_months]

            corr = trial_values.corr(control_values)

            distance = abs(trial_values - control_values)

            if distance.max() == distance.min():
                magnitude = 1
            else:
                magnitude = (
                    1 - (
                        (distance - distance.min()) /
                        (distance.max() - distance.min())
                    )
                ).mean()

            final_score = (corr + magnitude) / 2

            scores.append(
                [store, corr, magnitude, final_score]
            )

    scores = pd.DataFrame(
        scores,
        columns=[
            "STORE_NBR",
            "Correlation",
            "Magnitude",
            "FinalScore"
        ]
    )

    scores = scores.sort_values(
        "FinalScore",
        ascending=False
    )

    best_store = scores.iloc[0]

    return best_store["STORE_NBR"], best_store["FinalScore"]
for trial in [77, 86, 88]:
        control, score = find_control_store(trial)
        print(f'Trial Store {trial} → Control Store {control} | Final Score = {score:.3f}')
for trial_store in [77, 86, 88]:

    control_store, _ = find_control_store(trial_store)

    trial_data = pre_trial[
        pre_trial['STORE_NBR'] == trial_store
    ]

    control_data = pre_trial[
        pre_trial['STORE_NBR'] == control_store
    ]

    plt.figure(figsize=(10,5))

    plt.plot(
        trial_data['YEAR_MONTH'].astype(str),
        trial_data['TOTAL_SALES'],
        marker='o',
        label=f'Trial {trial_store}'
    )

    plt.plot(
        control_data['YEAR_MONTH'].astype(str),
        control_data['TOTAL_SALES'],
        marker='o',
        label=f'Control {int(control_store)}'
    )

    plt.title(f'Pre-Trial Sales Comparison ({trial_store})')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

trial_period = monthly_matrics[
    monthly_matrics['YEAR_MONTH'] >= '2019-02'
]

from scipy.stats import ttest_ind

for trial_store in [77, 86, 88]:

    control_store, _ = find_control_store(trial_store)

    trial_sales = trial_period[
        trial_period['STORE_NBR'] == trial_store
    ][['YEAR_MONTH', 'TOTAL_SALES']]

    control_sales = trial_period[
        trial_period['STORE_NBR'] == control_store
    ][['YEAR_MONTH', 'TOTAL_SALES']]

    comparison = trial_sales.merge(
        control_sales,
        on='YEAR_MONTH',
        suffixes=('_TRIAL', '_CONTROL')
    )

    comparison['DIFFERENCE'] = (
        comparison['TOTAL_SALES_TRIAL']
        - comparison['TOTAL_SALES_CONTROL']
    )

    print("\n====================================")
    print(f"Trial Store {trial_store} vs Control Store {int(control_store)}")
    print("====================================")
    print(comparison)

    t_stat, p_value = ttest_ind(
        comparison['TOTAL_SALES_TRIAL'],
        comparison['TOTAL_SALES_CONTROL'],
        equal_var=False
    )

    print(f"T-statistic = {t_stat:.3f}")
    print(f"P-value = {p_value:.5f}")

    for trial_store in [77, 86, 88]:
        control_store, _ = find_control_store(trial_store)

        trial_customers = trial_period[
            trial_period["STORE_NBR"] == trial_store
            ][["YEAR_MONTH", "TOTAL_CUSTOMERS"]]

        control_customers = trial_period[
            trial_period["STORE_NBR"] == control_store
            ][["YEAR_MONTH", "TOTAL_CUSTOMERS"]]

        customer_compare = trial_customers.merge(
            control_customers,
            on="YEAR_MONTH",
            suffixes=("_TRIAL", "_CONTROL")
        )

        print(f"\nCustomer Comparison - Store {trial_store}")
        print(customer_compare)
plt.figure(figsize=(10,5))

plt.plot(
    customer_compare["YEAR_MONTH"].astype(str),
    customer_compare["TOTAL_CUSTOMERS_TRIAL"],
    marker="o",
    label="Trial"
)

plt.plot(
    customer_compare["YEAR_MONTH"].astype(str),
    customer_compare["TOTAL_CUSTOMERS_CONTROL"],
    marker="o",
    label="Control"
)

plt.title(f"Customers Comparison - Store {trial_store}")
plt.xlabel("Month")
plt.ylabel("Customers")
plt.legend()

plt.tight_layout()
plt.show()
for trial_store in [77,86,88]:

    control_store, _ = find_control_store(trial_store)

    trial_txn = trial_period[
        trial_period["STORE_NBR"] == trial_store
    ][["YEAR_MONTH","AVG_TXN_PER_CUSTOMER"]]

    control_txn = trial_period[
        trial_period["STORE_NBR"] == control_store
    ][["YEAR_MONTH","AVG_TXN_PER_CUSTOMER"]]

    txn_compare = trial_txn.merge(
        control_txn,
        on="YEAR_MONTH",
        suffixes=("_TRIAL","_CONTROL")
    )

    print(txn_compare)
    t_stat, p_value = ttest_ind(
        txn_compare["AVG_TXN_PER_CUSTOMER_TRIAL"],
        txn_compare["AVG_TXN_PER_CUSTOMER_CONTROL"],
        equal_var=False
    )

    print("Transaction T =", t_stat)
    print("Transaction P =", p_value)
    plt.figure(figsize=(10, 5))

    plt.plot(
        txn_compare["YEAR_MONTH"].astype(str),
        txn_compare["AVG_TXN_PER_CUSTOMER_TRIAL"],
        marker="o",
        label="Trial"
    )

    plt.plot(
        txn_compare["YEAR_MONTH"].astype(str),
        txn_compare["AVG_TXN_PER_CUSTOMER_CONTROL"],
        marker="o",
        label="Control"
    )

    plt.title(f"Average Transactions per Customer ({trial_store})")
    plt.xlabel("Month")
    plt.ylabel("Average Transactions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Final campaign impact summary

    campaign_results = []

    for trial_store in [77, 86, 88]:
            control_store, _ = find_control_store(trial_store)

            trial_sales = trial_period[
                trial_period["STORE_NBR"] == trial_store
                ]["TOTAL_SALES"].sum()

            control_sales = trial_period[
                trial_period["STORE_NBR"] == control_store
                ]["TOTAL_SALES"].sum()

            sales_difference = trial_sales - control_sales

            sales_change_pct = (
                                       sales_difference / control_sales
                               ) * 100

            campaign_results.append({
                "TRIAL_STORE": trial_store,
                "CONTROL_STORE": control_store,
                "TRIAL_SALES": trial_sales,
                "CONTROL_SALES": control_sales,
                "SALES_DIFFERENCE": sales_difference,
                "SALES_CHANGE_%": sales_change_pct
            })
    campaign_results = pd.DataFrame(campaign_results)

    print("\nFinal Campaign Impact:")
    print(campaign_results)
    # ==========================================
    # Trial vs Control - Dashboard Results
    # ==========================================

    dashboard_results = []

    for trial_store in [77, 86, 88]:
        control_store, _ = find_control_store(trial_store)

        trial_data = trial_period[
            trial_period["STORE_NBR"] == trial_store
            ][[
            "YEAR_MONTH",
            "TOTAL_SALES",
            "TOTAL_CUSTOMERS",
            "AVG_TXN_PER_CUSTOMER"
        ]]

        control_data = trial_period[
            trial_period["STORE_NBR"] == control_store
            ][[
            "YEAR_MONTH",
            "TOTAL_SALES",
            "TOTAL_CUSTOMERS",
            "AVG_TXN_PER_CUSTOMER"
        ]]

        comparison = trial_data.merge(
            control_data,
            on="YEAR_MONTH",
            suffixes=("_TRIAL", "_CONTROL")
        )

        comparison["TRIAL_STORE"] = trial_store
        comparison["CONTROL_STORE"] = control_store

        dashboard_results.append(comparison)

    dashboard_results = pd.concat(
        dashboard_results,
        ignore_index=True
    )

    dashboard_results = dashboard_results[
        [
            "YEAR_MONTH",
            "TRIAL_STORE",
            "CONTROL_STORE",
            "TOTAL_SALES_TRIAL",
            "TOTAL_SALES_CONTROL",
            "TOTAL_CUSTOMERS_TRIAL",
            "TOTAL_CUSTOMERS_CONTROL",
            "AVG_TXN_PER_CUSTOMER_TRIAL",
            "AVG_TXN_PER_CUSTOMER_CONTROL"
        ]
    ]

    print("\nDashboard Results:")
    print(dashboard_results)
import os
print(os.getcwd())

# Customer Impact Summary

customer_results = []

for trial_store in [77, 86, 88]:

    control_store, _ = find_control_store(trial_store)

    trial_customers = merged_data[
        (merged_data["STORE_NBR"] == trial_store) &
        (merged_data["DATE"] >= "2019-02-01")
    ]["LYLTY_CARD_NBR"].nunique()

    control_customers = merged_data[
        (merged_data["STORE_NBR"] == control_store) &
        (merged_data["DATE"] >= "2019-02-01")
    ]["LYLTY_CARD_NBR"].nunique()

    customer_difference = trial_customers - control_customers

    customer_change_pct = (
        customer_difference / control_customers
    ) * 100

    customer_results.append({
        "TRIAL_STORE": trial_store,
        "CONTROL_STORE": control_store,
        "TRIAL_CUSTOMERS": trial_customers,
        "CONTROL_CUSTOMERS": control_customers,
        "CUSTOMER_DIFFERENCE": customer_difference,
        "CUSTOMER_CHANGE_%": customer_change_pct
    })

customer_results = pd.DataFrame(customer_results)


transaction_results = []

for trial_store in [77, 86, 88]:

    control_store, _ = find_control_store(trial_store)

    trial_avg_txn = trial_period[
        trial_period["STORE_NBR"] == trial_store
    ]["AVG_TXN_PER_CUSTOMER"].mean()

    control_avg_txn = trial_period[
        trial_period["STORE_NBR"] == control_store
    ]["AVG_TXN_PER_CUSTOMER"].mean()

    txn_difference = trial_avg_txn - control_avg_txn

    txn_change_pct = (
        txn_difference / control_avg_txn
    ) * 100

    transaction_results.append({
        "TRIAL_STORE": trial_store,
        "CONTROL_STORE": control_store,
        "TRIAL_AVG_TXN": trial_avg_txn,
        "CONTROL_AVG_TXN": control_avg_txn,
        "TXN_DIFFERENCE": txn_difference,
        "TXN_CHANGE_%": txn_change_pct
    })

transaction_results = pd.DataFrame(transaction_results)
with pd.ExcelWriter("FianlAnalysis.xlsx") as writer:

    merged_data.to_excel(
        writer,
        sheet_name="Clean Data",
        index=False
    )

    campaign_results.to_excel(
        writer,
        sheet_name="Campaign Results",
        index=False,
        startrow=0
    )

    customer_results.to_excel(
        writer,
        sheet_name="Campaign Results",
        index=False,
        startrow=6
    )

    transaction_results.to_excel(
        writer,
        sheet_name="Campaign Results",
        index=False,
        startrow=12
    )

    dashboard_results.to_excel(
        writer,
        sheet_name="Dashboard Results",
        index=False
    )

