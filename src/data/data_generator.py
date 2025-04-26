import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
NUM_ACCOUNTS = 500
NUM_TRANSACTIONS = 5000
FRAUD_RATIO = 0.02  # 2% fraud
COUNTRIES = ['US', 'UK', 'DE', 'IN', 'CN', 'BR', 'FR', 'AU', 'CA', 'ZA']
SUSPICIOUS_PAIRS = [('NG', 'US'), ('RU', 'CA'), ('CN', 'UK'), ('BR', 'AU')]

OUTPUT_DIR = "data/raw"

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def generate_accounts(num_accounts):
    return [f"A{str(i).zfill(5)}" for i in range(num_accounts)]

def generate_transactions(accounts, num_transactions, fraud_ratio):
    transactions = []
    start_time = datetime(2023, 1, 1)
    frequency_map = defaultdict(lambda: defaultdict(int))  # {account: {date: count}}

    for i in range(num_transactions):
        src = random.choice(accounts)
        dst = random.choice(accounts)
        while dst == src:
            dst = random.choice(accounts)

        # Timestamp and metadata
        timestamp = start_time + timedelta(minutes=random.randint(0, 525600))
        date_key = timestamp.date()
        amount = np.random.exponential(200)
        country_src = random.choice(COUNTRIES)
        country_dst = random.choice(COUNTRIES)
        is_fraud = 0

        if random.random() < fraud_ratio:
            is_fraud = 1
            amount *= random.uniform(5, 20)
            country_src, country_dst = random.choice(SUSPICIOUS_PAIRS)

        # Frequency tracking
        frequency_map[src][date_key] += 1

        transactions.append({
            "transaction_id": f"T{i:06d}",
            "source_account": src,
            "destination_account": dst,
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "source_country": country_src,
            "destination_country": country_dst,
            "is_fraud": is_fraud,
            "frequency": frequency_map[src][date_key]
        })

    return pd.DataFrame(transactions)

def generate_graph_data(df):
    # Edge list
    edge_list = df[["source_account", "destination_account", "timestamp", "amount", "is_fraud"]].copy()
    edge_list.columns = ["src", "dst", "time", "amount", "label"]

    # Node list
    accounts_info = {}
    for _, row in df.iterrows():
        if row.source_account not in accounts_info:
            accounts_info[row.source_account] = row.source_country
        if row.destination_account not in accounts_info:
            accounts_info[row.destination_account] = row.destination_country

    nodes = pd.DataFrame([
        {"account_id": acc, "country": country}
        for acc, country in accounts_info.items()
    ])

    return edge_list, nodes

if __name__ == "__main__":
    ensure_output_dir(OUTPUT_DIR)

    accounts = generate_accounts(NUM_ACCOUNTS)
    df = generate_transactions(accounts, NUM_TRANSACTIONS, FRAUD_RATIO)
    edge_list, node_list = generate_graph_data(df)

    # Save CSVs to data/raw/
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_fraud_dataset.csv"), index=False)
    edge_list.to_csv(os.path.join(OUTPUT_DIR, "graph_edges.csv"), index=False)
    node_list.to_csv(os.path.join(OUTPUT_DIR, "graph_nodes.csv"), index=False)

    
