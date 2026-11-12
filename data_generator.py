"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Data Generation Engine
Generates realistic synthetic banking transactions and customer profiles with controlled anomaly injections.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

SEED = 42

def set_seed(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)

def generate_customers(num_customers=350, seed=SEED):
    """
    Generate realistic wealth management and banking customer records.
    """
    set_seed(seed)
    
    first_names = [
        "Alexander", "Eleanor", "Marcus", "Victoria", "Julian", "Sophia", "Charles", "Audrey",
        "Sebastian", "Charlotte", "Oliver", "Amelia", "Theodore", "Isabella", "Harrison", "Genevieve",
        "Arthur", "Clara", "Maximilian", "Beatrice", "Montgomery", "Evelyn", "Vincent", "Penelope",
        "Benedict", "Camilla", "Lawrence", "Adeline", "Dominic", "Rosalind", "Sterling", "Vivian",
        "Archibald", "Florence", "Nathaniel", "Cordelia", "Jonathan", "Margaret", "Frederick", "Eleanor"
    ]
    
    last_names = [
        "Vanderbilt", "Rothschild", "Sterling", "Kensington", "Blackwood", "Sinclair", "Hastings",
        "Wellington", "Ashford", "Montgomery", "Chambers", "Winslow", "Prescott", "Pembroke",
        "Fairchild", "Ellington", "Covington", "Huntington", "Thorne", "Lancaster", "Davenport",
        "Barrington", "Mercer", "Harrington", "Whitmore", "Radcliffe", "Beaumont", "Kingsley"
    ]
    
    segments = [
        ("Ultra High Net Worth", 0.10, (500000, 5000000)),
        ("High Net Worth", 0.25, (150000, 1000000)),
        ("Affluent Wealth", 0.35, (50000, 300000)),
        ("Corporate Institutional", 0.15, (250000, 2500000)),
        ("Emerging Affluent", 0.15, (15000, 80000))
    ]
    
    account_types = {
        "Ultra High Net Worth": ["Private Banking Portfolio", "Trust & Estate Asset", "Multi-Asset Wealth"],
        "High Net Worth": ["Wealth Management Account", "Discretionary Investment Portfolio", "Private Banking Portfolio"],
        "Affluent Wealth": ["Premium Investment Checking", "Wealth Management Account", "High Yield Managed Fund"],
        "Corporate Institutional": ["Corporate Treasury", "Institutional Liquidity Fund", "Commercial Escrow"],
        "Emerging Affluent": ["Premium Investment Checking", "Personal Wealth Builder", "High Yield Managed Fund"]
    }
    
    customers = []
    
    for i in range(1, num_customers + 1):
        cust_id = f"CUST-{1000 + i}"
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        name = f"{fn} {ln}"
        age = int(np.random.normal(52, 14))
        age = max(24, min(85, age))
        
        # Select segment
        seg_choice = random.choices(
            [s[0] for s in segments],
            weights=[s[1] for s in segments]
        )[0]
        
        # Get balance bounds
        for s in segments:
            if s[0] == seg_choice:
                low_b, high_b = s[2]
                break
        
        balance = round(float(np.random.uniform(low_b, high_b)), 2)
        acc_type = random.choice(account_types[seg_choice])
        
        customers.append({
            "customer_id": cust_id,
            "customer_name": name,
            "age": age,
            "customer_segment": seg_choice,
            "account_type": acc_type,
            "account_balance": balance
        })
        
    df_customers = pd.DataFrame(customers)
    return df_customers

def generate_transactions(df_customers, num_transactions=6000, anomaly_rate=0.05, seed=SEED):
    """
    Generate normal and injected anomalous banking transactions.
    """
    set_seed(seed)
    
    merchant_categories = [
        "Securities & Investments", "Luxury Goods", "Travel & Hospitality",
        "Fine Dining", "Corporate Services", "Wire Transfers",
        "Electronics & Tech", "Real Estate & Escrow", "Consulting & Legal",
        "Groceries & Household", "Automotive & Marine"
    ]
    
    payment_methods = [
        "Wire Transfer", "ACH Direct Debit", "Corporate Credit Card",
        "Executive Debit Card", "SWIFT Cross-Border", "Online Portal"
    ]
    
    locations = [
        "New York, US", "London, UK", "Zurich, CH", "Singapore, SG",
        "Geneva, CH", "Tokyo, JP", "Hong Kong, HK", "Frankfurt, DE",
        "Dubai, AE", "San Francisco, US", "Chicago, US", "Paris, FR"
    ]
    
    # Establish base date range: 180 days ending today
    end_date = datetime(2026, 11, 12, 17, 0, 0)
    start_date = end_date - timedelta(days=180)
    
    # Determine anomaly count
    num_anomalies = int(num_transactions * anomaly_rate)
    num_normal = num_transactions - num_anomalies
    
    transactions = []
    
    # Pre-map customer metadata for fast lookups
    cust_map = df_customers.set_index("customer_id").to_dict("index")
    cust_ids = list(cust_map.keys())
    
    # Customer baseline spend profiles
    customer_spend_profiles = {}
    for cid, cdata in cust_map.items():
        seg = cdata["customer_segment"]
        if seg == "Ultra High Net Worth":
            base_mean = 8500
            base_std = 4200
        elif seg == "High Net Worth":
            base_mean = 3200
            base_std = 1800
        elif seg == "Corporate Institutional":
            base_mean = 14500
            base_std = 7000
        elif seg == "Affluent Wealth":
            base_mean = 950
            base_std = 450
        else: # Emerging Affluent
            base_mean = 420
            base_std = 220
        customer_spend_profiles[cid] = (base_mean, base_std)
        
    print(f"Generating {num_normal} normal transactions...")
    for i in range(1, num_normal + 1):
        txn_id = f"TXN-{100000 + i}"
        cid = random.choice(cust_ids)
        cinfo = cust_map[cid]
        base_mean, base_std = customer_spend_profiles[cid]
        
        # Log-normal distribution for positive realistic transaction amounts
        raw_amt = np.random.normal(base_mean, base_std)
        amount = round(max(15.0, raw_amt), 2)
        
        # Transaction type: 78% Debit, 22% Credit (inbound deposit / dividend)
        is_credit = np.random.rand() < 0.22
        txn_type = "Credit" if is_credit else "Debit"
        
        # Weighted category based on segment
        cat = random.choice(merchant_categories)
        pm = random.choice(payment_methods)
        loc = random.choice(locations)
        
        # Timestamp with business-hour bias (9am - 6pm)
        rand_days = np.random.uniform(0, 180)
        txn_datetime = start_date + timedelta(days=rand_days)
        # Business hour bias
        if np.random.rand() < 0.82:
            hour = int(np.random.normal(14, 3))
            hour = max(8, min(20, hour))
        else:
            hour = random.choice(list(range(0, 8)) + list(range(21, 24)))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        txn_datetime = txn_datetime.replace(hour=hour, minute=minute, second=second)
        
        # Current simulated account balance
        cur_balance = round(max(500.0, cinfo["account_balance"] + np.random.normal(0, base_mean * 0.5)), 2)
        
        transactions.append({
            "transaction_id": txn_id,
            "customer_id": cid,
            "transaction_date": txn_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_amount": amount,
            "transaction_type": txn_type,
            "merchant_category": cat,
            "payment_method": pm,
            "account_balance": cur_balance,
            "location": loc,
            "is_injected_anomaly": 0
        })
        
    print(f"Injecting {num_anomalies} synthetic anomalies across 5 structural risk typologies...")
    
    # 5 anomaly typologies:
    # 1. Extremely high transaction amounts
    # 2. Velocity / Burst transactions in short time window
    # 3. Unusually large withdrawal relative to customer liquid balance
    # 4. Inconsistent with customer historical behavior
    # 5. Overdraft / amount far exceeding account balance
    
    txn_counter = num_normal + 1
    
    anomaly_types_count = {
        "Extreme High Amount": 0,
        "Rapid Velocity Burst": 0,
        "Massive Balance Depletion": 0,
        "Atypical Behavior Inconsistency": 0,
        "Balance Overdraft/Excess": 0
    }
    
    while len(transactions) < num_transactions:
        typology = random.choice([1, 2, 3, 4, 5])
        cid = random.choice(cust_ids)
        cinfo = cust_map[cid]
        base_mean, base_std = customer_spend_profiles[cid]
        
        rand_days = np.random.uniform(5, 175)
        base_time = start_date + timedelta(days=rand_days)
        
        if typology == 1:
            # 1. Extremely high transaction amount (15x to 50x mean)
            amount = round(float(base_mean * np.random.uniform(15, 45) + np.random.uniform(50000, 300000)), 2)
            hour = random.choice([1, 2, 3, 4, 23]) # odd off-hours
            txn_dt = base_time.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
            
            transactions.append({
                "transaction_id": f"TXN-{100000 + txn_counter}",
                "customer_id": cid,
                "transaction_date": txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": amount,
                "transaction_type": "Debit",
                "merchant_category": "Wire Transfers",
                "payment_method": "SWIFT Cross-Border",
                "account_balance": cinfo["account_balance"],
                "location": random.choice(["Dubai, AE", "Hong Kong, HK", "Zurich, CH"]),
                "is_injected_anomaly": 1
            })
            txn_counter += 1
            anomaly_types_count["Extreme High Amount"] += 1
            
        elif typology == 2:
            # 2. Rapid Velocity Burst: 3-5 rapid successive transactions within 2-10 minutes
            burst_len = min(random.randint(3, 5), num_transactions - len(transactions))
            burst_start = base_time.replace(hour=random.randint(0, 23), minute=random.randint(0, 45))
            
            for b in range(burst_len):
                b_time = burst_start + timedelta(seconds=random.randint(20, 180) * (b + 1))
                b_amount = round(float(np.random.uniform(3500, 18000)), 2)
                transactions.append({
                    "transaction_id": f"TXN-{100000 + txn_counter}",
                    "customer_id": cid,
                    "transaction_date": b_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_amount": b_amount,
                    "transaction_type": "Debit",
                    "merchant_category": "Electronics & Tech",
                    "payment_method": "Online Portal",
                    "account_balance": cinfo["account_balance"],
                    "location": "New York, US",
                    "is_injected_anomaly": 1
                })
                txn_counter += 1
                anomaly_types_count["Rapid Velocity Burst"] += 1
                
        elif typology == 3:
            # 3. Unusually large withdrawal depleting 90-99% of customer account balance
            depletion_ratio = np.random.uniform(0.90, 0.99)
            amount = round(float(cinfo["account_balance"] * depletion_ratio), 2)
            txn_dt = base_time.replace(hour=random.randint(9, 17), minute=random.randint(0, 59))
            transactions.append({
                "transaction_id": f"TXN-{100000 + txn_counter}",
                "customer_id": cid,
                "transaction_date": txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": amount,
                "transaction_type": "Debit",
                "merchant_category": "Real Estate & Escrow",
                "payment_method": "Wire Transfer",
                "account_balance": cinfo["account_balance"],
                "location": random.choice(locations),
                "is_injected_anomaly": 1
            })
            txn_counter += 1
            anomaly_types_count["Massive Balance Depletion"] += 1
            
        elif typology == 4:
            # 4. Inconsistent with customer history (Emerging affluent suddenly doing $75k luxury transaction)
            cid_emerging = [c for c in cust_ids if cust_map[c]["customer_segment"] == "Emerging Affluent"]
            target_cid = random.choice(cid_emerging) if cid_emerging else cid
            amount = round(float(np.random.uniform(45000, 95000)), 2)
            txn_dt = base_time.replace(hour=random.choice([0, 1, 2, 23]), minute=random.randint(0, 59))
            transactions.append({
                "transaction_id": f"TXN-{100000 + txn_counter}",
                "customer_id": target_cid,
                "transaction_date": txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": amount,
                "transaction_type": "Debit",
                "merchant_category": "Luxury Goods",
                "payment_method": "Corporate Credit Card",
                "account_balance": cust_map[target_cid]["account_balance"],
                "location": "Geneva, CH",
                "is_injected_anomaly": 1
            })
            txn_counter += 1
            anomaly_types_count["Atypical Behavior Inconsistency"] += 1
            
        else:
            # 5. Overdraft/Excess: Transaction amount exceeding balance by 2x to 4x
            cur_bal = max(1000.0, float(cinfo["account_balance"] * 0.15))
            amount = round(float(cur_bal * np.random.uniform(2.5, 4.5)), 2)
            txn_dt = base_time.replace(hour=random.randint(8, 20), minute=random.randint(0, 59))
            transactions.append({
                "transaction_id": f"TXN-{100000 + txn_counter}",
                "customer_id": cid,
                "transaction_date": txn_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_amount": amount,
                "transaction_type": "Debit",
                "merchant_category": "Securities & Investments",
                "payment_method": "Wire Transfer",
                "account_balance": cur_bal,
                "location": random.choice(locations),
                "is_injected_anomaly": 1
            })
            txn_counter += 1
            anomaly_types_count["Balance Overdraft/Excess"] += 1

    df_transactions = pd.DataFrame(transactions)
    # Sort chronologically
    df_transactions["sort_dt"] = pd.to_datetime(df_transactions["transaction_date"])
    df_transactions = df_transactions.sort_values("sort_dt").reset_index(drop=True)
    df_transactions = df_transactions.drop(columns=["sort_dt"])
    
    print("\nAnomaly Injection Summary:")
    for k, v in anomaly_types_count.items():
        print(f" - {k}: {v}")
    print(f"Total Transactions: {len(df_transactions)} (Anomalies: {df_transactions['is_injected_anomaly'].sum()})")
    
    return df_transactions

def generate_and_save_dataset(data_dir="data", num_customers=350, num_transactions=6000, seed=SEED):
    """
    Main orchestration function to generate and persist synthetic data.
    """
    os.makedirs(data_dir, exist_ok=True)
    customers_file = os.path.join(data_dir, "synthetic_customers.csv")
    transactions_file = os.path.join(data_dir, "synthetic_transactions.csv")
    
    print(f"--- [FinGuard Data Generator] Initializing with SEED={seed} ---")
    df_customers = generate_customers(num_customers=num_customers, seed=seed)
    df_customers.to_csv(customers_file, index=False)
    print(f"Successfully generated and saved customers: {customers_file} ({len(df_customers)} records)")
    
    df_transactions = generate_transactions(df_customers, num_transactions=num_transactions, seed=seed)
    df_transactions.to_csv(transactions_file, index=False)
    print(f"Successfully generated and saved transactions: {transactions_file} ({len(df_transactions)} records)")
    
    return df_customers, df_transactions

if __name__ == "__main__":
    generate_and_save_dataset()

