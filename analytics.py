"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: SQL Analytics Engine & Institutional Reporting
Contains standardized SQL analytical queries and reporting procedures.
"""

import pandas as pd
from database import query_dataframe, DB_PATH

QUERIES = {
    "1. Executive KPI Summary": """
        SELECT 
            COUNT(t.transaction_id) AS total_transactions,
            ROUND(SUM(t.transaction_amount), 2) AS total_transaction_value,
            ROUND(AVG(t.transaction_amount), 2) AS avg_transaction_value,
            COUNT(DISTINCT t.customer_id) AS total_active_customers,
            COALESCE(SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END), 0) AS flagged_anomalies,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS anomaly_rate_pct,
            ROUND(SUM(CASE WHEN ar.anomaly_flag = 1 THEN t.transaction_amount ELSE 0 END), 2) AS flagged_anomaly_volume
        FROM transactions t
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id;
    """,

    "2. Top 10 Customers by Transaction Volume": """
        SELECT 
            c.customer_id,
            c.customer_name,
            c.customer_segment,
            c.account_type,
            COUNT(t.transaction_id) AS transaction_count,
            ROUND(SUM(t.transaction_amount), 2) AS total_amount,
            ROUND(AVG(t.transaction_amount), 2) AS avg_amount,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS anomaly_count
        FROM customers c
        JOIN transactions t ON c.customer_id = t.customer_id
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY c.customer_id, c.customer_name, c.customer_segment, c.account_type
        ORDER BY transaction_count DESC
        LIMIT 10;
    """,

    "3. Top 10 Customers by Total Transacted Value": """
        SELECT 
            c.customer_id,
            c.customer_name,
            c.customer_segment,
            c.account_balance,
            COUNT(t.transaction_id) AS transaction_count,
            ROUND(SUM(t.transaction_amount), 2) AS total_transacted_value,
            ROUND(MAX(t.transaction_amount), 2) AS max_single_transaction,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS anomaly_count
        FROM customers c
        JOIN transactions t ON c.customer_id = t.customer_id
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY c.customer_id, c.customer_name, c.customer_segment, c.account_balance
        ORDER BY total_transacted_value DESC
        LIMIT 10;
    """,

    "4. Merchant Category Breakdown & Risk Distribution": """
        SELECT 
            t.merchant_category,
            COUNT(t.transaction_id) AS transaction_count,
            ROUND(SUM(t.transaction_amount), 2) AS total_spend,
            ROUND(AVG(t.transaction_amount), 2) AS avg_ticket_size,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS anomaly_count,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS category_anomaly_rate_pct
        FROM transactions t
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY t.merchant_category
        ORDER BY total_spend DESC;
    """,

    "5. Daily Transaction Volume and Value Trends": """
        SELECT 
            SUBSTR(t.transaction_date, 1, 10) AS transaction_day,
            COUNT(t.transaction_id) AS daily_transactions,
            ROUND(SUM(t.transaction_amount), 2) AS daily_value,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS daily_anomalies
        FROM transactions t
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY SUBSTR(t.transaction_date, 1, 10)
        ORDER BY transaction_day ASC;
    """,

    "6. High-Value Transactions Audit (>= $50,000)": """
        SELECT 
            t.transaction_id,
            t.transaction_date,
            c.customer_id,
            c.customer_name,
            c.customer_segment,
            t.transaction_amount,
            t.transaction_type,
            t.merchant_category,
            t.payment_method,
            COALESCE(ar.anomaly_flag, 0) AS anomaly_flag,
            COALESCE(ar.risk_category, 'Low Risk') AS risk_category,
            COALESCE(ar.anomaly_score, 0.0) AS anomaly_score
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        WHERE t.transaction_amount >= 50000.0
        ORDER BY t.transaction_amount DESC;
    """,

    "7. Anomaly Distribution by Payment Method": """
        SELECT 
            t.payment_method,
            COUNT(t.transaction_id) AS total_transactions,
            ROUND(SUM(t.transaction_amount), 2) AS total_volume,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS flagged_count,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS anomaly_ratio_pct,
            ROUND(AVG(ar.anomaly_score), 2) AS avg_risk_score
        FROM transactions t
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY t.payment_method
        ORDER BY flagged_count DESC;
    """,

    "8. Customers with Highest Unusual Activity Concentration": """
        SELECT 
            c.customer_id,
            c.customer_name,
            c.customer_segment,
            c.account_type,
            COUNT(t.transaction_id) AS total_customer_txns,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS flagged_anomaly_count,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS customer_anomaly_rate_pct,
            ROUND(MAX(ar.anomaly_score), 2) AS peak_anomaly_score,
            ROUND(SUM(CASE WHEN ar.anomaly_flag = 1 THEN t.transaction_amount ELSE 0 END), 2) AS total_at_risk_capital
        FROM customers c
        JOIN transactions t ON c.customer_id = t.customer_id
        JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY c.customer_id, c.customer_name, c.customer_segment, c.account_type
        HAVING flagged_anomaly_count > 0
        ORDER BY flagged_anomaly_count DESC, peak_anomaly_score DESC
        LIMIT 15;
    """,

    "9. Customer Segment Risk & Portfolio Exposure Analysis": """
        SELECT 
            c.customer_segment,
            COUNT(DISTINCT c.customer_id) AS customer_count,
            ROUND(SUM(DISTINCT c.account_balance), 2) AS aggregate_balance,
            COUNT(t.transaction_id) AS transaction_count,
            ROUND(SUM(t.transaction_amount), 2) AS total_flow,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS total_anomalies,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS segment_anomaly_rate_pct,
            ROUND(SUM(CASE WHEN ar.risk_category = 'High Risk' THEN 1 ELSE 0 END), 2) AS high_risk_incidents
        FROM customers c
        LEFT JOIN transactions t ON c.customer_id = t.customer_id
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY c.customer_segment
        ORDER BY aggregate_balance DESC;
    """,

    "10. Off-Hours and Weekend High-Risk Surveillance": """
        SELECT 
            CASE 
                WHEN CAST(SUBSTR(t.transaction_date, 12, 2) AS INTEGER) < 6 
                  OR CAST(SUBSTR(t.transaction_date, 12, 2) AS INTEGER) >= 22 THEN 'Late Night / Early Morning'
                ELSE 'Regular Business Hours'
            END AS operational_timeframe,
            COUNT(t.transaction_id) AS txn_count,
            ROUND(SUM(t.transaction_amount), 2) AS volume,
            SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) AS flagged_count,
            ROUND(100.0 * SUM(CASE WHEN ar.anomaly_flag = 1 THEN 1 ELSE 0 END) / COUNT(t.transaction_id), 2) AS anomaly_rate_pct
        FROM transactions t
        LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
        GROUP BY operational_timeframe;
    """
}

def execute_analytics_query(query_name, db_path=DB_PATH):
    """
    Executes a named standard query and returns the DataFrame and raw SQL.
    """
    if query_name not in QUERIES:
        raise ValueError(f"Query '{query_name}' not defined.")
    sql = QUERIES[query_name]
    df = query_dataframe(sql, db_path=db_path)
    return df, sql

def execute_custom_sql(sql_query, db_path=DB_PATH):
    """
    Executes an arbitrary read-only SQL query supplied by the user.
    """
    # Safeguard against disruptive write statements in analytics view
    forbidden = ["DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT"]
    clean_sql = sql_query.strip().upper()
    for verb in forbidden:
        if clean_sql.startswith(verb):
            raise PermissionError(f"Action '{verb}' is restricted. Only read-only analytical statements (SELECT / WITH) are allowed.")
    
    return query_dataframe(sql_query, db_path=db_path)

