import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from scipy import stats

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = [12, 6]

def load_data():
    """Load the datasets"""
    print("Loading data...")
    transaction_data = pd.read_excel('QVI_transaction_data.xlsx')
    purchase_behavior = pd.read_csv('QVI_purchase_behaviour.csv')
    return transaction_data, purchase_behavior

def examine_data(transaction_data, purchase_behavior):
    """Examine the structure of the data"""
    print("\nTransaction Data Info:")
    print(transaction_data.info())
    print("\nSample of transaction data:")
    print(transaction_data.head())
    
    print("\nPurchase Behavior Info:")
    print(purchase_behavior.info())
    print("\nSample of purchase behavior data:")
    print(purchase_behavior.head())

def clean_transaction_data(df):
    """Clean and preprocess transaction data"""
    print("\nCleaning transaction data...")
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], origin='1899-12-30', unit='D')
    
    # Examine PROD_NAME
    print("\nExamining product names:")
    print(df['PROD_NAME'].value_counts().head())
    
    # Analyze words in product names
    words = pd.Series(' '.join(df['PROD_NAME']).split()).value_counts()
    words = words[~words.index.str.contains('\\d')]
    words = words[words.index.str.match('^[a-zA-Z]+$')]
    print("\nMost common words in product names:")
    print(words.head(10))
    
    # Remove salsa products
    df = df[~df['PROD_NAME'].str.contains('salsa', case=False)]
    
    # Check for outliers in PROD_QTY
    print("\nChecking for outliers in product quantity:")
    print(df['PROD_QTY'].describe())
    
    # Investigate transactions with quantity = 200
    high_qty = df[df['PROD_QTY'] == 200]
    print("\nTransactions with quantity = 200:")
    print(high_qty)
    
    # Remove outlier customer
    df = df[df['LYLTY_CARD_NBR'] != 226000]
    
    return df

def analyze_dates(df):
    """Analyze transaction dates"""
    print("\nAnalyzing transaction dates...")
    
    # Count transactions by date
    daily_transactions = df.groupby('DATE').size()
    
    # Plot transactions over time
    plt.figure()
    daily_transactions.plot()
    plt.title('Transactions over time')
    plt.xlabel('Date')
    plt.ylabel('Number of transactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Focus on December
    december_transactions = daily_transactions[daily_transactions.index.month == 12]
    plt.figure()
    december_transactions.plot()
    plt.title('December Transactions')
    plt.xlabel('Date')
    plt.ylabel('Number of transactions')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_pack_sizes(df):
    """Analyze pack sizes"""
    print("\nAnalyzing pack sizes...")
    
    # Extract pack size from product name
    df['PACK_SIZE'] = df['PROD_NAME'].apply(lambda x: int(re.search(r'(\d+)g', x).group(1)) if re.search(r'(\d+)g', x) else None)
    
    # Display pack size distribution
    pack_size_dist = df['PACK_SIZE'].value_counts().sort_index()
    print("\nPack size distribution:")
    print(pack_size_dist)
    
    # Plot pack size histogram
    plt.figure()
    plt.hist(df['PACK_SIZE'], bins=30)
    plt.title('Distribution of Pack Sizes')
    plt.xlabel('Pack Size (g)')
    plt.ylabel('Frequency')
    plt.show()
    
    return df

def analyze_brands(df):
    """Analyze brands"""
    print("\nAnalyzing brands...")
    
    # Extract brand names
    df['BRAND'] = df['PROD_NAME'].apply(lambda x: x.split()[0].upper())
    
    # Clean brand names
    brand_mapping = {
        'RRD': 'RED_ROCK_DELI',
        'SNBTS': 'SUNBITES',
        'INFZNS': 'INFUZIONS',
        'WW': 'WOOLWORTHS',
        'SMITH': 'SMITHS',
        'NCC': 'NATURAL',
        'DORITO': 'DORITOS',
        'GRAIN': 'GRAINWAVES'
    }
    df['BRAND'] = df['BRAND'].replace(brand_mapping)
    
    # Display brand distribution
    print("\nBrand distribution:")
    print(df['BRAND'].value_counts())
    
    return df

def analyze_customer_segments(data):
    """Analyze customer segments"""
    print("\nAnalyzing customer segments...")
    
    # Total sales by segment
    sales = data.pivot_table(
        values='TOT_SALES',
        index='PREMIUM_CUSTOMER',
        columns='LIFESTAGE',
        aggfunc='sum'
    )
    
    # Plot sales proportions
    sales_prop = sales.div(sales.sum().sum()) * 100
    plt.figure()
    sales_prop.plot(kind='bar', stacked=True)
    plt.title('Proportion of Sales by Customer Segment')
    plt.xlabel('Premium Customer Flag')
    plt.ylabel('Proportion of Sales (%)')
    plt.legend(title='Lifestage', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
    
    # Customer counts by segment
    customer_counts = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].nunique()
    print("\nCustomer counts by segment:")
    print(customer_counts)

def analyze_purchase_behavior(data):
    """Analyze purchase behavior"""
    print("\nAnalyzing purchase behavior...")
    
    # Average units per customer by segment
    avg_units = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({
        'PROD_QTY': lambda x: x.sum() / data['LYLTY_CARD_NBR'].nunique()
    }).round(2)
    print("\nAverage units per customer:")
    print(avg_units)
    
    # Average price per unit by segment
    data['PRICE_PER_UNIT'] = data['TOT_SALES'] / data['PROD_QTY']
    avg_price = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PRICE_PER_UNIT'].mean().round(2)
    print("\nAverage price per unit:")
    print(avg_price)
    
    # Plot average price per unit
    avg_price_pivot = avg_price.unstack()
    plt.figure()
    avg_price_pivot.plot(kind='bar')
    plt.title('Average Price per Unit by Segment')
    plt.xlabel('Premium Customer Flag')
    plt.ylabel('Average Price ($)')
    plt.legend(title='Lifestage', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def analyze_brand_affinity(data, target_segment):
    """Analyze brand affinity for target segment"""
    print(f"\nAnalyzing brand affinity for {target_segment}...")
    
    # Calculate brand preferences
    target = data[data['LIFESTAGE'] == target_segment]
    others = data[data['LIFESTAGE'] != target_segment]
    
    target_props = target['BRAND'].value_counts(normalize=True)
    other_props = others['BRAND'].value_counts(normalize=True)
    
    affinity = (target_props / other_props).round(2)
    affinity = affinity.sort_values(ascending=False)
    
    print(f"\nBrand affinity scores (>1 means {target_segment} more likely to purchase):")
    print(affinity)
    
    # Plot brand affinity
    plt.figure(figsize=(12, 6))
    affinity.head(10).plot(kind='bar')
    plt.title(f'Brand Affinity for {target_segment}')
    plt.xlabel('Brand')
    plt.ylabel('Affinity Score')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    transaction_data, purchase_behavior = load_data()
    
    # Examine data
    examine_data(transaction_data, purchase_behavior)
    
    # Clean and preprocess data
    transaction_data = clean_transaction_data(transaction_data)
    
    # Analyze dates
    analyze_dates(transaction_data)
    
    # Analyze pack sizes
    transaction_data = analyze_pack_sizes(transaction_data)
    
    # Analyze brands
    transaction_data = analyze_brands(transaction_data)
    
    # Merge with customer data
    data = pd.merge(transaction_data, purchase_behavior, on='LYLTY_CARD_NBR', how='left')
    
    # Analyze customer segments
    analyze_customer_segments(data)
    
    # Analyze purchase behavior
    analyze_purchase_behavior(data)
    
    # Analyze brand affinity for young singles/couples
    analyze_brand_affinity(data, 'YOUNG SINGLES/COUPLES')
    
    # Save processed data
    data.to_csv('processed_data.csv', index=False)
    print("\nAnalysis complete! Processed data saved to 'processed_data.csv'")

if __name__ == "__main__":
    main() 