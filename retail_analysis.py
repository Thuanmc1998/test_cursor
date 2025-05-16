import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from scipy import stats

# Set style for plots
plt.style.use('seaborn')
sns.set_palette('Set2')

def load_and_examine_data():
    """Load and examine the datasets"""
    print("Loading data...")
    transaction_data = pd.read_excel('QVI_transaction_data.xlsx')
    purchase_behavior = pd.read_csv('QVI_purchase_behaviour.csv')
    
    print("\nTransaction Data Info:")
    print(transaction_data.info())
    print("\nPurchase Behavior Info:")
    print(purchase_behavior.info())
    
    return transaction_data, purchase_behavior

def clean_transaction_data(df):
    """Clean and preprocess transaction data"""
    print("\nCleaning transaction data...")
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'], origin='1899-12-30', unit='D')
    
    # Remove salsa products
    df = df[~df['PROD_NAME'].str.contains('salsa', case=False)]
    
    # Remove outlier customer (buying 200 units)
    df = df[df['LYLTY_CARD_NBR'] != 226000]
    
    # Extract pack size
    df['PACK_SIZE'] = df['PROD_NAME'].apply(lambda x: int(re.search(r'(\d+)g', x).group(1)) if re.search(r'(\d+)g', x) else None)
    
    # Extract and clean brand names
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
    
    return df

def analyze_product_words(df):
    """Analyze words in product names"""
    print("\nAnalyzing product words...")
    
    # Split product names into words and count frequency
    words = pd.Series(' '.join(df['PROD_NAME']).split()).value_counts()
    
    # Remove digits and special characters
    words = words[~words.index.str.contains('\\d')]
    words = words[words.index.str.match('^[a-zA-Z]+$')]
    
    print("\nMost common words in product names:")
    print(words.head(10))
    
    return words

def analyze_customer_segments(data):
    """Analyze customer segments"""
    print("\nAnalyzing customer segments...")
    
    # Calculate total sales by segment
    sales_by_segment = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({
        'TOT_SALES': ['sum', 'count', 'mean'],
        'LYLTY_CARD_NBR': 'nunique'
    }).round(2)
    
    print("\nSales by customer segment:")
    print(sales_by_segment)
    
    return sales_by_segment

def plot_segment_proportions(data, value_col, title):
    """Create visualization for segment proportions"""
    pivot_table = pd.pivot_table(
        data,
        values=value_col,
        index='PREMIUM_CUSTOMER',
        columns='LIFESTAGE',
        aggfunc='sum'
    )
    
    # Calculate proportions
    proportions = pivot_table.div(pivot_table.sum().sum()) * 100
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = proportions.plot(kind='bar', stacked=True)
    plt.title(title)
    plt.xlabel('Premium Customer Flag')
    plt.ylabel('Proportion (%)')
    plt.legend(title='Lifestage', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    # Add percentage labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%')
    
    plt.show()

def analyze_purchase_behavior(data):
    """Analyze purchase behavior"""
    print("\nAnalyzing purchase behavior...")
    
    # Calculate average units per customer
    avg_units = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER']).agg({
        'PROD_QTY': lambda x: x.sum() / data['LYLTY_CARD_NBR'].nunique()
    }).round(2)
    
    # Calculate average price per unit
    data['PRICE_PER_UNIT'] = data['TOT_SALES'] / data['PROD_QTY']
    avg_price = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['PRICE_PER_UNIT'].mean().round(2)
    
    print("\nAverage units per customer:")
    print(avg_units)
    print("\nAverage price per unit:")
    print(avg_price)
    
    return avg_units, avg_price

def calculate_brand_affinity(data, target_segment):
    """Calculate brand affinity for a target segment"""
    print(f"\nCalculating brand affinity for {target_segment}...")
    
    target = data[data['LIFESTAGE'] == target_segment]
    others = data[data['LIFESTAGE'] != target_segment]
    
    target_props = target['BRAND'].value_counts(normalize=True)
    other_props = others['BRAND'].value_counts(normalize=True)
    
    affinity = (target_props / other_props).round(2)
    affinity = affinity.sort_values(ascending=False)
    
    print(f"\nBrand affinity for {target_segment}:")
    print(affinity.head(10))
    
    return affinity

def main():
    # Load data
    transaction_data, purchase_behavior = load_and_examine_data()
    
    # Clean transaction data
    transaction_data = clean_transaction_data(transaction_data)
    
    # Analyze product words
    product_words = analyze_product_words(transaction_data)
    
    # Merge transaction and customer data
    data = pd.merge(transaction_data, purchase_behavior, on='LYLTY_CARD_NBR', how='left')
    
    # Check for missing values after merge
    print("\nMissing values after merge:")
    print(data.isnull().sum())
    
    # Analyze customer segments
    sales_by_segment = analyze_customer_segments(data)
    
    # Plot sales proportions
    plot_segment_proportions(data, 'TOT_SALES', 'Proportion of Sales by Customer Segment')
    
    # Plot customer proportions
    customer_counts = data.groupby(['LIFESTAGE', 'PREMIUM_CUSTOMER'])['LYLTY_CARD_NBR'].nunique().reset_index()
    plot_segment_proportions(customer_counts, 'LYLTY_CARD_NBR', 'Proportion of Customers by Segment')
    
    # Analyze purchase behavior
    avg_units, avg_price = analyze_purchase_behavior(data)
    
    # Calculate brand affinity for young singles/couples
    brand_affinity = calculate_brand_affinity(data, 'YOUNG SINGLES/COUPLES')
    
    # Save processed data for future use
    data.to_csv('processed_data.csv', index=False)
    print("\nAnalysis complete! Processed data saved to 'processed_data.csv'")

if __name__ == "__main__":
    main() 