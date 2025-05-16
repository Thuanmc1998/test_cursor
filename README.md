<<<<<<< HEAD
# Retail Strategy and Analytics

This project analyzes transaction data and customer purchase behavior for a retail client, focusing on chips category.

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Make sure you have the following data files in the project directory:
- `QVI_transaction_data.xlsx`: Transaction data
- `QVI_purchase_behaviour.csv`: Customer behavior data

## Running the Analysis

Run the main analysis script:
```bash
python retail_analysis.py
```

The script will:
1. Load and clean the data
2. Analyze product names and brands
3. Analyze customer segments
4. Create visualizations for:
   - Sales proportions by customer segment
   - Customer proportions by segment
5. Analyze purchase behavior
6. Calculate brand affinity for young singles/couples
7. Save processed data to `processed_data.csv`

## Output

The script will generate:
- Console output with detailed analysis
- Visualizations (displayed during execution)
- `processed_data.csv` containing the cleaned and processed data

## Analysis Components

1. **Data Cleaning**:
   - Convert dates to proper format
   - Remove salsa products (non-chip products)
   - Remove outlier transactions
   - Extract pack sizes from product names
   - Clean and standardize brand names

2. **Customer Segmentation Analysis**:
   - Analysis by lifestage
   - Analysis by premium customer status
   - Sales distribution across segments
   - Customer distribution across segments

3. **Purchase Behavior Analysis**:
   - Average units per customer
   - Average price per unit
   - Brand preferences by segment

4. **Brand Affinity Analysis**:
   - Focus on young singles/couples segment
   - Compare brand preferences against other segments 
=======
# test_cursor
>>>>>>> 154de8d3799f96e68b704a7514048a9eeecb494b
