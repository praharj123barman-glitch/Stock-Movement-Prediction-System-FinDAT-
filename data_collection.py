import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def read_stock_list(csv_file):
    """Read stock list from CSV file."""
    df = pd.read_csv(csv_file)
    return df

def download_and_clean_stock_data(symbol, company_name, sector, start_date, end_date, output_dir):
    """Download stock data from Yahoo Finance and clean it in one step."""
    ticker = f"{symbol}.NS"
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data is not None and not data.empty:
            data['Symbol'] = symbol
            data['Company'] = company_name
            data['Sector'] = sector
            data = data.reset_index()  # Make 'Date' a column

            # Reorder and ensure required columns exist
            columns_order = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Sector']
            for col in columns_order:
                if col not in data.columns:
                    data[col] = ''

            data = data[columns_order]

            # Save to a temporary file first
            temp_file = os.path.join(output_dir, f"temp_{symbol}.csv")
            data.to_csv(temp_file, index=False)

            # Read file and remove 2nd line
            with open(temp_file, 'r') as f:
                lines = f.readlines()

            if len(lines) > 1:
                del lines[1]

            # Save final cleaned file
            output_file = os.path.join(output_dir, f"{symbol}.csv")
            with open(output_file, 'w') as f:
                f.writelines(lines)

            # Remove temp file
            os.remove(temp_file)

            print(f"Saved cleaned data to {output_file}")
            return True
        else:
            print(f"No data available for {symbol}")
            return False
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return False

def main(input_file, output_dir):
    """Main function to download and clean stock data."""
    os.makedirs(output_dir, exist_ok=True)
    stocks_df = read_stock_list(input_file)

    start_date = datetime(2022, 1,10 )
    end_date = datetime(2025, 1, 10)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Downloading data from {start_date_str} to {end_date_str}")

    successful_downloads = 0
    total_stocks = len(stocks_df)

    for i, row in stocks_df.iterrows():
        symbol = row['Symbol']
        company_name = row['Company Name']
        sector = row['Industry']

        print(f"Processing {i+1}/{total_stocks}: {symbol} ({company_name})")

        success = download_and_clean_stock_data(
            symbol, company_name, sector, start_date_str, end_date_str, output_dir
        )

        if success:
            successful_downloads += 1

    print(f"Completed processing. Successfully downloaded and cleaned {successful_downloads}/{total_stocks} stocks.")

if __name__ == "__main__":
    input_file = "ind_nifty500list_filtered_final-1.csv"
    output_dir = "data/cleaned"
    main(input_file, output_dir)
