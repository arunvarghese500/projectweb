import os
import streamlit as st
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2025, 1, 1)

# Fetch data using yfinance
AIB = yf.download("AIBG.L", start=start, end=end) 
BOI = yf.download("BIRG.IR", start=start, end=end)
Intesa = yf.download("ISP.MI", start=start, end=end)
Citibank = yf.download("C", start=start, end=end)

AIB_df = pd.DataFrame(AIB)
BOI_df = pd.DataFrame(BOI)
Intesa_df = pd.DataFrame(Intesa)
Citibank_df = pd.DataFrame(Citibank)

# st.write(AIB_df.shape)
# st.write(BOI_df.shape)
# st.write(Intesa_df.shape)
# st.write(Citibank_df.shape)



# Fetch data using yfinance
tickers = {
    "AIB": "AIBG.L",
    "BOI": "BIRG.IR",
    "Intesa": "ISP.MI",
    "Citibank": "C"
}
data = {}

for ticker_name, ticker_symbol in tickers.items():
    try:
        data[ticker_name] = yf.download(ticker_symbol, start=start, end=end)
        print(f"Downloaded data for {ticker_name}")
    except Exception as e:
        print(f"Error downloading data for {ticker_name}: {e}")

# Data Cleaning and Handling Missing Values
for ticker_name, df in data.items():
    # Handle missing values (e.g., forward fill)
    df.fillna(method='ffill', inplace=True)
    data[ticker_name] = df

# Exploratory Data Analysis (EDA)
for ticker_name, df in data.items():
    print(f"\nEDA for {ticker_name}:")
    print(df.describe())  # Summary statistics
    
    # Plotting closing prices
    plt.figure(figsize=(10, 6))
    plt.title(f'{ticker_name} Closing Price')
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()

    # Plotting volume traded
    plt.figure(figsize=(10, 6))
    plt.title(f'{ticker_name} Volume Traded')
    plt.plot(df['Volume'])
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.show()

    # Add more EDA plots (histograms, box plots, etc.) as needed.

# Financial Analytics
for ticker_name, df in data.items():
    print(f"\nFinancial Analytics for {ticker_name}:")

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Calculate cumulative returns
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()

    print("Cumulative Return:")
    print(df['Cumulative_Return'].tail(1)) # Show last value of cumulative return

    # Other financial metrics (e.g., moving averages, volatility, etc.) can be added here

    # Example: Calculate 50-day and 200-day moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # Plot closing price with moving averages
    plt.figure(figsize=(12,6))
    plt.title(f'{ticker_name} Closing Price with Moving Averages')
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['MA50'], label='MA50')
    plt.plot(df['MA200'], label='MA200')
    plt.legend()
    plt.show()


for ticker_name, df in data.items():
    print(f"\nMachine Learning for {ticker_name}:")
    
    # Prepare data for linear regression
    df['Close_shifted'] = df['Close'].shift(-1) # Predict next day's close
    df.dropna(inplace=True) # Drop rows with NaN (due to shifting)


    X = df[['Open', 'High', 'Low', 'Volume']]  # Features
    y = df['Close_shifted']  # Target variable
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot actual vs. predicted closing prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{ticker_name} - Actual vs. Predicted Closing Prices')
    plt.legend()
    plt.show()



st.title("Bank Stock Analysis Dashboard")

# Define the path to the video folder
video_folder = r'C:\Users\divya.pasupuleti\Downloads\website\videos'

# Function to list and display videos from the folder
def play_videos_from_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        st.write("No videos found in the folder.")
        return

    st.write("### Available CSR Videos")
    for video in video_files:
        video_path = os.path.join(folder_path, video)
        st.write(f"#### {video}")
        st.video(video_path)

# Streamlit app layout
def main():
    # Title of the website
    st.title('Sustainable Development and CSR in Banking')

    # Introductory Text
    st.write("""
        Welcome to the platform dedicated to **Sustainable Development** and **Corporate Social Responsibility (CSR)**.
        This website aims to provide thorough knowledge and educational resources about CSR initiatives, especially in the context of banking, with a focus on the actions taken by Ireland's financial institutions.
    """)

    # Navigation
    menu = ["Home", "About CSR", "Ireland Banks' Initiatives", "Case Studies", "Tools for Goal Creation", "Educational Materials"]
    choice = st.sidebar.selectbox("Select a page", menu)

    # Content for the selected page
    if choice == "Home":
        st.subheader('Welcome to our CSR and Sustainable Development platform!')
        st.write("""
            Here, you can explore the importance of **CSR** and how it impacts the environment, society, and the economy. 
            The banking sector plays a crucial role in promoting sustainability and ethical business practices.
        """)
    elif choice == "About CSR":
        st.subheader('About Corporate Social Responsibility (CSR)')
        st.write("""
            **Corporate Social Responsibility (CSR)** is a business model that helps companies become more responsible for their actions. 
            It encourages businesses to consider the social, environmental, and economic effects of their decisions.
        """)

        # Call the function to play videos from the folder
        play_videos_from_folder(video_folder)

    elif choice == "Ireland Banks' Initiatives":
        st.subheader('CSR Initiatives by Ireland Banks')
        st.write("""
        Discover the various CSR initiatives undertaken by banks in Ireland to promote sustainability and societal well-being. 
        These include green banking, social investments, ethical lending, and more.
        """)
        st.write("""
        Here are some key CSR initiatives by Irish banks:
        
        1. **Green Banking Programs**  
           - Promoting eco-friendly financial products such as green loans, mortgages, and bonds to support renewable energy and sustainable projects.  
        
        2. **Community Investment Initiatives**  
           - Funding educational programs, financial literacy workshops, and community development projects to foster local growth.  
        
        3. **Diversity and Inclusion Programs**  
           - Enhancing workplace diversity through inclusive hiring practices and employee resource groups supporting underrepresented communities.  
        
        4. **Ethical and Responsible Lending**  
           - Adopting strict policies to avoid lending to industries with negative social or environmental impacts, such as fossil fuels or arms manufacturing.  
        
        5. **Support for SMEs and Startups**  
           - Offering tailored financial products and mentorship programs to help small businesses thrive and innovate.  
        
        6. **Carbon Neutral Commitments**  
           - Implementing measures to reduce their carbon footprint, such as energy-efficient buildings and digital banking to reduce paper usage.  
        
        7. **Charitable Contributions**  
           - Donating to charities and NGOs focused on healthcare, education, poverty alleviation, and disaster relief.  
        
        8. **Employee Volunteering Programs**  
           - Encouraging employees to engage in community service through paid volunteer days and partnerships with local organizations.  
        
        9. **Sustainable Supply Chain Practices**  
           - Working with suppliers that follow sustainable and ethical practices, ensuring a positive impact throughout the value chain.  
        
        10. **Digital Inclusion and Accessibility**  
           - Developing banking services accessible to all, including people with disabilities and those in underserved regions, to bridge the digital divide.
        """)

    
    elif choice == "Case Studies":

        st.subheader('Successful CSR Case Studies')
        st.write("""
        Browse through case studies showcasing the successful implementation of CSR projects by various banks around the world.
        """)

        # st.write(data)

        # Display all the graphs and analysis results here.
        # for ticker_name, df in data.items():
        #     st.write(df['Close'].head())
        #     st.write(f"## Analysis for {ticker_name}")

        #     # EDA plots
        #     st.write("### Closing Price")
        #     st.line_chart(df['Close'])

        #     st.write("### Volume Traded")
        #     st.line_chart(df['Volume'])

        #     # Financial analytics plots
        #     st.write("### Cumulative Return")
        #     st.line_chart(df['Cumulative_Return'])

        #     st.write("### Closing Price with Moving Averages")
        #     st.line_chart(df[['Close', 'MA50', 'MA200']])

        #     # Machine learning results
        #     st.write("### Machine Learning - Actual vs. Predicted Closing Prices")
        #     fig, ax = plt.subplots()
        #     ax.plot(df['Close_shifted'].values[-len(y_pred):], label='Actual') # Fixed plotting issue
        #     ax.plot(y_pred, label='Predicted')
        #     ax.set_title(f'{ticker_name} - Actual vs. Predicted Closing Prices')
        #     ax.legend()
        #     st.pyplot(fig) # Use st.pyplot() to display matplotlib plots in Streamlit

        #     st.write(f"RMSE: {rmse}")

        for ticker_name, df in data.items():
            # st.write(df.columns)
            df.columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'Daily_Return', 'Cumulative_Return','MA50','MA200','Close_shifted']

            if 'Close' in df.columns:
                df['MA50'] = df['Close'].rolling(window=50).mean()
                df['MA200'] = df['Close'].rolling(window=200).mean()
                df['Cumulative_Return'] = (df['Close'] / df['Close'].iloc[0]) - 1
                df['Close_shifted'] = df['Close'].shift(-1)

        # Visualizations
        for ticker_name, df in data.items():
            st.write(f"## Analysis for {ticker_name}")
            # st.write("### Data Preview")
            # st.write(df.head())

            # Closing Price
            if 'Close' in df.columns:
                st.write("### Closing Price")
                st.line_chart(df['Close'])
            else:
                st.write("")

            # Volume Traded
            if 'Volume' in df.columns:
                st.write("### Volume Traded")
                st.line_chart(df['Volume'])
            else:
                st.write("")

            # Cumulative Return
            if 'Cumulative_Return' in df.columns:
                st.write("### Cumulative Return")
                st.line_chart(df['Cumulative_Return'])
            else:
                st.write("")

            # Closing Price with Moving Averages
            if {'Close', 'MA50', 'MA200'}.issubset(df.columns):
                st.write("### Closing Price with Moving Averages")
                st.line_chart(df[['Close', 'MA50', 'MA200']])
            else:
                st.write("")

            # Machine Learning results
            if 'Close_shifted' in df.columns and 'y_pred' in locals():
                st.write("### Machine Learning - Actual vs. Predicted Closing Prices")
                fig, ax = plt.subplots()
                ax.plot(df['Close_shifted'][-len(y_pred):], label='Actual')
                ax.plot(y_pred, label='Predicted')
                ax.set_title(f'{ticker_name} - Actual vs. Predicted Closing Prices')
                ax.legend()
                st.pyplot(fig)
                st.write(f"RMSE: {rmse}")
            else:
                st.write("")

    
    elif choice == "Tools for Goal Creation":
        st.subheader('Tools for Creating CSR Goals')
        st.write("""
            Learn about different tools and frameworks to create and track CSR goals effectively. 
            These tools will guide organizations in integrating sustainability into their operations.
        """)
    
    elif choice == "Educational Materials":
        st.subheader('Educational Resources on CSR')
        st.write("""
            Access various educational materials, such as articles, research papers, and reports, that can enhance your understanding of CSR and sustainable development.
        """)

# Run the Streamlit app
if __name__ == '__main__':
    main()
