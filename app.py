from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import json


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key

# Configuration
RESULTS_CSV = 'results_summary.csv'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize results file if it does not exist
if not os.path.isfile(RESULTS_CSV):
    df = pd.DataFrame(columns=['avg_profit', 'total_trades', 'total_profit', 'winning_trades', 'losing_trades'])
    df.to_csv(RESULTS_CSV, index=False)


@app.route('/')
def index():
    # Load existing summary data
    if os.path.exists(RESULTS_CSV):
        summary_data = pd.read_csv(RESULTS_CSV)
    else:
        summary_data = pd.DataFrame()

    # List all CSV files in the upload folder and gather metadata
    uploaded_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                df = pd.read_csv(filepath)
                rows, columns = df.shape
                uploaded_files.append({
                    'name': filename,
                    'rows': rows,
                    'columns': columns
                })
            except Exception as e:
                # If there's an error reading the CSV, skip the file and optionally log the error
                flash(f'Error processing file "{filename}": {e}')
                continue

    return render_template('index.html',
                           reports=summary_data.to_dict(orient='records'),
                           uploaded_files=uploaded_files)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(url_for('index'))

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        flash('No files selected for uploading.')
        return redirect(url_for('index'))

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(save_path)
                flash(f'File "{filename}" uploaded successfully.')

                # Process the uploaded CSV file
                orders_df = pd.read_csv(save_path)

                # Drop 'Balance' column if it exists
                if 'Balance' in orders_df.columns:
                    orders_df = orders_df.drop(columns=['Balance'])
                else:
                    flash(f'File "{filename}" does not contain a "Balance" column. Skipping this file.')
                    continue  # Skip processing this file if 'Balance' column is missing

                # Calculate metrics
                avg_profit = orders_df['P&L'].mean()
                total_trades = len(orders_df)
                total_profit = orders_df['P&L'].sum()
                winning_trades = len(orders_df.loc[orders_df['P&L'] >= 0])
                losing_trades = len(orders_df.loc[orders_df['P&L'] < 0])

                # Append metrics to the summary CSV
                new_entry = {
                    'avg_profit': avg_profit,
                    'total_trades': total_trades,
                    'total_profit': total_profit,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades
                }
                summary_df = pd.read_csv(RESULTS_CSV)
                summary_df = summary_df.append(new_entry, ignore_index=True)
                summary_df.to_csv(RESULTS_CSV, index=False)

            except Exception as e:
                flash(f'An error occurred while processing file "{filename}": {e}')
                continue
        else:
            flash(f'File "{file.filename}" is not a supported CSV file.')

    return redirect(url_for('index'))


@app.route('/orders', methods=['GET', 'POST'])
def orders():
    uploaded_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            uploaded_files.append(filename)

    selected_file = None
    table_data = None

    if request.method == 'POST':
        selected_file = request.form.get('csv_file')
        if selected_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    # Sort by the first column
                    first_column = df.columns[0]
                    df = df.sort_values(by=first_column)

                    table_data = df.to_dict(orient='records')
                    headers = df.columns.tolist()
                    return render_template('orders.html',
                                           uploaded_files=uploaded_files,
                                           selected_file=selected_file,
                                           headers=headers,
                                           table_data=table_data)
                except Exception as e:
                    flash(f'Error reading file "{selected_file}": {e}')
            else:
                flash(f'File "{selected_file}" does not exist.')

    return render_template('orders.html',
                           uploaded_files=uploaded_files,
                           selected_file=selected_file,
                           table_data=table_data)


# app.py

@app.route('/summary', methods=['GET', 'POST'])
def summary():
    uploaded_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            uploaded_files.append(filename)

    selected_file = None
    summary_table_symbol = None
    summary_table_side = None
    initial_balance = 100000.00  # Default initial balance

    # Mapping OrderType to side
    order_type_mapping = {
        1: 'buy',
        6: 'sell'
    }

    if request.method == 'POST':
        selected_file = request.form.get('csv_file')
        initial_balance_input = request.form.get('initial_balance')

        # Validate initial balance
        try:
            initial_balance = float(initial_balance_input)
            if initial_balance <= 0:
                flash('Initial balance must be a number greater than 0.')
                initial_balance = 100000.00  # Reset to default if invalid
        except (ValueError, TypeError):
            flash('Invalid initial balance. Please enter a valid number greater than 0.')
            initial_balance = 100000.00  # Reset to default if invalid

        if selected_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    # Check required columns
                    required_columns = ['OrderTicket', 'OrderType', 'Symbol', 'Volume', 'OpenPrice',
                                        'OpenTime', 'ClosePrice', 'CloseTime', 'Profit']
                    if not all(col in df.columns for col in required_columns):
                        flash(f'File "{selected_file}" is missing required columns.')
                        return redirect(url_for('summary'))

                    # Handle missing Symbol entries by filling with 'Unknown'
                    df['Symbol'] = df['Symbol'].fillna('Unknown')

                    # Compute 'G/P' based on 'Profit'
                    df['G/P'] = df['Profit'].apply(lambda x: 'G' if x >= 0 else 'P')

                    # Map 'OrderType' to 'side'
                    df['side'] = df['OrderType'].map(order_type_mapping).fillna('Unknown')

                    # Convert 'OpenTime' and 'CloseTime' to datetime
                    df['OpenTime'] = pd.to_datetime(df['OpenTime'], format='%Y.%m.%d %H:%M')
                    df['CloseTime'] = pd.to_datetime(df['CloseTime'], format='%Y.%m.%d %H:%M')

                    # Calculate 'Duración' in minutes
                    df['Duración'] = (df['CloseTime'] - df['OpenTime']).dt.total_seconds() / 60

                    # First Summary: Group by 'Symbol' and 'G/P'
                    grouped_symbol = df.groupby(['Symbol', 'G/P']).agg(
                        Count_id=('OrderTicket', 'count'),
                        Average_P_L=('Profit', 'mean'),
                        Average_Duración=('Duración', 'mean'),
                        Sum_P_L=('Profit', 'sum')
                    ).reset_index()

                    # Exclude rows where Symbol is 'Unknown'
                    grouped_symbol = grouped_symbol[grouped_symbol['Symbol'] != 'Unknown']

                    # Calculate total count per symbol for percentage calculations
                    total_counts_symbol = grouped_symbol.groupby('Symbol')['Count_id'].transform('sum')

                    # Calculate % Rentabilidad and %
                    grouped_symbol['% Rentabilidad'] = (grouped_symbol['Count_id'] * grouped_symbol['Average_P_L']) / initial_balance * 100
                    grouped_symbol['%'] = (grouped_symbol['Count_id'] / total_counts_symbol) * 100
                    grouped_symbol['Beneficio_Esperado'] = grouped_symbol['Count_id'] * grouped_symbol['Average_P_L']

                    # Replace NaN values with 0
                    grouped_symbol['% Rentabilidad'] = grouped_symbol['% Rentabilidad'].fillna(0)
                    grouped_symbol['%'] = grouped_symbol['%'].fillna(0)
                    grouped_symbol['Beneficio_Esperado'] = grouped_symbol['Beneficio_Esperado'].fillna(0)

                    # Round numerical values for better readability
                    grouped_symbol['Average_P_L'] = grouped_symbol['Average_P_L'].round(2)
                    grouped_symbol['Average_Duración'] = grouped_symbol['Average_Duración'].round(2)
                    grouped_symbol['Sum_P_L'] = grouped_symbol['Sum_P_L'].round(2)
                    grouped_symbol['% Rentabilidad'] = grouped_symbol['% Rentabilidad'].round(2)
                    grouped_symbol['%'] = grouped_symbol['%'].round(2)
                    grouped_symbol['Beneficio_Esperado'] = grouped_symbol['Beneficio_Esperado'].round(2)

                    # Prepare summary table data for Symbol
                    summary_table_symbol = grouped_symbol.to_dict(orient='records')

                    # Calculate total result for Symbol
                    total_count_symbol = grouped_symbol['Count_id'].sum()
                    total_avg_p_l_symbol = grouped_symbol['Average_P_L'].mean().round(2)
                    total_avg_duracion_symbol = grouped_symbol['Average_Duración'].mean().round(2)
                    total_sum_p_l_symbol = grouped_symbol['Sum_P_L'].sum().round(2)
                    total_beneficio_esperado_symbol = grouped_symbol['Beneficio_Esperado'].sum().round(2)

                    total_result_symbol = {
                        'Symbol': 'Total Result',
                        'G/P': '',
                        'Count_id': total_count_symbol,
                        'Average_P_L': '',
                        'Average_Duración': '',
                        'Sum_P_L': '',
                        '% Rentabilidad': '',
                        '%': '',
                        'Beneficio_Esperado': total_beneficio_esperado_symbol
                    }

                    summary_table_symbol.append(total_result_symbol)

                    # Second Summary: Group by 'side' and 'G/P'
                    grouped_side = df.groupby(['side', 'G/P']).agg(
                        Count_id=('OrderTicket', 'count'),
                        Average_P_L=('Profit', 'mean'),
                        Average_Duración=('Duración', 'mean'),
                        Sum_P_L=('Profit', 'sum')
                    ).reset_index()

                    # Exclude rows where side is 'Unknown'
                    grouped_side = grouped_side[grouped_side['side'] != 'Unknown']

                    # Calculate total count per side for percentage calculations
                    total_counts_side = grouped_side.groupby('side')['Count_id'].transform('sum')

                    # Calculate % Rentabilidad and %
                    grouped_side['% Rentabilidad'] = (grouped_side['Count_id'] * grouped_side['Average_P_L']) / initial_balance * 100
                    grouped_side['%'] = (grouped_side['Count_id'] / total_counts_side) * 100
                    grouped_side['Beneficio_Esperado'] = grouped_side['Count_id'] * grouped_side['Average_P_L']

                    # Replace NaN values with 0
                    grouped_side['% Rentabilidad'] = grouped_side['% Rentabilidad'].fillna(0)
                    grouped_side['%'] = grouped_side['%'].fillna(0)
                    grouped_side['Beneficio_Esperado'] = grouped_side['Beneficio_Esperado'].fillna(0)

                    # Round numerical values for better readability
                    grouped_side['Average_P_L'] = grouped_side['Average_P_L'].round(2)
                    grouped_side['Average_Duración'] = grouped_side['Average_Duración'].round(2)
                    grouped_side['Sum_P_L'] = grouped_side['Sum_P_L'].round(2)
                    grouped_side['% Rentabilidad'] = grouped_side['% Rentabilidad'].round(2)
                    grouped_side['%'] = grouped_side['%'].round(2)
                    grouped_side['Beneficio_Esperado'] = grouped_side['Beneficio_Esperado'].round(2)

                    # Prepare summary table data for side
                    summary_table_side = grouped_side.to_dict(orient='records')

                    # Calculate total result for side
                    total_count_side = grouped_side['Count_id'].sum()
                    total_avg_p_l_side = grouped_side['Average_P_L'].mean().round(2)
                    total_avg_duracion_side = grouped_side['Average_Duración'].mean().round(2)
                    total_sum_p_l_side = grouped_side['Sum_P_L'].sum().round(2)
                    total_beneficio_esperado_side = grouped_side['Beneficio_Esperado'].sum().round(2)

                    total_result_side = {
                        'side': 'Total Result',
                        'G/P': '',
                        'Count_id': total_count_side,
                        'Average_P_L': '',
                        'Average_Duración': '',
                        'Sum_P_L': '',
                        '% Rentabilidad': '',
                        '%': '',
                        'Beneficio_Esperado': total_beneficio_esperado_side
                    }

                    summary_table_side.append(total_result_side)

                    # Profit over time ------------------------------------------
                    df_sorted = df.sort_values(by='CloseTime')
                    df_sorted['CloseDate'] = df_sorted['CloseTime'].dt.date
                    daily_profit = df_sorted.groupby('CloseDate')['Profit'].sum().reset_index()
                    daily_profit['Cumulative_Profit'] = daily_profit['Profit'].cumsum()
                    daily_profit['Balance'] = initial_balance + daily_profit['Cumulative_Profit']
                    plot_data = daily_profit[['CloseDate', 'Balance']].copy()
                    plot_data['CloseDate'] = plot_data['CloseDate'].astype(str)
                    dates = plot_data['CloseDate'].tolist()
                    balances = plot_data['Balance'].tolist()

                    # New Code: Calculate Daily Open Orders for Bar Plot ------------------------
                    df_sorted['CloseDate_only'] = df_sorted['CloseTime'].dt.date
                    df_sorted['OpenDate_only'] = df_sorted['OpenTime'].dt.date
                    all_dates = pd.date_range(start=df_sorted['OpenDate_only'].min(),
                                              end=df_sorted['CloseDate'].max(),
                                              freq='D').date
                    open_orders_daily = []
                    for current_date in all_dates:
                        open_orders = df_sorted[
                            (df_sorted['OpenDate_only'] <= current_date) &
                            ((df_sorted['CloseDate_only'] > current_date) | (df_sorted['CloseDate_only'].isna()))
                        ]
                        open_orders_daily.append({'Date': current_date.strftime('%Y-%m-%d'), 'Open_Orders': open_orders.shape[0]})
                    open_orders_daily_df = pd.DataFrame(open_orders_daily)
                    open_dates = open_orders_daily_df['Date'].tolist()
                    open_orders = open_orders_daily_df['Open_Orders'].tolist()


                    flash(f'File "{selected_file}" processed successfully. Initial Balance: ${initial_balance:,.2f}')
                except Exception as e:
                    flash(f'Error processing file "{selected_file}": {e}')
            else:
                flash(f'File "{selected_file}" does not exist.')

    return render_template('summary.html',
                           uploaded_files=uploaded_files,
                           selected_file=selected_file,
                           summary_table_symbol=summary_table_symbol,
                           summary_table_side=summary_table_side,
                           initial_balance=initial_balance,
                           dates=json.dumps(dates),
                           balances=json.dumps(balances),
                           open_dates=json.dumps(open_dates),
                           open_orders=json.dumps(open_orders)
                       )


@app.route('/summary2', methods=['GET', 'POST'])
def summary2():
    uploaded_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if allowed_file(filename):
            uploaded_files.append(filename)

    summary_data = []
    initial_balance = 100000.00  # Default initial balance
    aggregated_metrics = {}
    corr_matrix_list = []

    if request.method == 'POST':
        initial_balance_input = request.form.get('initial_balance')

        try:
            initial_balance = float(initial_balance_input)
            if initial_balance <= 0:
                flash('Initial balance must be a number greater than 0.')
                initial_balance = 100000.00
        except (ValueError, TypeError):
            flash('Invalid initial balance. Please enter a valid number greater than 0.')
            initial_balance = 100000.00

        for selected_file in uploaded_files:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)

                    required_columns = ['OrderTicket', 'OrderType', 'Symbol', 'Volume', 'OpenPrice',
                                        'OpenTime', 'ClosePrice', 'CloseTime', 'Profit']
                    if not all(col in df.columns for col in required_columns):
                        flash(f'File "{selected_file}" is missing required columns.')
                        continue

                    df['Symbol'] = df['Symbol'].fillna('Unknown')
                    df['G/P'] = df['Profit'].apply(lambda x: 'G' if x >= 0 else 'P')
                    order_type_mapping = {
                        1: 'buy',
                        6: 'sell'
                    }
                    df['side'] = df['OrderType'].map(order_type_mapping).fillna('Unknown')
                    df['OpenTime'] = pd.to_datetime(df['OpenTime'], format='%Y.%m.%d %H:%M')
                    df['CloseTime'] = pd.to_datetime(df['CloseTime'], format='%Y.%m.%d %H:%M')
                    df['Duración'] = (df['CloseTime'] - df['OpenTime']).dt.total_seconds() / 60

                    ID = selected_file
                    operaciones = len(df)
                    p_l_promedio = df['Profit'].mean()
                    duracion_media = df['Duración'].mean()
                    rentabilidad = (df['Profit'].sum() / initial_balance) * 100

                    max_p_l_largos = df[df['side'] == 'buy']['Profit'].max()
                    if pd.isna(max_p_l_largos):
                        max_p_l_largos = 0

                    min_p_l_largos = df[df['side'] == 'buy']['Profit'].min()
                    if pd.isna(min_p_l_largos):
                        min_p_l_largos = 0

                    max_p_l_cortos = df[df['side'] == 'sell']['Profit'].max()
                    if pd.isna(max_p_l_cortos):
                        max_p_l_cortos = 0

                    min_p_l_cortos = df[df['side'] == 'sell']['Profit'].min()
                    if pd.isna(min_p_l_cortos):
                        min_p_l_cortos = 0

                    total_trades = len(df)
                    ganadoras = len(df[df['Profit'] > 0])
                    perdedoras = len(df[df['Profit'] < 0])
                    porcentaje_ganadoras = (ganadoras / total_trades) * 100 if total_trades > 0 else 0
                    porcentaje_perdedoras = (perdedoras / total_trades) * 100 if total_trades > 0 else 0

                    p_l_prom_ganadoras = df[df['Profit'] > 0]['Profit'].mean()
                    p_l_prom_ganadoras = p_l_prom_ganadoras if not pd.isna(p_l_prom_ganadoras) else 0

                    p_l_prom_perdedoras = df[df['Profit'] < 0]['Profit'].mean()
                    p_l_prom_perdedoras = p_l_prom_perdedoras if not pd.isna(p_l_prom_perdedoras) else 0

                    duracion_media_ganadoras = df[df['Profit'] > 0]['Duración'].mean()
                    duracion_media_ganadoras = duracion_media_ganadoras if not pd.isna(duracion_media_ganadoras) else 0

                    duracion_media_perdedoras = df[df['Profit'] < 0]['Duración'].mean()
                    duracion_media_perdedoras = duracion_media_perdedoras if not pd.isna(duracion_media_perdedoras) else 0

                    summary_data.append({
                        'ID': ID,
                        'Operaciones': operaciones,
                        'P_L_Promedio': round(p_l_promedio, 2),
                        'Duracion_Media': round(duracion_media, 2),
                        'Rentabilidad': round(rentabilidad, 2),
                        'Max_P_L_Largos': round(max_p_l_largos, 2),
                        'Min_P_L_Largos': round(min_p_l_largos, 2),
                        'Max_P_L_Cortos': round(max_p_l_cortos, 2),
                        'Min_P_L_Cortos': round(min_p_l_cortos, 2),
                        'Porcentaje_Ganadoras': round(porcentaje_ganadoras, 2),
                        'Porcentaje_Perdedoras': round(porcentaje_perdedoras, 2),
                        'P_L_Prom_Ganadoras': round(p_l_prom_ganadoras, 2),
                        'P_L_Prom_Perdedoras': round(p_l_prom_perdedoras, 2),
                        'Duracion_Media_Ganadoras': round(duracion_media_ganadoras, 2),
                        'Duracion_Media_Perdedoras': round(duracion_media_perdedoras, 2)
                    })
                except Exception as e:
                    flash(f'Error processing file "{selected_file}": {e}')
            else:
                flash(f'File "{selected_file}" does not exist.')

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            numeric_cols = ['Operaciones', 'P_L_Promedio', 'Duracion_Media', 'Rentabilidad',
                           'Max_P_L_Largos', 'Min_P_L_Largos', 'Max_P_L_Cortos', 'Min_P_L_Cortos',
                           'Porcentaje_Ganadoras', 'Porcentaje_Perdedoras',
                           'P_L_Prom_Ganadoras', 'P_L_Prom_Perdedoras',
                           'Duracion_Media_Ganadoras', 'Duracion_Media_Perdedoras']
            promedio = df_summary[numeric_cols].mean().round(2).to_dict()
            desv_t = df_summary[numeric_cols].std().round(2).to_dict()
            cv = (df_summary[numeric_cols].std() / df_summary[numeric_cols].mean()).round(3).to_dict()

            aggregated_metrics = {
                'Promedio': promedio,
                'Desv_T': desv_t,
                'CV': cv
            }

            corr_matrix = df_summary[numeric_cols].corr().round(3).to_dict()
            corr_matrix_list = []
            for row in numeric_cols:
                row_dict = {'Metric': row}
                for col in numeric_cols:
                    row_dict[col] = corr_matrix[row][col]
                corr_matrix_list.append(row_dict)

            return render_template('summary2.html',
                                   uploaded_files=uploaded_files,
                                   summary_data=summary_data,
                                   aggregated_metrics=aggregated_metrics,
                                   corr_matrix=corr_matrix_list,
                                   initial_balance=initial_balance)

    return render_template('summary2.html',
                           uploaded_files=uploaded_files,
                           summary_data=summary_data,
                           aggregated_metrics=aggregated_metrics,
                           corr_matrix=corr_matrix_list,
                           initial_balance=initial_balance)

if __name__ == "__main__":
    app.run(debug=True)
