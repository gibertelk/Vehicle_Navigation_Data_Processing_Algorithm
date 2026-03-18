import pandas as pd

def ReadFile(filename):
    """
    Чтение файлов .dat
    """
    # Читаем файл и сохраняем в DataFrame
    df = pd.read_csv(filename, 
                 delimiter='\s+',  
                 header=None,
                 low_memory=False)      
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index(drop=True)
    df.index.name = None
    
    # Делаем все данные типом float
    df = df.astype(float)
    return(df)
    

def calculate_second_average(df, time_column='TimeImu'):
    """
    Вычисляет среднесекундные значенияs
    """
    # Создаем словарь для хранения сумм и счетчиков для каждой секунды
    second_sums = {}
    second_counts = {}
    
    # Проходим по всем строкам DataFrame
    for _, row in df.iterrows():
        # Округляем время вниз до целой секунды
        second = int(row[time_column])
        
        for col in df.columns:
            if col != time_column:
                # Добавляем значение в сумму для этой секунды
                second_sums.setdefault(second, {}).setdefault(col, 0)
                second_sums[second][col] += row[col]
                
                # Увеличиваем счетчик для этой секунды
                second_counts.setdefault(second, {}).setdefault(col, 0)
                second_counts[second][col] += 1
    
    result = []
    seconds = sorted(second_sums.keys()) 
    
    # Вычисляем среднее для каждой секунды и колонки
    for second in seconds:
        row = {}
        for col in second_sums[second]:
            row[col] = second_sums[second][col] / second_counts[second][col]
        result.append(row)
    
    # DataFrame с результатами
    result_df = pd.DataFrame(result)
    result_df.insert(0, 'TimeImu', seconds) 
    #result_df = result_df.reset_index(drop=True)
    return result_df

def update_timeimu_column(df):
    """
    Обновляет столбец TimeImu: первое значение сохраняется, последующие значения = предыдущее значение + 0.1
    """
    
    if len(df) == 0:
        return df
    
    # Сохраняем первое значение
    first_value = df['TimeImu'].iloc[0]
    
    # Создаем новые значения
    new_values = [first_value]
    #new_values = [0]
    for i in range(1, len(df)):
        new_values.append(new_values[i-1] + 0.01)

    df['TimeImu'] = new_values
    
    return df

def gyro_bias(df, calibration_time=30):
    """
    Высчитывает и вычитает смещения нулей гироскопов на начальном этапе
    """
    # Определяем смещения на начальном участке
    start_time = df['TimeImu'].iloc[0]
    end_time = start_time + calibration_time
    
    calib_mask = (df['TimeImu'] >= start_time) & (df['TimeImu'] <= end_time)

    bias = {
        'Wx': df.loc[calib_mask, 'Wx'].mean(),
        'Wy': df.loc[calib_mask, 'Wy'].mean(),
        'Wz': df.loc[calib_mask, 'Wz'].mean()
    }

    # Проверка и вывод информации про посчитанное смещение 
    # Можно закомментировать, если не нужно
    #calib_data = df.loc[calib_mask]
    #gyro_bias_check(calib_data, bias)
    #print()
    
    df_calibrated = df.copy()
    df_calibrated['Wx_cal'] = df['Wx'] - bias['Wx']
    df_calibrated['Wy_cal'] = df['Wy'] - bias['Wy']
    df_calibrated['Wz_cal'] = df['Wz'] - bias['Wz']
    
    return df_calibrated, bias

def print_df(df, n_rows=10, n_cols=None, max_col_width=30, float_precision=4):
    """
    Вывод DataFrame в консоль
    """
    
    # Сохраняем текущие настройки
    current_max_rows = pd.get_option('display.max_rows')
    current_max_columns = pd.get_option('display.max_columns')
    current_width = pd.get_option('display.width')
    current_max_colwidth = pd.get_option('display.max_colwidth')
    current_float_format = pd.get_option('display.float_format')
    
    # Устанавливаем временные настройки
    pd.set_option('display.max_rows', n_rows if n_rows else len(df))
    pd.set_option('display.max_columns', n_cols if n_cols else len(df.columns))
    pd.set_option('display.width', None) 
    pd.set_option('display.max_colwidth', max_col_width)
    pd.set_option('display.float_format', lambda x: f'{x:.{float_precision}f}' 
                  if isinstance(x, (float, pd.core.dtypes.common.is_float_dtype)) else str(x))
    
    # Выводим DataFrame
    if n_rows:
        display_df = df.head(n_rows)
    else:
        display_df = df
    
    if n_cols:
        display_df = display_df.iloc[:, :n_cols]
    
    with pd.option_context('display.colheader_justify', 'left',
                          'display.expand_frame_repr', True,
                          'display.max_colwidth', max_col_width):
        print(display_df.to_string(col_space=8, index=False)) 
    
    # Восстанавливаем настройки
    pd.set_option('display.max_rows', current_max_rows)
    pd.set_option('display.max_columns', current_max_columns)
    pd.set_option('display.width', current_width)
    pd.set_option('display.max_colwidth', current_max_colwidth)
    pd.set_option('display.float_format', current_float_format)