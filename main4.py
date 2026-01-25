import pandas as pd
import numpy as np


def ReadFile(filename):
    """
    Чтение файлов .dat
    """
    # Читаем файл и сохраняем в DataFrame
    df = pd.read_csv(filename, 
                 delimiter='\s+',  
                 header=None)      
    df.columns = df.iloc[0]
    df = df[1:]
    df = df.reset_index(drop=True)
    df.index.name = None
    
    # Делаем все данные типом float
    df = df.astype(float)
    return(df)
    

def calculate_second_average(df, time_column='TimeImu'):
    """
    Вычисляет среднесекундные значения.
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
    new_values = [0]
    for i in range(1, len(df)):
        new_values.append(new_values[i-1] + 0.01)

    df['TimeImu'] = new_values
    
    return df

def calculate_initial_angles(df):
    """
    Вычисляет углы крена и тангажа из данных акселерометров.
    
    Args:
        df: DataFrame с колонками Ax, Ay, Az
        
    Returns:
        DataFrame: DataFrame с добавленными колонками roll и pitch
    """

    g = 9.80666
    g_array = np.full_like(df['Ay'], g)
    # Создаем копию DataFrame для сохранения результатов
    result = df.copy()
    
    # Вычисляем углы в радианах 
    """
    Эти формулы аналогичны представленным в учебнике "Основы_построения_БИНС" В_В_Матвеева и др. в главе 3.5 Начальная выставка БИНС, 162 стр.
    Они считаются для поворота YZX, OY перпендикулярна плоскости аппарата и направлена вверх.
    """
    result['roll'] = np.arctan2(-df['Az'], df['Ay']) # Угол крена
    result['pitch'] = np.arcsin(df['Ax']/g) # Угол тангажа


    """"
    result['roll'] = np.arctan2(df['Ay'], df['Az']/g_array) #Угол крена
    result['pitch'] = np.arctan2(-df['Ax'], np.sqrt(df['Ay']**2 + df['Az']**2)) #Угол тангажа
    """
    """  
    result['roll'] = np.arctan2(-df['Ay'], df['Az']) #Угол крена
    result['pitch'] = np.arcsin(df['Ax']/g) #Угол тангажа
    """
    """
    result['roll_acc'] = np.arctan2(df['Ay'], df['Az']) #Угол крена
    result['pitch_acc'] = np.arctan2(-df['Ax'], np.sqrt(df['Ay']**2 + df['Az']**2)) #Угол тангажа
    """
    return result



def gyro_bias_check(calib_data, bias, calibration_time):
    """
    Вывод информации про посчитанное смещение нулей гироскопов
    """
    print("\nСмещения нулей гироскопов (bias):")
    print(f"Wx = {bias['Wx']:10.6f} рад/с  ({np.degrees(bias['Wx']):8.4f} град/с)")
    print(f"Wy = {bias['Wy']:10.6f} рад/с  ({np.degrees(bias['Wy']):8.4f} град/с)")  
    print(f"Wz = {bias['Wz']:10.6f} рад/с  ({np.degrees(bias['Wz']):8.4f} град/с)")

        # Вычисляем стандартные отклонения
    std = {
        'Wx': np.std(calib_data['Wx']),
        'Wy': np.std(calib_data['Wy']),
        'Wz': np.std(calib_data['Wz'])
    }
    
    print("\nСтандартные отклонения:")
    print(f"σ_Wx = {std['Wx']:10.6f} рад/с  ({np.degrees(std['Wx']):8.4f} град/с)")
    print(f"σ_Wy = {std['Wy']:10.6f} рад/с  ({np.degrees(std['Wy']):8.4f} град/с)")
    print(f"σ_Wz = {std['Wz']:10.6f} рад/с  ({np.degrees(std['Wz']):8.4f} град/с)")
 

    # Оценка дрейфа за время калибровки
    drift_angle = {
        'X': bias['Wx'] * calibration_time,
        'Y': bias['Wy'] * calibration_time,
        'Z': bias['Wz'] * calibration_time
    }
    
    print(f"\nОжидаемый угловой дрейф за {calibration_time} сек:")
    print(f"По X: {np.degrees(drift_angle['X']):8.4f} градусов")
    print(f"По Y: {np.degrees(drift_angle['Y']):8.4f} градусов")
    print(f"По Z: {np.degrees(drift_angle['Z']):8.4f} градусов")

    print()
    # Проверяем, что объект был неподвижен с помощью акселерометров
    g = 9.80665
    accel_norm = np.sqrt(calib_data['Ax']**2 + calib_data['Ay']**2 + calib_data['Az']**2)
    norm_mean = accel_norm.mean()
    norm_std = accel_norm.std()
    
    print(f"Норма ускорения: {norm_mean:.4f} ± {norm_std:.4f} м/с*c")
    print(f"Отклонение от g: {abs(norm_mean - g):.4f} м/с*c")
    
    if abs(norm_mean - g) < 0.1:
        print("Объект был неподвижен (норма ускорения близка к g)")
    else:
        print("Возможно, объект двигался")
    

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
    calib_data = df.loc[calib_mask]
    gyro_bias_check(calib_data, bias, calibration_time)
    print()
    
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
    


def main():
    # Основная функция программы

    #filename = "Out_00048_car_6.7.dat"
    filename = 'Out_00084_car_6.9_1_.dat'
    df = ReadFile(filename)
    
    df = update_timeimu_column(df)
    #df.to_excel("output.xlsx", index=False, engine='openpyxl')
    #df.to_csv("output.csv", index=False, encoding='utf-8')
    
    #first_seven = df.iloc[:, :7]
    #specific_cols = df[['PitchI', 'RollI']]
    #df = pd.concat([first_seven, specific_cols], axis=1)

    df = df.iloc[:, :7]
    
    
    new_df = calculate_second_average(df)
    new_df = calculate_initial_angles(new_df)

    new_df, bias = gyro_bias(new_df)
    print_df(new_df)


if __name__ == "__main__":
    main()