from constants import U0, R0, G0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gyro_bias_check(calib_data, bias):
    """
    Вывод информации про посчитанное смещение нулей гироскопов
    """
    print("\nСмещения нулей гироскопов (bias):")
    print(f"Wx = {bias['Wx']:10.6f} рад/с")
    print(f"Wy = {bias['Wy']:10.6f} рад/с")  
    print(f"Wz = {bias['Wz']:10.6f} рад/с")

    # Вычисляем стандартные отклонения
    std = {
        'Wx': np.std(calib_data['Wx']),
        'Wy': np.std(calib_data['Wy']),
        'Wz': np.std(calib_data['Wz'])
    }
    
    print("\nСтандартные отклонения:")
    print(f"для Wx = {std['Wx']:10.6f} рад/с ")
    print(f"для Wy = {std['Wy']:10.6f} рад/с ")
    print(f"для Wz = {std['Wz']:10.6f} рад/с ")

    print()
    # Проверяем, что объект был неподвижен с помощью акселерометров
    g = G0
    accel_norm = np.sqrt(calib_data['Ax']**2 + calib_data['Ay']**2 + calib_data['Az']**2)
    norm_mean = accel_norm.mean()
    norm_std = accel_norm.std()
    
    print(f"Норма ускорения: {norm_mean:.4f} +- {norm_std:.4f} м/с*c")
    print(f"Отклонение от g: {abs(norm_mean - g):.4f} м/с*c")
    


def compare_ins_results(df, lat_arr, lon_arr, h_arr, Vn_arr, Ve_arr, Vu_arr, N=250, plot=True):
    # используется для проверки сейчас, в финальной версии работы не будет. написано с помощью ИИ
    """
    Сравнивает первые N значений INS-решения из df с рассчитанными массивами.
    
    Параметры
    ---------
    df : pd.DataFrame
        Должен содержать колонки: LatI, LonI, HeightI, VeI, VnI, VupI.
    lat_arr, lon_arr, h_arr : array-like
        Рассчитанные широта (град/рад), долгота (град/рад), высота (м).
    Vn_arr, Ve_arr, Vu_arr : array-like
        Рассчитанные скорости север/восток/вверх (м/с).
    N : int
        Количество первых точек для сравнения.
    plot : bool
        Построить графики сравнения.
    
    Возвращает
    --------
    dict : метрики ошибок для каждой переменной.
    """
    # --- 1. Подготовка эталонных данных ---
    # Берём первые N значений из df (обрезаем до минимума из N и длины df)
    N = min(N, len(df))
    ref_lat = df['LatI'].iloc[:N].values      # широта эталонная
    ref_lon = df['LonI'].iloc[:N].values      # долгота эталонная
    ref_h   = df['HeightI'].iloc[:N].values   # высота эталонная
    ref_Ve  = df['VeI'].iloc[:N].values       # восточная скорость
    ref_Vn  = df['VnI'].iloc[:N].values       # северная скорость
    ref_Vu  = df['VupI'].iloc[:N].values      # вертикальная скорость

    # --- 2. Подготовка расчётных данных (обрезаем до N) ---
    calc_lat = np.array(lat_arr[:N])
    calc_lon = np.array(lon_arr[:N])
    calc_h   = np.array(h_arr[:N])
    calc_Vn  = np.array(Vn_arr[:N])
    calc_Ve  = np.array(Ve_arr[:N])
    calc_Vu  = np.array(Vu_arr[:N])

    # --- 3. Функция вычисления метрик ---
    def compute_metrics(ref, calc, name):
        diff = calc - ref
        abs_diff = np.abs(diff)
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean(diff**2))
        maxae = np.max(abs_diff)
        # Относительная ошибка (если возможно)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(diff) / (np.abs(ref) + 1e-12)
            mape = np.mean(rel_diff) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MaxAE': maxae,
            'MAPE(%)': mape
        }
        return metrics, diff

    # --- 4. Вычисляем метрики для каждой величины ---
    errors = {}
    diffs = {}
    
    metrics_lat, diff_lat = compute_metrics(ref_lat, calc_lat, 'lat')
    metrics_lon, diff_lon = compute_metrics(ref_lon, calc_lon, 'lon')
    metrics_h,   diff_h   = compute_metrics(ref_h,   calc_h,   'h')
    metrics_Ve,  diff_Ve  = compute_metrics(ref_Ve,  calc_Ve,  'Ve')
    metrics_Vn,  diff_Vn  = compute_metrics(ref_Vn,  calc_Vn,  'Vn')
    metrics_Vu,  diff_Vu  = compute_metrics(ref_Vu,  calc_Vu,  'Vu')
    
    errors['lat'] = metrics_lat
    errors['lon'] = metrics_lon
    errors['h']   = metrics_h
    errors['Ve']  = metrics_Ve
    errors['Vn']  = metrics_Vn
    errors['Vu']  = metrics_Vu
    
    diffs['lat'] = diff_lat
    diffs['lon'] = diff_lon
    diffs['h']   = diff_h
    diffs['Ve']  = diff_Ve
    diffs['Vn']  = diff_Vn
    diffs['Vu']  = diff_Vu

    # --- 5. Вывод статистики ---
    print(f"Сравнение первых {N} отсчётов")
    print("-" * 70)
    header = f"{'Величина':<12} {'MAE':<12} {'RMSE':<12} {'MaxAE':<12} {'MAPE(%)':<12}"
    print(header)
    print("-" * 70)
    for name in ['lat', 'lon', 'h', 'Vn', 'Ve', 'Vu']:
        m = errors[name]
        print(f"{name:<12} {m['MAE']:<12.6f} {m['RMSE']:<12.6f} {m['MaxAE']:<12.6f} {m['MAPE(%)']:<12.6f}")
    print("-" * 70)

    # --- 6. Опциональные графики ---
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Сравнение INS: первые {N} отсчётов', fontsize=14)
        
        # Широта
        ax = axes[0, 0]
        ax.plot(ref_lat, label='Эталон (LatI)', linewidth=1)
        ax.plot(calc_lat, '--', label='Расчёт (lat_arr)', linewidth=1)
        ax.set_title('Широта')
        ax.legend()
        ax.grid(True)
        
        # Долгота
        ax = axes[0, 1]
        ax.plot(ref_lon, label='Эталон (LonI)', linewidth=1)
        ax.plot(calc_lon, '--', label='Расчёт (lon_arr)', linewidth=1)
        ax.set_title('Долгота')
        ax.legend()
        ax.grid(True)
        
        # Высота
        ax = axes[1, 0]
        ax.plot(ref_h, label='Эталон (HeightI)', linewidth=1)
        ax.plot(calc_h, '--', label='Расчёт (h_arr)', linewidth=1)
        ax.set_title('Высота, м')
        ax.legend()
        ax.grid(True)
        
        # Северная скорость
        ax = axes[1, 1]
        ax.plot(ref_Vn, label='Эталон (VnI)', linewidth=1)
        ax.plot(calc_Vn, '--', label='Расчёт (Vn_arr)', linewidth=1)
        ax.set_title('Vn, м/с')
        ax.legend()
        ax.grid(True)
        
        # Восточная скорость
        ax = axes[2, 0]
        ax.plot(ref_Ve, label='Эталон (VeI)', linewidth=1)
        ax.plot(calc_Ve, '--', label='Расчёт (Ve_arr)', linewidth=1)
        ax.set_title('Ve, м/с')
        ax.legend()
        ax.grid(True)
        
        # Вертикальная скорость
        ax = axes[2, 1]
        ax.plot(ref_Vu, label='Эталон (VupI)', linewidth=1)
        ax.plot(calc_Vu, '--', label='Расчёт (Vu_arr)', linewidth=1)
        ax.set_title('Vu, м/с')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return errors
