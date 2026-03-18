from constants import U0, R0, G0
from data_io import ReadFile, calculate_second_average, update_timeimu_column, gyro_bias, print_df
from Poisson import ins_2poisson

#from tests import generate_test_df
from Utils import compare_ins_results

import pandas as pd

def main():
    """ Основная функция программы """

    #filename = "Out_00048_car_6.7.dat"
    filename = 'Out_00084_car_6.9_1_.dat'
    df = ReadFile(filename)
    
    df = update_timeimu_column(df)
    #df.to_excel("output.xlsx", index=False, engine='openpyxl')
    #df.to_csv("output.csv", index=False, encoding='utf-8')

    # Берём нужные колонки для работы
    first_seven = df.iloc[:, :7]
    specific_cols = df[['LonI', 'LatI', 'HeightI', 'VeI', 'VnI', 'VupI']]
    df = pd.concat([first_seven, specific_cols], axis=1)
    
    df = calculate_second_average(df)

    df, bias = gyro_bias(df) # вычитаем смещение гироскопов

    #test = -1 # я делала пару простых тестов, позже уберу их из кода. test = -1 это работа с данными из файлов, а не с тестовыми
    #if test != -1:
    #    df = generate_test_df(test, duration=1000)
    
    print_df(df)
    lon0, lat0, h0 = df['LonI'].iloc[0], df['LatI'].iloc[0], df['HeightI'].iloc[0]
    dt = 1
    D_list, Cu_list, C_list, lat_arr, lon_arr, h_arr, Vn_arr, Ve_arr, Vu_arr = ins_2poisson(df, dt, lat0, lon0, h0, test = -1)
    
    errors = compare_ins_results(df,
                             lat_arr, lon_arr, h_arr,
                             Vn_arr, Ve_arr, Vu_arr,
                             N=100, plot=True)

    #print(df.columns.tolist())


if __name__ == "__main__":
    main()