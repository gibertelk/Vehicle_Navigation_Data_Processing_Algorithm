from constants import U0, R0, G0, Rp
import pandas as pd
import numpy as np
import math

def angles_from_accelerometer(Ax, Ay, Az):
    # В.В. Матвеев, Основы построения БИНС, стр 162
    #print (Ax, Ay, Az)
    roll = np.arctan2(-Az, Ay)
    pitch = np.arcsin(Ax/G0)
    return (roll, pitch)


"""
def initial_yaw(wx, pitch, lat):
    # стр 162
    # явно недостаточно точные гироскопы для этой формулы
    print('wx ', wx)
    print (U0*np.cos(lat)*np.cos(pitch))
    cos_yaw = (wx - U0*np.sin(pitch)*np.sin(lat))/(U0*np.cos(lat)*np.cos(pitch))
    print('cos yaw ', cos_yaw)
    cos_yaw = max(-1.0, min(1.0, cos_yaw))
    yaw = math.acos(cos_yaw)
    return yaw
"""

def skew(v):
    """Кососимметрическая матрица"""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0]
    ])

def rotation_matrix_yzx(yaw, pitch, roll):
    """
    Матрица поворота C из связанной в географическую систему.
    В.В. Матвеев, Основы построения БИНС, стр 129
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cp*cy,  -cr*cy*sp + sr*sy,  sr*cy*sp + cr*sy],
        [sp,      cr*cp,            -sr*cp],
        [-cp*sy,  cr*sy*sp + sr*cp, -sr*sy*sp + cr*cy]
    ])

def angles_from_C(C):
    """Расчёт углов ориентации из марицы С"""
    # В.В. Матвеев, Основы построения БИНС, стр 129
    yaw   = np.arctan2(-C[2, 0], C[0, 0])  
    pitch = np.arcsin(C[1, 0])               
    roll = np.arctan2(-C[1, 2], C[1, 1])  
    return yaw, pitch, roll 

'''
def rk4(M, w):
    """
    Метод Рунге-Кутты 4-го порядка, численный метод решения диф. уравнений
    Находит M из M' = M * w
    """
    omega = skew(w)
    # Вычисляем приращения
    dt = 1
    k1 = M @ omega * dt
    k2 = (M + 0.5 * k1) @ omega * dt
    k3 = (M + 0.5 * k2) @ omega * dt
    k4 = (M + k3) @ omega * dt
    M_new = M + (k1 + 2*k2 + 2*k3 + k4) / 6
    return M_new 
'''

def rodrigues_rotation(theta_vec):
    """
    Матрица поворота через формулу Родрига
    Считает экспоненту от кососимметрической матрицы [w]
    """
    theta = np.linalg.norm(theta_vec)
    if theta < 1e-12:
        return np.eye(3)
    k = theta_vec / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def cu_from_lat_lon(lat, lon):
    """
    Матрица перехода из географической NUE в инерциальную.
    стр. 134
    """
    phi = lat
    lam = lon
    cp, sp = np.cos(phi), np.sin(phi)
    cl, sl = np.cos(lam), np.sin(lam)

    return np.array([
        [-sp*cl,  cp*cl, -sl],
        [-sp*sl,  cp*sl,  cl],
        [ cp,     sp,     0.0]
    ])

def lat_lon_from_cu(Cu, t, t0=0.0, U=7.292115e-5):
    """
    Извлечение широты и географической долготы из матрицы Cu.
    """
    # Широта (стр 134)
    phi = np.arctan2(Cu[2, 1], Cu[2, 0])

    # Инерциальная долгота (стр 135)
    sin_lam_s = -Cu[0, 2]      
    cos_lam_s =  Cu[1, 2]      
    lam_s_half = np.arctan2(sin_lam_s, 1 + cos_lam_s)  
    lam_s = 2 * lam_s_half 

    # Приведение lam_s к [0, 2pi)
    if lam_s < 0:
        lam_s += 2 * np.pi

    # Географическая долгота (стр 134)
    lam = lam_s - U * (t - t0)
    #lam = lam % (2 * np.pi)
    if lam > 2 * np.pi - 1e-8: # если почти 2pi, обнулить
        lam = 0.0
    if abs(lam) < 1e-8: # Неправильно считало при близких к нулю значениях без этой проверки
        lam = 0.0 
    else:
        lam = lam % (2 * np.pi)

    return phi, lam


def geocentric_radius(lat):
    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    numerator = np.sqrt((R0**2 * coslat)**2 + (Rp**2 * sinlat)**2)
    denominator = np.sqrt((R0 * coslat)**2 + (Rp * sinlat)**2)
    R3 = numerator / denominator
    return R3

def compute_w_ig(lat, h, Vn, Ve):
    # Проекции абсолютной угловой скорости географического трёъгранника на его оси
    # стр. 120, (3.6)
    R3 = geocentric_radius(lat)
    R = R3 + h
    w_north = U0 * np.cos(lat) + Ve / R
    w_up =  U0 * np.sin(lat) + Ve * np.tan(lat) / R
    w_east = -Vn / R
    # возвращаем в порядке NUE
    return np.array([w_north, w_up, w_east])


def gravity(lat, h):
    return np.array([0.0, -G0, 0.0])


def ins_2poisson(df, dt, lat0, lon0, h0, test = -1):
    """
    Интегрирование БИНС с двумя уравнениями Пуассона.
    """
    N = len(df)

    # Начальные значения
    Cu = cu_from_lat_lon(lat0, lon0)
    lat, lon, h = lat0, lon0, h0
    Vn, Ve, Vu = 0.0, 0.0, 0.0
    t = df.iloc[0]['TimeImu']

    # Массивы для сохранения
    D_list = [None] * N
    Cu_list = [None] * N
    C_list = [None] * N
    lat_arr = np.zeros(N)
    lon_arr = np.zeros(N)
    h_arr = np.zeros(N)
    Vn_arr = np.zeros(N)
    Ve_arr = np.zeros(N)
    Vu_arr = np.zeros(N)

    # Начальная ориентация
    n_init = 30
    Ax0 = df['Ax'].iloc[:n_init].mean()
    Ay0 = df['Ay'].iloc[:n_init].mean()
    Az0 = df['Az'].iloc[:n_init].mean()
    roll0, pitch0 = angles_from_accelerometer(Ax0, Ay0, Az0)
    #wx = df['Wx_cal'].iloc[:n_init].mean()
    #yaw0 = initial_yaw(df['Wx_cal'].iloc[0], pitch0, lat0)
    yaw0 = 0

    if test == 3:
        #pitch0 = 0
        Ve = 10
    if test == 4:
        Vn = 10

    D = cu_from_lat_lon(lat0, lon0) @ rotation_matrix_yzx(yaw0, pitch0, roll0) # D = Cu * C, стр 134

    # Вывод для проверки начальных расчетов
    print(Ax0, Ay0, Az0)
    print(df['Wx_cal'].iloc[0], df['Wy_cal'].iloc[0], df['Wz_cal'].iloc[0])
    print(f"roll0  = {roll0:.4f}")
    print(f"pitch0 = {pitch0:.4f}")
    print(f"yaw0   = {yaw0:.4f}")
    print("Матрица D:")
    print(D)
    C0_check = D @ cu_from_lat_lon(lat0, lon0).T
    print("C0 из D и Cu:\n", C0_check)
    print("Ожидаемая C0\n", rotation_matrix_yzx(yaw0,pitch0,roll0))

    # Основной цикл
    for i in range(N):
        lat_old = lat
        lon_old = lon
        h_old = h
        Vn_old = Vn
        Ve_old = Ve
        Vu_old = Vu
        D_old = D.copy()
        Cu_old = Cu.copy()
    
        # Чтение текущих измерений 
        wx = df['Wx_cal'].iloc[i]
        wy = df['Wy_cal'].iloc[i]
        wz = df['Wz_cal'].iloc[i]
        n_b = np.array([df['Ax'].iloc[i], df['Ay'].iloc[i], df['Az'].iloc[i]]) # Кажущееся ускорение, измерения акселерометров
    
        # Вычисление матрицы C для текущего момента времени
        # стр 135
        C = D_old @ Cu_old.T
        n_g = C @ n_b # Проекция вектора кажущегося ускорения на оси географической системы
        g_g = gravity(lat_old, h_old)

        R3 = geocentric_radius(lat)
        R = R3 + h

        # Проекции кориолисова ускорения на оси NUE
        # стр 121, (3.9)
        coriolis = np.array([2*U0 * Ve_old * np.sin(lat_old), 
                                 -2 * U0 * Ve_old * np.cos(lat_old), 
                                 2* (Vu_old * U0 * np.cos(lat_old) - U0 * Vn_old * np.sin(lat_old))]
        )
        
        # Проекции относительного ускорения (без производных скоростей) на оси NUE
        # стра 121, (3.11)
        a_r = np.array([((Ve_old**2) * np.tan(lat_old))/R + (Vn_old*Vu_old)/R,
                        (-Ve_old**2)/R - (Vn_old**2)/R,
                        Ve_old*Vu_old/R - Vn_old * Ve_old * np.tan(lat_old) / R
                        ]
        )

        # Компенсирующие составляющие ускорения
        # стр 122 (3.15)
        a_k = a_r + coriolis
        a_k[1] += G0

        # Ускорение в географической системе
        a_g = n_g - a_k

        if i < 5:  # вывод для первых 5 шагов
            print(f"\nШаг {i}")
            print(f"wx_cal = {wx:.3e}, wy_cal = {wy:.3e}, wz_cal = {wz:.3e}")
            print(f"n_b = [{n_b[0]:.3f}, {n_b[1]:.3f}, {n_b[2]:.3f}]")
            print(f"n_g = [{n_g[0]:.3f}, {n_g[1]:.3f}, {n_g[2]:.3f}]")
            print(f"g_g = [{g_g[0]:.3f}, {g_g[1]:.3f}, {g_g[2]:.3f}]")
            print(f"a_g = [{a_g[0]:.3e}, {a_g[1]:.3e}, {a_g[2]:.3e}]")
            print(f"V = [{Vn:.3e}, {Ve:.3e}, {Vu:.3e}]")
    
        # Интегрирование скорости 
        Vn = Vn_old + a_g[0] * dt
        Ve = Ve_old + a_g[2] * dt
        Vu = Vu_old + a_g[1] * dt
    
        # Интегрирование высоты 
        h = h_old + Vu * dt
    
        # Обновление матрицы D
        theta = np.array([wx, wy, wz]) * dt
        D = D_old @ rodrigues_rotation(theta)
        #D = rk4(D_old, np.array([wx, wy, wz]))
        """
        if i < 5:
            print(theta, "\n", rodrigues_rotation(theta), '\n\n', D, '\n')
        """
        # Вычисление абсолютной угловой скорости географической системы для обновления Cu 
        # стр 134, (3.61)
        w_ig = compute_w_ig(lat_old, h_old, Vn_old, Ve_old)
        theta_g = w_ig * dt
        Cu = Cu_old @ rodrigues_rotation(theta_g)
        #Cu = rk4(Cu_old, w_ig)
    
        # Обновление времени 
        t += dt
    
        # Извлечение новых широты и долготы из обновлённой Cu 
        lat, lon = lat_lon_from_cu(Cu, t, t0=0.0, U=U0)
    
        # Сохранение результатов 
        D_list[i] = D.copy()
        Cu_list[i] = Cu.copy()
        C_list[i] = D @ Cu.T
        lat_arr[i] = lat
        lon_arr[i] = lon
        h_arr[i] = h
        Vn_arr[i] = Vn
        Ve_arr[i] = Ve
        Vu_arr[i] = Vu
    
    return D_list, Cu_list, C_list, lat_arr, lon_arr, h_arr, Vn_arr, Ve_arr, Vu_arr


'''
def solve_poisson_equation_d(omega_x, omega_y, omega_z, roll0, pitch0, yaw0=0.0):
    """
    Решение первого уравнения Пуассона для матрицы D
    """
    N = len(omega_x)
    D_list = [None] * N

    # Начальная матрица D0 (в t=0 инерциальная = географическая)
    D0 = rotation_matrix_yzx(yaw0, pitch0, roll0)
    D_list[0] = D0
    D_cur = D0.copy()

    # В.В. Матвеев, основы построения БИНС, стр 134
    for i in range(1, N):
        omega = np.array([omega_x[i], omega_y[i], omega_z[i]])
        #theta_vec = omega * dt
        R_delta = rodrigues_rotation(omega)
        D_cur = D_cur @ R_delta
        D_list[i] = D_cur.copy()

    return D_list
'''


'''
# проверка cu_from_lat_lon и lat_lon_from_cu
# должны давать те же координаты
lat0 = 0.9      
lon0 = 1.253  
Cu = cu_from_lat_lon(lat0, lon0)
lat_r, lon_r = lat_lon_from_cu(Cu, t=0.0)
print(f"Исходная широта:  {lat0:.6f}")
print(f"Восстановленная:  {lat_r:.6f}")
print(f"Исходная долгота: {lon0:.6f}")
print(f"Восстановленная:  {lon_r:.6f}")
'''