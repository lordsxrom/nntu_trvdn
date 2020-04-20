# coding=utf-8
import cmath
import math

import numpy as np
import pandas as pd

# Линейное напряжение, В


Zcl = 1.29 + 0.405j
Zohl = 0.792 + 1.452j
Ztrvdn = 0.01 + 0.05j
Zg = 0.00001 + 0.00001j

# Длительно допустимые токи, А
Iddkl = 285
Iddvl = 450

Un = 6000 / math.sqrt(3)
Unom_trvdn = 6300

# Сопротивление нагрузки
Z_list = []

for Zn in range(5, 9):
    for Znj in range(2, 5):
        Z1 = complex(Zn, Znj)
        Z_list.extend([Z1])

# Напряжение на источнике
Ec_list = []

for Ecmod in range(5040, 7561, 100):
    for Ecyg in range(-4, 5):
        Ec = (Ecmod / math.sqrt(3)) * cmath.exp((cmath.pi / 180) * Ecyg * cmath.sqrt(-1))
        Ec_list.extend([Ec])

Ec_result_list = []
Zn_result_list = []
I_result_list = []
I1_list = []
I2_list = []
I3_list = []
Un_list = []

# Решение СЛАУ

for i in range(0, len(Z_list)):
    for ii in range(0, len(Ec_list)):
        Z_1 = np.array([[1, 1, -1], [Zcl, -(Zohl + Ztrvdn), 0], [Zcl, 0, Z_list[i] + Zg]])
        E_1 = np.array([0, 0, Ec_list[ii]])
        result = np.linalg.solve(Z_1, E_1)
        Ec_result_list.extend([Ec_list[ii]])
        Zn_result_list.extend([Z_list[i]])
        I_result_list.extend([result.tolist()])

# Разбиение на показательную форму токов (1 - КЛ, 2 - ВЛ, 3 - нагрузка)
for i in range(0, len(I_result_list)):
    I1_list.append(I_result_list[i][0])
    I2_list.append(I_result_list[i][1])
    I3_list.append(I_result_list[i][2])

I1_list_real = []
I1_list_imag = []
I2_list_real = []
I2_list_imag = []
I1_list_mod = []
I2_list_mod = []

# Длинна вектора тока
for i in range(0, len(I_result_list)):
    I1_list_real.append(I1_list[i].real)
    I1_list_imag.append(I1_list[i].imag)
    I2_list_real.append(I2_list[i].real)
    I2_list_imag.append(I2_list[i].imag)
    I1_list_mod.append(abs(I1_list[i]))
    I2_list_mod.append(abs(I2_list[i]))

# Модуль нагрузки не используется

Zn_list_real = []
Zn_list_imag = []

for i in range(0, len(Zn_result_list)):
    Zn_list_real.append(Zn_result_list[i].real)
    Zn_list_imag.append(Zn_result_list[i].imag)

KPD_list = []

for i in range(0, len(Zn_result_list)):
    KPD = (abs(I3_list[i]) * abs(I3_list[i]) * Zn_result_list[i].real) / ((
            abs(I3_list[i]) * abs(I3_list[i]) * (Zn_result_list[i].real + Zg.real) + abs(I2_list[i]) * abs(
        I2_list[i]) * Zohl.real + abs(I1_list[i]) * abs(I1_list[i]) * Zcl.real))
    KPD_list.append(KPD)

Un_real = []
Un_imag = []
Un_mod = []

for i in range(0, len(Zn_result_list)):
    U_n = Zn_result_list[i] * I3_list[i]
    Un_list.append(U_n)
    Un_real.append(Un_list[i].real)
    Un_imag.append(Un_list[i].imag)
    Un_mod.append(abs(Un_list[i]))

# Проверка длинны списков, сгенерированных программой
number = len(Zn_result_list)
print("Number of generated results", number)

# Создание файла Excel
index = list(range(0, number))
cols = ['I1', 'I2', 'I3', 'Un_mod', 'Ec_result', 'Zn_result', 'KPD']

df1 = pd.DataFrame(columns=cols, index=index)
df1['I1'] = I1_list
df1['I2'] = I2_list
df1['I3'] = I3_list
df1['Un_list'] = Un_list
df1['Ec_result'] = Ec_result_list
df1['Zn_result'] = Zn_result_list
df1['KPD'] = KPD_list

df1.to_excel('generating_step.xls', sheet_name='my_sheet')

# Вторая часть - фильтрую по напряжению, перезаписывая листы
# Листы красной зоны

Ec_Red_list = []
Zn_Red_list = []
I1_Red_list = []
I2_Red_list = []
I3_Red_list = []
Un_Red_list = []
KPD_Red_list = []

# Листы желтой зоны

Ec_Yellow_list = []
Zn_Yellow_list = []
I1_Yellow_list = []
I2_Yellow_list = []
I3_Yellow_list = []
Un_Yellow_list = []
KPD_Yellow_list = []

# Листы зеленой зоны

Ec_Green_list = []
Zn_Green_list = []
I1_Green_list = []
I2_Green_list = []
I3_Green_list = []
Un_Green_list = []
KPD_Green_list = []

# Сравнение по току 1 - КЛ, 2 - ВЛ (ввести коэфециенты для напряжения - минимальный и максимальный)


for i in range(0, len(Un_list)):
    # Фильтрация зеленой зоны
    if Un * 0.95 <= Un_mod[i] <= Un * 1.05 and I1_list_mod[i] <= 0.8 * Iddkl and I2_list_mod[i] <= 0.8 * Iddvl:
        Ec_Green_list.append(Ec_result_list[i])
        Zn_Green_list.append(Zn_result_list[i])
        I1_Green_list.append(I1_list[i])
        I2_Green_list.append(I2_list[i])
        I3_Green_list.append(I3_list[i])
        Un_Green_list.append(Un_list[i])
        KPD_Green_list.append(KPD_list[i])
    # Фильтрация красной зоны
    elif Un * 1.1 < Un_mod[i] or Un_mod[i] < Un * 0.9 or I1_list_mod[i] > 0.9 * Iddkl or I2_list_mod[i] > 0.9 * Iddvl:
        Ec_Red_list.append(Ec_result_list[i])
        Zn_Red_list.append(Zn_result_list[i])
        I1_Red_list.append(I1_list[i])
        I2_Red_list.append(I2_list[i])
        I3_Red_list.append(I3_list[i])
        Un_Red_list.append(Un_list[i])
        KPD_Red_list.append(KPD_list[i])
    # Фильтрация желтой зоны
    elif Un * 0.9 <= Un_mod[i] <= Un * 1.1 or I1_list_mod[i] <= 0.9 * Iddkl or I2_list_mod[
        i] <= 0.9 * Iddvl:
        Ec_Yellow_list.append(Ec_result_list[i])
        Zn_Yellow_list.append(Zn_result_list[i])
        I1_Yellow_list.append(I1_list[i])
        I2_Yellow_list.append(I2_list[i])
        I3_Yellow_list.append(I3_list[i])
        Un_Yellow_list.append(Un_list[i])
        KPD_Yellow_list.append(KPD_list[i])

# Проверка колличества позиций в красном и зеленом списке
nuber_red = len(I1_Red_list)
nuber_yellow = len(I1_Yellow_list)
nuber_green = len(I1_Green_list)

print("Number of results in red zone", nuber_red)
print("Number of results in yellow zone", nuber_yellow)
print("Number of results in green zone", nuber_green)

Ec_DOTRVDN_list = []
Zn_DOTRVDN_list = []
I1_DOTRVDN_list = []
I2_DOTRVDN_list = []
I3_DOTRVDN_list = []
Un_DOTRVDN_list = []
KPD_DOTRVDN_list = []

# Создание списка режимов расспределенных по зеленой, желтой и красной зоны

for i in range(0, nuber_green):
    Ec_DOTRVDN_list.append(Ec_Green_list[i])
    Zn_DOTRVDN_list.append(Zn_Green_list[i])
    I1_DOTRVDN_list.append(I1_Green_list[i])
    I2_DOTRVDN_list.append(I2_Green_list[i])
    I3_DOTRVDN_list.append(I3_Green_list[i])
    Un_DOTRVDN_list.append(Un_Green_list[i])
    KPD_DOTRVDN_list.append(KPD_Green_list[i])

for i in range(0, nuber_yellow):
    Ec_DOTRVDN_list.append(Ec_Yellow_list[i])
    Zn_DOTRVDN_list.append(Zn_Yellow_list[i])
    I1_DOTRVDN_list.append(I1_Yellow_list[i])
    I2_DOTRVDN_list.append(I2_Yellow_list[i])
    I3_DOTRVDN_list.append(I3_Yellow_list[i])
    Un_DOTRVDN_list.append(Un_Yellow_list[i])
    KPD_DOTRVDN_list.append(KPD_Yellow_list[i])

for i in range(0, nuber_red):
    Ec_DOTRVDN_list.append(Ec_Red_list[i])
    Zn_DOTRVDN_list.append(Zn_Red_list[i])
    I1_DOTRVDN_list.append(I1_Red_list[i])
    I2_DOTRVDN_list.append(I2_Red_list[i])
    I3_DOTRVDN_list.append(I3_Red_list[i])
    Un_DOTRVDN_list.append(Un_Red_list[i])
    KPD_DOTRVDN_list.append(KPD_Red_list[i])

V_list = []
V_list_result = []

Vprod = []
Vpop = []

for i in range(0, 1000, 10):
    V = i * Unom_trvdn / 10000
    Vprod.append(V)

for i in range(0, 855, 10):
    V = i * Unom_trvdn / 10000
    Vpop.append(V)

for i in range(0, len(Vpop)):
    for ii in range(0, len(Vprod)):
        V = complex(Vprod[ii], Vpop[i])
        V_list.extend([V])

# ТРВДН разбить на 2 части Добавить напряжение
t = 0
percent = 0
counter = 1

# Рассчет для зеленой зоны
Ec_GV_list = []
Zn_GV_list = []
V_GV_list = []
I_GV_list = []
Zn_GDS = []
Ec_GDS = []
I1_GV_list = []
I2_GV_list = []
I3_GV_list = []
KPD_GDS = []
VtrvdnG = []

for i in range(0, len(Ec_Green_list)):
    I1 = -1
    I2 = -1
    I3 = -1
    Ec = -1
    Zn = -1
    Vopt = -1
    KPDopt = -1
    Unag = -1
    for ii in range(0, len(V_list)):
        Z_2 = np.array([[1, 1, -1], [Zcl, -(Zohl + Ztrvdn), 0], [Zcl, 0, Zn_Green_list[i] + Zg]])
        E_2 = np.array([0, -V_list[ii], Ec_Green_list[i]])
        result_V = np.linalg.solve(Z_2, E_2)
        Ec_GV_list.extend([Ec_Green_list[i]])
        Zn_GV_list.extend([Zn_Green_list[i]])
        V_GV_list.extend([V_list[ii]])
        I_GV_list.extend([result_V.tolist()])
        KPD = (abs(I_GV_list[t + ii][2]) * abs(I_GV_list[t + ii][2]) * Zn_GV_list[i].real) / (
                abs(I_GV_list[t + ii][2]) * abs(I_GV_list[t + ii][2]) * (Zn_GV_list[i].real + Zg.real) + abs(
            I_GV_list[t + ii][1]) * abs(I_GV_list[t + ii][1]) * Zohl.real + abs(I_GV_list[t + ii][0]) * abs(
            I_GV_list[t + ii][0]) * Zcl.real)
        Unag = abs((I_GV_list[t + ii][2]) * (Zn_GV_list[i]))
        if abs(I_GV_list[t + ii][0]) <= 0.8 * Iddkl and abs(
                I_GV_list[t + ii][1]) <= 0.8 * Iddvl and KPD > KPDopt and Un * 0.95 <= Unag <= Un * 1.05:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_GV_list[t + ii][0]
            I2 = I_GV_list[t + ii][1]
            I3 = I_GV_list[t + ii][2]
            Zn = Zn_GV_list[t + ii]
            Ec = Ec_GV_list[t + ii]
    Zn_GDS.extend([Zn])
    Ec_GDS.extend([Ec])
    I1_GV_list.extend([I1])
    I2_GV_list.extend([I2])
    I3_GV_list.extend([I3])
    KPD_GDS.extend([KPDopt])
    VtrvdnG.extend([Vopt])
    t = t + len(V_list)
    percent = counter * 100 / len(Ec_Green_list)
    counter = counter + 1
    if percent % 10 == 0:
        print("Optimisation progress for green zone is: ", percent)

# Рассчет для желтой зоны
t = 0
percent = 0
counter = 1

Ec_YV_list = []
Zn_YV_list = []
V_YV_list = []
I_YV_list = []
Zn_YDS = []
Ec_YDS = []
I1_YV_list = []
I2_YV_list = []
I3_YV_list = []
KPD_YDS = []
VtrvdnY = []

for i in range(0, len(Ec_Yellow_list)):
    I1 = -1
    I2 = -1
    I3 = -1
    Ec = -1
    Zn = -1
    Vopt = -1
    KPDopt = -1
    Unag = -1
    for ii in range(0, len(V_list)):
        Z_3 = np.array([[1, 1, -1], [Zcl, -(Zohl + Ztrvdn), 0], [Zcl, 0, Zn_Yellow_list[i] + Zg]])
        E_3 = np.array([0, -V_list[ii], Ec_Yellow_list[i]])
        result_V = np.linalg.solve(Z_3, E_3)
        Ec_YV_list.extend([Ec_Yellow_list[i]])
        Zn_YV_list.extend([Zn_Yellow_list[i]])
        V_YV_list.extend([V_list[ii]])
        I_YV_list.extend([result_V.tolist()])
        KPD = (abs(I_YV_list[t + ii][2]) * abs(I_YV_list[t + ii][2]) * Zn_YV_list[i].real) / (
                abs(I_YV_list[t + ii][2]) * abs(I_YV_list[t + ii][2]) * (Zn_YV_list[i].real + Zg.real) + abs(
            I_YV_list[t + ii][1]) * abs(I_YV_list[t + ii][1]) * Zohl.real + abs(I_YV_list[t + ii][0]) * abs(
            I_YV_list[t + ii][0]) * Zcl.real)
        Unag = abs((I_YV_list[t + ii][2]) * (Zn_YV_list[i]))
        if abs(I_YV_list[t + ii][0]) <= 0.8 * Iddkl and abs(
                I_YV_list[t + ii][1]) <= 0.8 * Iddvl and KPD > KPDopt and Un * 0.95 <= Unag <= Un * 1.05:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_YV_list[t + ii][0]
            I2 = I_YV_list[t + ii][1]
            I3 = I_YV_list[t + ii][2]
            Zn = Zn_YV_list[t + ii]
            Ec = Ec_YV_list[t + ii]
        elif abs(I_YV_list[t + ii][0]) <= 0.9 * Iddkl and abs(
                I_YV_list[t + ii][1]) <= 0.9 * Iddvl and KPD > KPDopt:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_YV_list[t + ii][0]
            I2 = I_YV_list[t + ii][1]
            I3 = I_YV_list[t + ii][2]
            Zn = Zn_YV_list[t + ii]
            Ec = Ec_YV_list[t + ii]
    Zn_YDS.extend([Zn])
    Ec_YDS.extend([Ec])
    I1_YV_list.extend([I1])
    I2_YV_list.extend([I2])
    I3_YV_list.extend([I3])
    KPD_YDS.extend([KPDopt])
    VtrvdnY.extend([Vopt])
    t = t + len(V_list)
    percent = counter * 100 / len(Ec_Yellow_list)
    counter = counter + 1
    if percent % 10 == 0:
        print("Optimisation progress for yellow zone is: ", percent)

# Рассчет для красной зоны

t = 0
counter = 1
percent = 0

Ec_RV_list = []
Zn_RV_list = []
V_RV_list = []
I_RV_list = []
Zn_RDS = []
Ec_RDS = []
I1_RV_list = []
I2_RV_list = []
I3_RV_list = []
KPD_RDS = []
VtrvdnR = []

for i in range(0, len(Ec_Red_list)):
    I1 = -1
    I2 = -1
    I3 = -1
    Ec = -1
    Zn = -1
    Vopt = -1
    KPDopt = -1
    Unag = -1
    for ii in range(0, len(V_list)):
        Z_4 = np.array([[1, 1, -1], [Zcl, -(Zohl + Ztrvdn), 0], [Zcl, 0, Zn_Red_list[i] + Zg]])
        E_4 = np.array([0, -V_list[ii], Ec_Red_list[i]])
        result_V = np.linalg.solve(Z_4, E_4)
        Ec_RV_list.extend([Ec_Red_list[i]])
        Zn_RV_list.extend([Zn_Red_list[i]])
        V_RV_list.extend([V_list[ii]])
        I_RV_list.extend([result_V.tolist()])
        KPD = (abs(I_RV_list[t + ii][2]) * abs(I_RV_list[t + ii][2]) * Zn_RV_list[i].real) / (
                abs(I_RV_list[t + ii][2]) * abs(I_RV_list[t + ii][2]) * (Zn_RV_list[i].real + Zg.real) + abs(
            I_RV_list[t + ii][1]) * abs(I_RV_list[t + ii][1]) * Zohl.real + abs(I_RV_list[t + ii][0]) * abs(
            I_RV_list[t + ii][0]) * Zcl.real)
        Unag = abs((I_RV_list[t + ii][2]) * (Zn_RV_list[i]))
        if abs(I_RV_list[t + ii][0]) <= 0.8 * Iddkl and abs(
                I_RV_list[t + ii][1]) <= 0.8 * Iddvl and KPD > KPDopt and Un * 0.95 <= Unag <= Un * 1.05:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_RV_list[t + ii][0]
            I2 = I_RV_list[t + ii][1]
            I3 = I_RV_list[t + ii][2]
            Zn = Zn_RV_list[t + ii]
            Ec = Ec_RV_list[t + ii]
        elif abs(I_RV_list[t + ii][0]) <= 0.9 * Iddkl and abs(
                I_RV_list[t + ii][1]) <= 0.9 * Iddvl and KPD > KPDopt and Un * 0.9 <= Unag <= Un * 1.1:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_RV_list[t + ii][0]
            I2 = I_RV_list[t + ii][1]
            I3 = I_RV_list[t + ii][2]
            Zn = Zn_RV_list[t + ii]
            Ec = Ec_RV_list[t + ii]
        elif abs(I_RV_list[t + ii][0]) <= 0.95 * Iddkl and abs(
                I_RV_list[t + ii][1]) <= 0.95 * Iddvl and KPD > KPDopt:
            KPDopt = KPD
            Vopt = V_list[ii]
            I1 = I_RV_list[t + ii][0]
            I2 = I_RV_list[t + ii][1]
            I3 = I_RV_list[t + ii][2]
            Zn = Zn_RV_list[t + ii]
            Ec = Ec_RV_list[t + ii]
    Zn_RDS.extend([Zn])
    Ec_RDS.extend([Ec])
    I1_RV_list.extend([I1])
    I2_RV_list.extend([I2])
    I3_RV_list.extend([I3])
    KPD_RDS.extend([KPDopt])
    VtrvdnR.extend([Vopt])
    t = t + len(V_list)
    percent = counter * 100 / len(Ec_Red_list)
    counter = counter + 1
    if percent % 10 == 0:
        print("Optimisation progress for red zone is: ", percent)

Ec_TRVDN_list = []
Zn_TRVDN_list = []
I1_TRVDN_list = []
I2_TRVDN_list = []
I3_TRVDN_list = []
Un_TRVDN_list = []
KPD_TRVDN_list = []
V_TRVDN_list = []

for i in range(0, nuber_green):
    Ec_TRVDN_list.append(Ec_GDS[i])
    Zn_TRVDN_list.append(Zn_GDS[i])
    I1_TRVDN_list.append(I1_GV_list[i])
    I2_TRVDN_list.append(I2_GV_list[i])
    I3_TRVDN_list.append(I3_GV_list[i])
    Un_TRVDN_list.append(I3_GV_list[i] * Zn_TRVDN_list[i])
    KPD_TRVDN_list.append(KPD_GDS[i])
    V_TRVDN_list.append(VtrvdnG[i])

for i in range(0, nuber_yellow):
    Ec_TRVDN_list.append(Ec_YDS[i])
    Zn_TRVDN_list.append(Zn_YDS[i])
    I1_TRVDN_list.append(I1_YV_list[i])
    I2_TRVDN_list.append(I2_YV_list[i])
    I3_TRVDN_list.append(I3_YV_list[i])
    Un_TRVDN_list.append(I3_YV_list[i] * Zn_TRVDN_list[i])
    KPD_TRVDN_list.append(KPD_YDS[i])
    V_TRVDN_list.append(VtrvdnY[i])

for i in range(0, nuber_red):
    Ec_TRVDN_list.append(Ec_RDS[i])
    Zn_TRVDN_list.append(Zn_RDS[i])
    I1_TRVDN_list.append(I1_RV_list[i])
    I2_TRVDN_list.append(I2_RV_list[i])
    I3_TRVDN_list.append(I3_RV_list[i])
    Un_TRVDN_list.append(I3_RV_list[i] * Zn_TRVDN_list[i])
    KPD_TRVDN_list.append(KPD_RDS[i])
    V_TRVDN_list.append(VtrvdnR[i])

I1_act_TRVDN = []
I1_react_TRVDN = []
I2_act_TRVDN = []
I2_react_TRVDN = []
Zn_act_TRVDN = []
Zn_react_TRVDN = []
Pn_TRVDN = []
Qn_TRVDN = []
Ec_act_TRVDN = []
Ec_react_TRVDN = []
Un_act_TRVDN = []
Un_react_TRVDN = []
V_act_TRVDN = []
V_react_TRVDN = []
KPD_TRVDN = []
Ec_mod_TRVDN = []
Ec_yg_TRVDN = []

for i in range(0, len(V_TRVDN_list)):
    V_act_TRVDN.append(V_TRVDN_list[i].real)
    V_react_TRVDN.append(V_TRVDN_list[i].imag)
    KPD_TRVDN.append(KPD_TRVDN_list[i])
    Un_act_TRVDN.append((Zn_TRVDN_list[i] * I3_TRVDN_list[i]).real)
    Un_react_TRVDN.append((Zn_TRVDN_list[i] * I3_TRVDN_list[i]).imag)
    Zn_act_TRVDN.append(Zn_TRVDN_list[i].real)
    Zn_react_TRVDN.append(Zn_TRVDN_list[i].imag)
    Ec_act_TRVDN.append(Ec_TRVDN_list[i].real)
    Ec_react_TRVDN.append(Ec_TRVDN_list[i].imag)
    I1_act_TRVDN.append(I1_TRVDN_list[i].real)
    I1_react_TRVDN.append(I1_TRVDN_list[i].imag)
    I2_act_TRVDN.append(I2_TRVDN_list[i].real)
    I2_react_TRVDN.append(I2_TRVDN_list[i].imag)
    Pn_TRVDN.append(abs(I3_TRVDN_list[i]) * abs(I3_TRVDN_list[i]) * Zn_act_TRVDN[i])
    Qn_TRVDN.append(abs(I3_TRVDN_list[i]) * abs(I3_TRVDN_list[i]) * Zn_react_TRVDN[i])
    Ec_yg_TRVDN.append((math.atan(Ec_TRVDN_list[i].imag / Ec_TRVDN_list[i].real) * 180 / math.pi))
    Ec_mod_TRVDN.append(abs(Ec_TRVDN_list[i]))

V_act_doTRVDN = []
V_react_doTRVDN = []
KPD_doTRVDN = []
Un_act_doTRVDN = []
Un_react_doTRVDN = []
Zn_act_doTRVDN = []
Zn_react_doTRVDN = []
Ec_act_doTRVDN = []
Ec_react_doTRVDN = []
I1_act_doTRVDN = []
I1_react_doTRVDN = []
I2_act_doTRVDN = []
I2_react_doTRVDN = []
Pn_doTRVDN = []
Qn_doTRVDN = []
Ec_mod_doTRVDN = []
Ec_yg_doTRVDN = []

for i in range(0, len(V_TRVDN_list)):
    V_act_doTRVDN.append(V_TRVDN_list[i].real)
    V_react_doTRVDN.append(V_TRVDN_list[i].imag)
    KPD_doTRVDN.append(KPD_DOTRVDN_list[i])
    Un_act_doTRVDN.append((Zn_DOTRVDN_list[i] * I3_DOTRVDN_list[i]).real)
    Un_react_doTRVDN.append((Zn_DOTRVDN_list[i] * I3_DOTRVDN_list[i]).imag)
    Zn_act_doTRVDN.append(Zn_DOTRVDN_list[i].real)
    Zn_react_doTRVDN.append(Zn_DOTRVDN_list[i].imag)
    Ec_act_doTRVDN.append(Ec_DOTRVDN_list[i].real)
    Ec_react_doTRVDN.append(Ec_DOTRVDN_list[i].imag)
    I1_act_doTRVDN.append(I1_DOTRVDN_list[i].real)
    I1_react_doTRVDN.append(I1_DOTRVDN_list[i].imag)
    I2_act_doTRVDN.append(I2_DOTRVDN_list[i].real)
    I2_react_doTRVDN.append(I2_DOTRVDN_list[i].imag)
    Pn_doTRVDN.append(abs(I3_DOTRVDN_list[i]) * abs(I3_DOTRVDN_list[i]) * Zn_act_doTRVDN[i])
    Qn_doTRVDN.append(abs(I3_DOTRVDN_list[i]) * abs(I3_DOTRVDN_list[i]) * Zn_react_doTRVDN[i])
    Ec_mod_doTRVDN.append(abs(Ec_DOTRVDN_list[i]))
    Ec_yg_doTRVDN.append((math.atan(Ec_DOTRVDN_list[i].imag / Ec_DOTRVDN_list[i].real) * 180 / math.pi))

Un__doTRVDN = Zn_DOTRVDN_list[0] * I3_DOTRVDN_list[0]
Sn__doTRVDN = I3_DOTRVDN_list[0] * Un__doTRVDN

print(I3_DOTRVDN_list[0])
print(Un__doTRVDN)
print(Sn__doTRVDN)

number_TRVDN = len(I1_act_TRVDN)
index = list(range(0, number_TRVDN))
cols = ['Zn_act', 'Zn_react', 'Ec_act', 'Ec_react', 'Ec_mod', 'Ec_yg', 'I1_act', 'I1_react', 'I2_act', 'I2_react', 'Pn',
        'Qn',
        'Un_act_TRVDN', 'Un_react_TRVDN', 'V_act', 'V_react', 'KPD']

df2 = pd.DataFrame(columns=cols, index=index)
df2['Zn_act'] = Zn_act_TRVDN
df2['Zn_react'] = Zn_react_TRVDN
df2['Ec_act'] = Ec_act_TRVDN
df2['Ec_react'] = Ec_react_TRVDN
df2['Ec_mod'] = Ec_mod_TRVDN
df2['Ec_yg'] = Ec_yg_TRVDN
df2['I1_act'] = I1_act_TRVDN
df2['I1_react'] = I1_react_TRVDN
df2['I2_act'] = I2_act_TRVDN
df2['I2_react'] = I2_react_TRVDN
df2['Pn'] = Pn_TRVDN
df2['Qn'] = Qn_TRVDN
df2['Un_act_TRVDN'] = Un_act_TRVDN
df2['Un_react_TRVDN'] = Un_react_TRVDN
df2['V_act'] = V_act_TRVDN
df2['V_react'] = V_react_TRVDN
df2['KPD'] = KPD_TRVDN

df2.to_excel('optimisation_step.xls', sheet_name='sheet_TRVDN')

number_TRVDN = len(I1_act_doTRVDN)
index = list(range(0, number_TRVDN))
cols = ['Zn_act', 'Zn_react', 'Ec_act', 'Ec_react', 'Ec_mod', 'Ec_yg', 'I1_act', 'I1_react', 'I2_act', 'I2_react', 'Pn',
        'Qn',
        'Un_act_TRVDN', 'Un_react_TRVDN', 'V_act', 'V_react', 'KPD']

df3 = pd.DataFrame(columns=cols, index=index)
df3['Zn_act'] = Zn_act_doTRVDN
df3['Zn_react'] = Zn_react_doTRVDN
df3['Ec_act'] = Ec_act_doTRVDN
df3['Ec_react'] = Ec_react_doTRVDN
df3['Ec_mod'] = Ec_mod_doTRVDN
df3['Ec_yg'] = Ec_yg_doTRVDN
df3['I1_act'] = I1_act_doTRVDN
df3['I1_react'] = I1_react_doTRVDN
df3['I2_act'] = I2_act_doTRVDN
df3['I2_react'] = I2_react_doTRVDN
df3['Pn'] = Pn_doTRVDN
df3['Qn'] = Qn_doTRVDN
df3['Un_act_TRVDN'] = Un_act_doTRVDN
df3['Un_react_TRVDN'] = Un_react_doTRVDN
df3['V_act'] = V_act_doTRVDN
df3['V_react'] = V_react_doTRVDN
df3['KPD'] = KPD_doTRVDN

df3.to_excel('data_for_learnig.xls', sheet_name='sheet_NN')
