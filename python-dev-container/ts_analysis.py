import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
from docx import Document
from docx.shared import Inches
import io
import tempfile

# Функция для проведения теста Дики-Фуллера на стационарность
def adf_test(series, title=''):
    """
    Проведение расширенного теста Дики-Фуллера (ADF) для проверки стационарности

    H0: Ряд имеет единичный корень (не стационарен)
    H1: Ряд не имеет единичный корень (стационарен)

    Если p-value < 0.05, отвергаем нулевую гипотезу и считаем ряд стационарным
    """
    result = adfuller(series.dropna())

    output = []
    output.append(f'ADF Test для {title}:')
    output.append(f'ADF Statistic: {result[0]:.4f}')
    output.append(f'p-value: {result[1]:.4f}')

    # Критические значения для разных уровней значимости
    for key, value in result[4].items():
        output.append(f'Критическое значение ({key}): {value:.4f}')

    # Вывод результата
    if result[1] < 0.05:
        output.append(f"Результат: Отвергаем нулевую гипотезу. Ряд {title} стационарен.\n")
        is_stationary = True
    else:
        output.append(f"Результат: Не отвергаем нулевую гипотезу. Ряд {title} не стационарен.\n")
        is_stationary = False

    return is_stationary, output

# Функция для проведения теста KPSS на стационарность
def kpss_test(series, title=''):
    """
    Проведение теста Квятковски-Филлипса-Шмидта-Шина (KPSS) для проверки стационарности

    H0: Ряд стационарен
    H1: Ряд не стационарен

    Если p-value < 0.05, отвергаем нулевую гипотезу и считаем ряд нестационарным
    """
    result = kpss(series.dropna(), regression='c', nlags="auto")

    output = []
    output.append(f'KPSS Test для {title}:')
    output.append(f'KPSS Statistic: {result[0]:.4f}')
    output.append(f'p-value: {result[1]:.4f}')

    # Критические значения для разных уровней значимости
    for key, value in result[3].items():
        output.append(f'Критическое значение ({key}): {value:.4f}')

    # Вывод результата
    if result[1] < 0.05:
        output.append(f"Результат: Отвергаем нулевую гипотезу. Ряд {title} не стационарен.\n")
        is_stationary = False
    else:
        output.append(f"Результат: Не отвергаем нулевую гипотезу. Ряд {title} стационарен.\n")
        is_stationary = True

    return is_stationary, output

# Дополнительны тест на стационарность Phillips-Perron
def phillips_perron_test(series, title=''):
    """
    Проведение теста Филлипса-Перрона для проверки стационарности
    используя библиотеку arch

    H0: Ряд имеет единичный корень (не стационарен)
    H1: Ряд не имеет единичный корень (стационарен)

    Если p-value < 0.05, отвергаем нулевую гипотезу и считаем ряд стационарным
    """
    # Удаляем пропуски
    series = series.dropna()

    # Базовый ADF тест
    adf_result = adfuller(series)

    # Вычисляем разности первого порядка
    diff_series = series.diff().dropna()

    # Реализация PP теста на основе ADF
    T = len(diff_series)
    demeaned_diff = diff_series - diff_series.mean()

    # Оценка дисперсии остатков
    sigma_sq = np.sum(demeaned_diff**2) / (T - 1)

    # Оценка долгосрочной дисперсии (упрощенная)
    long_run_variance = sigma_sq

    # Статистика PP
    pp_stat = (adf_result[0] - T/2 * (sigma_sq - long_run_variance)) / np.sqrt(2 * long_run_variance / T)

    # Критические значения для PP теста (приближенные)
    critical_values = {
        '1%': -3.43,
        '5%': -2.86,
        '10%': -2.57
    }

    # Вычисление p-value (приближенное)
    p_value = stats.t.sf(abs(pp_stat), T-1)*2

    # Вывод результатов
    output = []
    output.append(f'Phillips-Perron Test для {title}:')
    output.append(f'PP Statistic: {pp_stat:.4f}')
    output.append(f'p-value: {p_value:.4f}')

    # Критические значения
    for key, value in critical_values.items():
        output.append(f'Критическое значение ({key}): {value:.4f}')

    # Интерпретация результата
    if p_value < 0.05:
        output.append(f"Результат: Отвергаем нулевую гипотезу. Ряд {title} стационарен.\n")
        is_stationary = True
    else:
        output.append(f"Результат: Не отвергаем нулевую гипотезу. Ряд {title} не стационарен.\n")
        is_stationary = False

    return is_stationary, output

# Функция для проверки стационарности с помощью трех тестов
def check_stationarity(series, title=''):
    """
    Комплексная проверка стационарности с помощью трех тестов:
    - ADF тест
    - KPSS тест
    - Phillips-Perron тест

    Ряд признается стационарным, если минимум 2 из 3 тестов показывают стационарность
    """
    output = []

    # Проведение тестов
    adf_result, adf_output = adf_test(series, title)
    kpss_result, kpss_output = kpss_test(series, title)
    pp_result, pp_output = phillips_perron_test(series, title)

    # Собираем результаты всех тестов
    output.extend(adf_output)
    output.extend(kpss_output)
    output.extend(pp_output)

    # Интерпретация результатов тестов
    test_details = [
        ("ADF", adf_result),
        ("KPSS", kpss_result),
        ("Phillips-Perron", pp_result)
    ]

    # Подсчет количества тестов, указывающих на стационарность
    stationary_count = sum(test[1] for test in test_details)
    total_tests = len(test_details)

    # Детальный вывод результатов каждого теста
    detailed_results = "\nДетальные результаты тестов:\n"
    for test_name, is_stationary in test_details:
        detailed_results += f"{test_name} тест: {'Стационарен' if is_stationary else 'Не стационарен'}\n"

    # Определение статуса на основе критериев:
    # 1. Минимум 2 теста показывают стационарность
    if stationary_count >= 2:
        conclusion = f"Итоговый результат: Ряд {title} стационарен (по {stationary_count} из {total_tests} тестов)."
        status = "Стационарен"
    else:
        conclusion = f"Итоговый результат: Ряд {title} не стационарен (стационарен только по {stationary_count} из {total_tests} тестов)."
        status = "Не стационарен"

    # Добавляем детальные результаты к выводу
    output.append(conclusion)
    output.append(detailed_results)

    return status, output

# Функция для проверки коинтеграции
def test_cointegration(series1, series2, name1, name2):
    """
    Проверка наличия коинтеграции между двумя рядами

    H0: Нет коинтеграции
    H1: Есть коинтеграция

    Если p-value < 0.05, отвергаем H0 и считаем, что ряды коинтегрированы
    """
    # Объединяем ряды без пропусков
    df_temp = pd.DataFrame({name1: series1, name2: series2})
    df_temp = df_temp.dropna()

    # Проводим тест Йохансена на коинтеграцию (через ADF)
    result = coint(df_temp[name1], df_temp[name2])

    output = []
    output.append(f'Тест на коинтеграцию между {name1} и {name2}:')
    output.append(f'Test Statistic: {result[0]:.4f}')
    output.append(f'p-value: {result[1]:.4f}')

    # Критические значения
    for i, value in enumerate(result[2]):
        output.append(f'Критическое значение ({i+1}%): {value:.4f}')

    # Интерпретация результата
    if result[1] < 0.05:
        output.append(f"Результат: Отвергаем нулевую гипотезу. Ряды {name1} и {name2} коинтегрированы.\n")
        is_cointegrated = True
    else:
        output.append(f"Результат: Не отвергаем нулевую гипотезу. Ряды {name1} и {name2} НЕ коинтегрированы.\n")
        is_cointegrated = False

    return is_cointegrated, output

# Функция для теста Йохансена на коинтеграцию
def johansen_test(df, endogenous_vars, det_order=1, k_ar_diff=1):
    """
    Проведение теста Йохансена на коинтеграцию для нескольких временных рядов

    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с временными рядами
    endogenous_vars : list
        Список имен переменных для анализа коинтеграции
    det_order : int
        Порядок детерминистических компонент
    k_ar_diff : int
        Количество лагов в VAR в разностях
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Выделяем нужные переменные и удаляем пропуски
    data = df[endogenous_vars].dropna()

    # Проводим тест Йохансена
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # Получаем статистику и критические значения
    trace_stat = result.lr1
    eigen_stat = result.lr2

    # Критические значения для trace test (90%, 95%, 99%)
    crit_vals_trace = result.cvt

    # Критические значения для eigenvalue test (90%, 95%, 99%)
    crit_vals_eigen = result.cvm

    # Количество переменных
    n_vars = len(endogenous_vars)

    output = []
    output.append(f'Тест Йохансена на коинтеграцию для переменных: {", ".join(endogenous_vars)}')
    output.append(f'Количество переменных: {n_vars}')
    output.append(f'Порядок детерминистических компонент: {det_order}')
    output.append(f'Количество лагов: {k_ar_diff}\n')

    # Вывод результатов trace test
    output.append('Результаты Trace Test:')
    output.append('H0: Ранг коинтеграции <= r | H1: Ранг коинтеграции > r')
    output.append('-' * 70)
    output.append(f"{'r':^5}|{'Статистика':^15}|{'90%':^10}|{'95%':^10}|{'99%':^10}|{'Результат':^15}")
    output.append('-' * 70)

    trace_results = []
    for i in range(n_vars):
        reject_null = trace_stat[i] > crit_vals_trace[i, 1]  # Используем 95% критическое значение
        trace_results.append(reject_null)

        # Форматируем вывод
        result_text = "Отклоняем H0" if reject_null else "Не отклоняем H0"
        output.append(f"{i:^5}|{trace_stat[i]:^15.4f}|{crit_vals_trace[i, 0]:^10.4f}|{crit_vals_trace[i, 1]:^10.4f}|{crit_vals_trace[i, 2]:^10.4f}|{result_text:^15}")

    # Определяем ранг коинтеграции на основе trace test
    cointegration_rank = 0
    for i, reject in enumerate(trace_results):
        if not reject:
            cointegration_rank = i
            break

    # Делаем заключение
    output.append('\nЗаключение:')
    if cointegration_rank == 0:
        conclusion = "Нет коинтеграционных соотношений между переменными. Все переменные независимы в долгосрочном периоде."
        is_cointegrated = False
    elif cointegration_rank == 1:
        conclusion = f"Обнаружено {cointegration_rank} коинтеграционное соотношение. Существует один стабильный долгосрочный вектор между переменными."
        is_cointegrated = True
    elif cointegration_rank == 2:
        conclusion = f"Обнаружено {cointegration_rank} коинтеграционных соотношения. Существует два стабильных долгосрочных вектора между переменными."
        is_cointegrated = True
    else:
        conclusion = f"Обнаружено {cointegration_rank} коинтеграционных соотношений. Сложная долгосрочная динамика взаимосвязи между переменными."
        is_cointegrated = True
    
    output.append(conclusion)
    
    # Возвращаем результат в конце
    return is_cointegrated, output

# Функция для анализа разностей (дифференцирования) ряда
def analyze_differences(series, title='', max_diff=4):
    """
    Анализ разностей временного ряда для достижения стационарности
    Поддерживает до 4 порядков дифференцирования
    """
    all_outputs = []

    # Проверка исходного ряда
    orig_result, orig_output = check_stationarity(series, f"{title} (исходный)")
    all_outputs.extend(orig_output)

    # Если ряд уже стационарен, нет смысла дифференцировать
    if orig_result == "Стационарен":
        return 0, all_outputs

    # Дифференцирование первого порядка
    diff1 = series.diff().dropna()
    diff1_result, diff1_output = check_stationarity(diff1, f"{title} (первые разности)")
    all_outputs.extend(diff1_output)

    if diff1_result == "Стационарен":
        return 1, all_outputs

    # Дифференцирование второго порядка, если необходимо
    if max_diff >= 2:
        diff2 = diff1.diff().dropna()
        diff2_result, diff2_output = check_stationarity(diff2, f"{title} (вторые разности)")
        all_outputs.extend(diff2_output)

        if diff2_result == "Стационарен":
            return 2, all_outputs

    # Дифференцирование третьего порядка, если необходимо
    if max_diff >= 3:
        diff3 = diff2.diff().dropna()
        diff3_result, diff3_output = check_stationarity(diff3, f"{title} (третьи разности)")
        all_outputs.extend(diff3_output)

        if diff3_result == "Стационарен":
            return 3, all_outputs

    # Дифференцирование четвертого порядка, если необходимо
    if max_diff >= 4:
        diff4 = diff3.diff().dropna()
        diff4_result, diff4_output = check_stationarity(diff4, f"{title} (четвертые разности)")
        all_outputs.extend(diff4_output)

        if diff4_result == "Стационарен":
            return 4, all_outputs

    # Если даже после четырех дифференцирований ряд не стационарен
    return "Не достигнуто", all_outputs

# Основная функция для анализа данных
def analyze_data(df, endogenous_vars=None):
   """
   Основная функция для анализа временных рядов
   
   Parameters:
   -----------
   df : pandas.DataFrame
       Датафрейм с временными рядами
   endogenous_vars : list or None
       Список имен эндогенных переменных для анализа
       
   Returns:
   --------
   dict
       Результаты анализа в формате JSON
   """
   # Словарь для хранения результатов
   results = {'variable_results': {}}
   
   # Если эндогенные переменные не указаны, используем все числовые колонки
   if endogenous_vars is None:
       # Получаем все числовые колонки
       all_numeric_columns = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
       
       # Если не указаны эндогенные, берем первые две как эндогенные
       endogenous_vars = all_numeric_columns[:2]
   else:
       # Получаем все числовые колонки, исключая эндогенные
       all_numeric_columns = [col for col in df.columns 
                              if df[col].dtype in [np.float64, np.int64] 
                              and col not in endogenous_vars]
   
   # Проверяем, что эндогенные переменные существуют в датафрейме
   endogenous_vars = [var for var in endogenous_vars if var in df.columns]
   
   # Анализ эндогенных переменных
   for column in endogenous_vars:
       d_value, test_outputs = analyze_differences(df[column], column, max_diff=4)
       results['variable_results'][column] = {
           'd_value': d_value,
           'test_outputs': test_outputs,
           'type': 'endogenous'
       }
   
   # Анализ экзогенных переменных
   for column in all_numeric_columns:
       d_value, test_outputs = analyze_differences(df[column], column, max_diff=4)
       results['variable_results'][column] = {
           'd_value': d_value,
           'test_outputs': test_outputs,
           'type': 'exogenous'
       }
   
   # Анализ коинтеграции
   if len(endogenous_vars) >= 2:
       # Для анализа методом Энгла-Грейнджера создаем все возможные пары
       eg_results = []
       
       for i in range(len(endogenous_vars)):
           for j in range(i+1, len(endogenous_vars)):
               var1 = endogenous_vars[i]
               var2 = endogenous_vars[j]
               
               is_cointegrated_eg, cointegration_output_eg = test_cointegration(
                   df[var1],
                   df[var2],
                   var1,
                   var2
               )
               
               eg_results.append({
                   'variables': [var1, var2],
                   'is_cointegrated': is_cointegrated_eg,
                   'details': cointegration_output_eg
               })
       
       # Тест Йохансена на коинтеграцию (для всех переменных сразу)
       is_cointegrated_johansen, cointegration_output_johansen = johansen_test(
           df,
           endogenous_vars,
           det_order=1,
           k_ar_diff=1
       )
       
       # Определяем общий результат коинтеграции
       any_eg_cointegrated = any([result['is_cointegrated'] for result in eg_results])
       is_cointegrated = any_eg_cointegrated or is_cointegrated_johansen
       
       # Сохраняем результаты обоих тестов
       results['cointegration_results'] = {
           'engle_granger_pairs': eg_results,
           'engle_granger_summary': {
               'is_cointegrated': any_eg_cointegrated,
               'cointegrated_pairs': [f"{result['variables'][0]} и {result['variables'][1]}" 
                                    for result in eg_results if result['is_cointegrated']]
           },
           'johansen': {
               'is_cointegrated': is_cointegrated_johansen,
               'details': cointegration_output_johansen
           },
           'conclusion': "Переменные коинтегрированы." if is_cointegrated else "Переменные не коинтегрированы.",
           'recommended_model': "VECM" if is_cointegrated else "VAR в разностях"
       }
   
   return results
# Функция создания отчета Word
def create_word_report(results, df):
    """
    Создает отчет в формате Word с результатами анализа стационарности и коинтеграции
    """
    # Создание нового документа Word
    doc = Document()
    
    # Добавление заголовка
    doc.add_heading('Отчет о стационарности временных рядов', 0)
    
    # Добавление текущей даты
    current_date = datetime.now().strftime("%d.%m.%Y")
    doc.add_paragraph(f'Дата создания отчета: {current_date}')
    
    # Сводная таблица результатов для всех переменных
    doc.add_heading('Сводная таблица анализа стационарности', 1)
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    
    # Заголовки таблицы
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Переменная'
    hdr_cells[1].text = 'Порядок интеграции'
    hdr_cells[2].text = 'Статус'
    
    # Заполнение таблицы для всех переменных
    for var, result in results['variable_results'].items():
        row_cells = table.add_row().cells
        row_cells[0].text = var
        row_cells[1].text = f"I({result['d_value']})"
        
        # Определение статуса
        d = result['d_value']
        if d == 0:
            status = "Стационарный"
        elif d == 1:
            status = "Интегрированный первого порядка"
        elif d == 2:
            status = "Интегрированный второго порядка"
        else:
            status = "Требует дополнительного анализа"
            
        row_cells[2].text = status
    
    # Если есть результаты коинтеграции
    if 'cointegration_results' in results:
        doc.add_heading('Анализ коинтеграции между переменными', 1)
        doc.add_paragraph(results['cointegration_results']['conclusion'])
        
        # Рекомендуемая модель
        doc.add_heading('Рекомендуемая модель', 1)
        doc.add_paragraph(results['cointegration_results']['recommended_model'])
    
    # Временный файл для отчета
    with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as temp_file:
        temp_path = temp_file.name
        doc.save(temp_path)
    
    return temp_path
