import os
import io
import tempfile
import smtplib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import threading
from scipy import stats
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from docx import Document
from docx.shared import Inches
from pydantic import BaseModel, EmailStr
from typing import Optional
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from datetime import datetime


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
# Модель данных для формы обратной связи
class FeedbackForm(BaseModel):
    name: str
    email: EmailStr
    subject: Optional[str] = "Сообщение с сайта"
    message: str

def save_message(feedback):
    try:
        # Создаем директорию, если её нет
        os.makedirs("feedback_messages", exist_ok=True)
        
        # Формируем сообщение
        message = {
            "name": feedback.name,
            "email": feedback.email,
            "subject": feedback.subject,
            "message": feedback.message,
            "date": datetime.now().isoformat()
        }
        
        # Создаем уникальное имя файла
        filename = f"feedback_messages/msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Сохраняем в файл
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(message, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Ошибка при сохранении сообщения: {str(e)}")
        return False

def cleanup_old_messages():
    """Удаляет сообщения старше 1 месяца"""
    while True:
        try:
            # Проверяем существование директории
            if not os.path.exists("feedback_messages"):
                time.sleep(86400)  # Проверяем раз в день
                continue
                
            # Текущая дата минус 1 месяц
            one_month_ago = datetime.now() - timedelta(days=30)
            
            # Проверяем все файлы в директории
            for filename in os.listdir("feedback_messages"):
                file_path = os.path.join("feedback_messages", filename)
                
                # Проверяем, является ли файл .json
                if not filename.endswith('.json'):
                    continue
                    
                # Получаем время создания файла из имени или из метаданных
                try:
                    # Пытаемся получить дату из имени файла
                    date_str = filename.split('_')[1].split('.')[0]
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                except:
                    # Если не получилось, используем время модификации файла
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Удаляем файл, если он старше 1 месяца
                if file_date < one_month_ago:
                    os.remove(file_path)
                    print(f"Удален устаревший файл с обратной связью: {filename}")
            
            # Проверяем раз в день
            time.sleep(86400)
            
        except Exception as e:
            print(f"Ошибка при очистке старых сообщений: {str(e)}")
            time.sleep(86400)  # В случае ошибки тоже ждем день

# Запускаем поток для очистки старых сообщений
cleanup_thread = threading.Thread(target=cleanup_old_messages, daemon=True)
cleanup_thread.start()

def transform_to_stationary(series):
    """
    Преобразование ряда к стационарному виду (I(0)) с использованием 
    последовательных разностей и тестов на стационарность
    """
    # Проверяем исходный ряд на стационарность
    is_stationary, _ = check_stationarity(series, "")
    
    if is_stationary == "Стационарен":
        # Если ряд уже стационарен, возвращаем его без изменений
        return series
    
    # Пробуем первые разности
    diff1 = series.diff().dropna()
    is_stationary, _ = check_stationarity(diff1, "")
    
    if is_stationary == "Стационарен":
        # Пополняем пропущенные значения (первое наблюдение)
        padded_series = pd.Series([None] + diff1.tolist(), index=series.index)
        return padded_series
    
    # Пробуем вторые разности
    diff2 = diff1.diff().dropna()
    is_stationary, _ = check_stationarity(diff2, "")
    
    if is_stationary == "Стационарен":
        # Пополняем пропущенные значения (первые два наблюдения)
        padded_series = pd.Series([None, None] + diff2.tolist(), index=series.index)
        return padded_series
    
    # Если не удалось достичь стационарности, возвращаем вторые разности
    padded_series = pd.Series([None, None] + diff2.tolist(), index=series.index)
    return padded_series

def transform_to_first_order(series):
    """
    Преобразование ряда к первому порядку интеграции (I(1))
    """
    # Проверяем исходный ряд
    is_stationary, _ = check_stationarity(series, "")
    
    if is_stationary == "Стационарен":
        # Если ряд стационарен (I(0)), интегрируем его один раз
        integrated = series.cumsum()
        return integrated
    
    # Проверяем первые разности
    diff1 = series.diff().dropna()
    is_stationary, _ = check_stationarity(diff1, "")
    
    if is_stationary == "Стационарен":
        # Если первые разности стационарны (I(1)), оставляем исходный ряд
        return series
    
    # Проверяем вторые разности
    diff2 = diff1.diff().dropna()
    is_stationary, _ = check_stationarity(diff2, "")
    
    if is_stationary == "Стационарен":
        # Если вторые разности стационарны (I(2)), интегрируем один раз
        # Начинаем с первой разности и интегрируем
        padded_diff1 = pd.Series([0] + diff1.tolist(), index=series.index)
        integrated = padded_diff1.cumsum()
        return integrated
    
    # Если не определено, возвращаем исходный ряд
    return series

def transform_to_second_order(series):
    """
    Преобразование ряда ко второму порядку интеграции (I(2))
    """
    # Проверяем исходный ряд
    is_stationary, _ = check_stationarity(series, "")
    
    if is_stationary == "Стационарен":
        # Если ряд стационарен (I(0)), интегрируем его дважды
        integrated1 = series.cumsum()
        integrated2 = integrated1.cumsum()
        return integrated2
    
    # Проверяем первые разности
    diff1 = series.diff().dropna()
    is_stationary, _ = check_stationarity(diff1, "")
    
    if is_stationary == "Стационарен":
        # Если первые разности стационарны (I(1)), интегрируем один раз
        integrated = series.cumsum()
        return integrated
    
    # Проверяем вторые разности
    diff2 = diff1.diff().dropna()
    is_stationary, _ = check_stationarity(diff2, "")
    
    if is_stationary == "Стационарен":
        # Если вторые разности стационарны (I(2)), оставляем исходный ряд
        return series
    
    # Если не определено, возвращаем исходный ряд
    return series

def format_time_series_for_preview(dates, values):
    """
    Форматирует временной ряд для предпросмотра в формате для JSON
    
    Returns:
    --------
    list: Список словарей {date: str, value: float}
    """
    result = []
    for date, value in zip(dates, values):
        if pd.notna(value):  # Исключаем NaN значения
            result.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                "value": float(value)
            })
    
    return result
def build_varx_model(df, endogenous_vars, exogenous_vars=None, lags=1, train_size=0.8):
    """
    Обучение VARX модели на данных и оценка её качества.
    Если указана одна эндогенная переменная, строится ARX модель.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм с временными рядами
    endogenous_vars : list
        Список имен эндогенных переменных для VARX/ARX модели
    exogenous_vars : list, optional
        Список имен экзогенных переменных для VARX/ARX модели
    lags : int, default=1
        Количество лагов для VARX/ARX модели
    train_size : float, default=0.8
        Доля данных для обучения модели (0-1)
        
    Returns:
    --------
    dict
        Результаты обучения и оценки модели
    """
    import statsmodels.api as sm
    import pickle
    import numpy as np
    import io
    import base64
    from matplotlib import pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    # Убираем пропуски из данных
    df_clean = df.dropna()
    
    # Подготовка данных
    endog = df_clean[endogenous_vars]
    exog = df_clean[exogenous_vars] if exogenous_vars else None
    
    # Разделение на обучающую и тестовую выборки
    train_size_idx = int(len(df_clean) * train_size)
    train_endog = endog.iloc[:train_size_idx]
    train_exog = exog.iloc[:train_size_idx] if exog is not None else None
    test_endog = endog.iloc[train_size_idx:]
    test_exog = exog.iloc[train_size_idx:] if exog is not None else None
    
    results = {
        "success": True,
        "model_info": {},
        "forecasts": {},
        "diagnostics": {},
        "plots": {}
    }
    
    try:
        # Определение оптимального числа лагов с помощью ACF и PACF
        acf_pacf_plots = []
        optimal_lags = {}
        
        for var in endogenous_vars:
            series = train_endog[var]
            
            # ACF и PACF анализ
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # ACF
            plot_acf(series, lags=20, ax=axes[0], title=f'ACF для {var}')
            
            # PACF
            plot_pacf(series, lags=20, ax=axes[1], title=f'PACF для {var}')
            
            plt.tight_layout()
            
            # Сохраняем график в base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            acf_pacf_plots.append(img_str)
            plt.close()
            
            # Определяем оптимальное число лагов по PACF
            # Значения выше 95% доверительного интервала считаются значимыми
            from statsmodels.tsa.stattools import pacf
            pacf_values, conf_intervals = pacf(series, nlags=20, alpha=0.05, method='ywm')
            
            # Находим последний значимый лаг
            significant_lags = []
            for i in range(1, len(pacf_values)):
                if abs(pacf_values[i]) > abs(conf_intervals[i, 1]):  # Верхняя граница 95% интервала
                    significant_lags.append(i)
            
            # Если нет значимых лагов или только первый лаг значим, берем 1
            # В противном случае берем последний значимый лаг (до максимума 10)
            if not significant_lags:
                optimal_lag = 1
            else:
                optimal_lag = min(max(significant_lags), 10)
            
            optimal_lags[var] = optimal_lag
        
        # Определяем итоговое количество лагов (максимальное из оптимальных)
        optimal_lag_final = max(optimal_lags.values()) if optimal_lags else lags
        
        # Используем переданное значение лагов, но сохраняем оптимальное для информации
        model_lags = lags
        
        # Обрезаем данные в соответствии с выбранным числом лагов
        train_endog_adj = train_endog.iloc[model_lags:]
        train_exog_adj = train_exog.iloc[model_lags:] if train_exog is not None else None
        
        # Обучаем модель
        if len(endogenous_vars) == 1:
            # ARX модель для одной эндогенной переменной
            endog_var = endogenous_vars[0]
            model = sm.tsa.SARIMAX(
                train_endog_adj[endog_var], 
                exog=train_exog_adj,
                order=(model_lags, 0, 0),  # (p, d, q) - используем только AR часть
                trend='c'
            )
            fitted_model = model.fit(disp=False)
        else:
            # VARX модель для нескольких эндогенных переменных
            model = sm.tsa.VARMAX(train_endog_adj, order=(model_lags, 0), trend='c', exog=train_exog_adj)
            fitted_model = model.fit(maxiter=1000, disp=False)
        
        # Сохраняем информацию о модели
        model_type = "ARX" if len(endogenous_vars) == 1 else "VARX"
        results["model_info"] = {
            "model_type": model_type,
            "aic": fitted_model.aic,
            "bic": fitted_model.bic,
            "hqic": fitted_model.hqic if hasattr(fitted_model, 'hqic') else None,
            "parameters": fitted_model.params.to_dict() if hasattr(fitted_model.params, 'to_dict') else dict(zip(fitted_model.params.index.tolist(), fitted_model.params.values.tolist())),
            "lags": model_lags,
            "optimal_lags": optimal_lags,
            "optimal_lag_final": optimal_lag_final,
            "endogenous_vars": endogenous_vars,
            "exogenous_vars": exogenous_vars or [],
            "train_size": train_size_idx,
            "test_size": len(test_endog)
        }
        
        # Прогнозы на тестовом наборе, если он не пустой
        if len(test_endog) > 0:
            try:
                # Создаем прогнозы
                if len(endogenous_vars) == 1:
                    # Для ARX модели
                    endog_var = endogenous_vars[0]
                    forecast = fitted_model.forecast(steps=len(test_endog), exog=test_exog)
                    # Преобразуем в DataFrame для совместимости с VARX моделью
                    forecast_df = pd.DataFrame({endog_var: forecast}, index=test_endog.index)
                else:
                    # Для VARX модели
                    forecast_df = fitted_model.forecast(steps=len(test_endog), exog=test_exog)
                
                # Оценка качества прогнозов
                mse = {}
                mae = {}
                rmse = {}
                mape = {}
                
                for i, var in enumerate(endogenous_vars):
                    actual = test_endog[var].values
                    if len(endogenous_vars) == 1:
                        predicted = forecast_df[var].values
                    else:
                        predicted = forecast_df.iloc[:, i].values
                    
                    # Убираем NaN перед расчетом метрик
                    valid_indices = ~(np.isnan(actual) | np.isnan(predicted))
                    
                    if np.any(valid_indices):
                        actual_valid = actual[valid_indices]
                        predicted_valid = predicted[valid_indices]
                        
                        var_mse = np.mean((actual_valid - predicted_valid) ** 2)
                        var_mae = np.mean(np.abs(actual_valid - predicted_valid))
                        var_rmse = np.sqrt(var_mse)
                        
                        # MAPE может вызвать деление на ноль, поэтому используем try/except
                        try:
                            var_mape = np.mean(np.abs((actual_valid - predicted_valid) / actual_valid)) * 100
                        except:
                            var_mape = np.nan
                        
                        mse[var] = var_mse
                        mae[var] = var_mae
                        rmse[var] = var_rmse
                        mape[var] = var_mape
                
                # Создаем DataFrame для сравнения
                comparison = {}
                for i, var in enumerate(endogenous_vars):
                    if len(endogenous_vars) == 1:
                        pred_values = forecast_df[var].tolist()
                    else:
                        pred_values = forecast_df.iloc[:, i].tolist()
                    
                    comparison[var] = {
                        "actual": test_endog[var].tolist(),
                        "predicted": pred_values,
                        "dates": test_endog.index.strftime("%Y-%m-%d").tolist()
                    }
                
                results["forecasts"] = {
                    "comparison": comparison,
                    "metrics": {
                        "mse": mse,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape
                    }
                }
                
                # Создаем графики прогнозов
                forecast_plots = []
                for i, var in enumerate(endogenous_vars):
                    plt.figure(figsize=(10, 6))
                    plt.plot(test_endog.index, test_endog[var], 'b-', label='Фактические значения')
                    
                    if len(endogenous_vars) == 1:
                        pred_values = forecast_df[var]
                    else:
                        pred_values = forecast_df.iloc[:, i]
                    
                    plt.plot(test_endog.index, pred_values, 'r--', label='Прогноз')
                    plt.title(f'Прогноз vs. Факт: {var}')
                    plt.legend()
                    plt.grid(True)
                    
                    # Форматирование дат на оси X
                    plt.gcf().autofmt_xdate()
                    
                    # Сохраняем график в base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    forecast_plots.append(img_str)
                    plt.close()
                
                results["plots"]["forecast_plots"] = forecast_plots
                
            except Exception as e:
                results["forecasts"] = {"error": str(e)}
        
        # Сохраняем графики ACF и PACF
        results["plots"]["acf_pacf_plots"] = acf_pacf_plots
        
        # Добавляем информацию об оптимальных лагах
        results["diagnostics"]["lag_analysis"] = {
            "optimal_lags": optimal_lags,
            "optimal_lag_final": optimal_lag_final,
            "chosen_lag": model_lags
        }
        
        # Сериализуем модель в base64 для возможного сохранения
        model_bytes = io.BytesIO()
        pickle.dump(fitted_model, model_bytes)
        model_bytes.seek(0)
        results["model_base64"] = base64.b64encode(model_bytes.read()).decode('utf-8')
        
    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results
