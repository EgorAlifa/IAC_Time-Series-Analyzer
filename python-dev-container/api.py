from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from fastapi import Query
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tempfile
import os
import json
import requests
from io import BytesIO
import uuid
import shutil
import threading
import time
from pathlib import Path

from ts_analysis import analyze_data, create_word_report, test_cointegration, johansen_test, save_message, cleanup_old_messages, FeedbackForm, transform_to_stationary, transform_to_first_order, transform_to_second_order, format_time_series_for_preview

app = FastAPI(title="Анализ стационарности временных рядов")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Frontend в режиме разработки
        "https://egoralifa.github.io",  # GitHub Pages
        "https://egoralifa.github.io/IAC_Time-Series-Analyzer",  # Полный путь репозитория
        "http://37.252.23.30:8000",  # API сервер
        "https://iac-time-series-analyzer.ru",  # Ваш новый домен с HTTPS
        "http://iac-time-series-analyzer.ru",  # Ваш новый домен с HTTP
        "null",  # Для локальных файлов
        "file://*",  # Файловый протокол
        "*"  # Максимально открытый вариант для отладки
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "Access-Control-Allow-Headers", 
        "Access-Control-Allow-Origin",
    ],
    expose_headers=["*"],  # Открываем все заголовки ответа
    max_age=3600,
)

# Создаем директорию для временного хранения файлов, если ее нет
TEMP_FILES_DIR = Path("temp_files")
TEMP_FILES_DIR.mkdir(exist_ok=True)

# Словарь для хранения времени загрузки файлов
file_timestamps = {}

# Функция для удаления просроченных файлов
def cleanup_old_files():
    while True:
        current_time = time.time()
        expired_files = []

        # Находим все файлы старше 1 часа
        for file_id, timestamp in list(file_timestamps.items()):
            if current_time - timestamp > 3600:  # 3600 секунд = 1 час
                expired_files.append(file_id)
        
        # Удаляем просроченные файлы
        for file_id in expired_files:
            file_path = TEMP_FILES_DIR / file_id
            if file_path.exists():
                os.remove(file_path)
            file_timestamps.pop(file_id, None)
            print(f"Файл {file_id} удален после истечения срока хранения")
        
        # Проверяем каждые 5 минут
        time.sleep(300)

# Запускаем поток для очистки файлов
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

class AnalysisResult(BaseModel):
    variable_results: Dict[str, Dict[str, Any]]
    cointegration_results: Optional[Dict[str, Any]] = None
def unpack_variables(variables):
    """
    Распаковывает переменные из возможных форматов (строка, список, JSON-строка)
    
    Args:
        variables: Переменные в различных форматах
        
    Returns:
        list: Список переменных
    """
    # Если None, возвращаем пустой список
    if variables is None:
        return []
    
    # Если строка, проверяем, является ли она JSON
    if isinstance(variables, str):
        try:
            # Пробуем распарсить как JSON
            import json
            parsed = json.loads(variables)
            if isinstance(parsed, list):
                return parsed
            else:
                return [variables]  # Не список JSON, возвращаем как есть
        except:
            # Не JSON, возвращаем как есть
            return [variables]
    
    # Если список, проверяем каждый элемент на JSON
    if isinstance(variables, list):
        result = []
        for item in variables:
            if isinstance(item, str):
                try:
                    # Пробуем распарсить как JSON
                    import json
                    parsed = json.loads(item)
                    if isinstance(parsed, list):
                        result.extend(parsed)  # Добавляем все элементы из JSON-массива
                    else:
                        result.append(parsed)  # Добавляем одиночное значение
                except:
                    # Не JSON, добавляем как есть
                    result.append(item)
            else:
                # Не строка, добавляем как есть
                result.append(item)
        return result
    
    # Для других типов возвращаем в списке
    return [variables]
@app.get("/api")
async def root():
    return {"message": "API для анализа стационарности временных рядов"}

@app.get("/api/news")
async def get_news_endpoint(
    q: str = Query(default="Ставка", description="Поисковый запрос"),
    limit: int = Query(default=5, description="Лимит количества новостей")
):
    try:
        print(f"Запрос новостей с параметрами: q={q}, limit={limit}")
        
        # API ключ для NewsAPI
        api_key = "1709ee5d87c94b238a395fefc8375b92"
        
        # Вычисляем дату 20 дней назад
        current_date = datetime.now()
        previous_date = current_date - timedelta(days=20)
        formatted_date = previous_date.strftime('%Y-%m-%d')
        
        # Формируем URL для запроса к NewsAPI
        url = f'https://newsapi.org/v2/everything?q={q}&from={formatted_date}&sortBy=publishedAt&apiKey={api_key}'
        print(f"URL запроса: {url}")
        
        # Выполняем запрос только один раз
        response = requests.get(url)
        print(f"Статус ответа: {response.status_code}")
        
        data = response.json()
        print(f"Получены данные: {data.get('status')}")
        
        news_list = []
        if data.get('status') == 'ok' and 'articles' in data:
            articles = data['articles']
            # Фильтруем статьи
            filtered_articles = [
                article for article in articles
                if "путин" not in article.get('title', '').lower() and "война" not in article.get('title', '').lower() and "войны" not in article.get('title', '').lower()
            ]
            
            # Ограничиваем количество новостей
            filtered_articles = filtered_articles[:limit]
            
            for article in filtered_articles:
                news_list.append({
                    'title': article.get('title', 'Без заголовка'),
                    'description': article.get('description', ''),
                    'url': article.get('url', '#'),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', 'Неизвестный источник')
                })
        else:
            # Возвращаем информацию об ошибке
            return {
                'error': 'Failed to fetch news',
                'details': data.get('message', 'Unknown error'),
                'status': data.get('status', 'error')
            }
        
        return news_list
    
    except Exception as e:
        print(f"Ошибка при получении новостей: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'error': 'Exception occurred',
            'message': str(e)
        }

@app.post("/cointegration-analysis")
async def cointegration_analysis(
    file_id: str = Form(...),
    date_column: str = Form("Дата"),
    endogenous_vars: Optional[Union[List[str], str]] = Form(None),
    det_order: int = Form(1),  # Значение по умолчанию 1
    k_ar_diff: int = Form(1)   # Значение по умолчанию 1
):
    """
    Анализ коинтеграции для выбранных переменных
    
    - **file_id**: Идентификатор файла, полученный при загрузке
    - **date_column**: Имя столбца с датами (по умолчанию "Дата")
    - **endogenous_vars**: Список имен эндогенных переменных для анализа коинтеграции
    """
    # Распаковываем переменные из возможных форматов
    unpacked_vars = unpack_variables(endogenous_vars)
    
    # Добавим отладочный вывод
    print(f"Cointegration analysis - file_id: {file_id}")
    print(f"Cointegration analysis - date_column: {date_column}")
    print(f"Cointegration analysis - endogenous_vars: {endogenous_vars}")
    print(f"Cointegration analysis - unpacked_vars: {unpacked_vars}")
    
    # Проверяем наличие хотя бы двух переменных
    if len(unpacked_vars) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Для анализа коинтеграции необходимо минимум две переменные"}
        )
    
    file_path = TEMP_FILES_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или срок его хранения истек")
    
    try:
        # Определяем тип файла по расширению
        file_extension = os.path.splitext(str(file_path))[1].lower()
        
        # Чтение данных из файла
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            # Пытаемся определить тип файла автоматически
            try:
                df = pd.read_excel(file_path)
            except:
                try:
                    df = pd.read_csv(file_path)
                except:
                    raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
        
        # Проверка наличия столбца с датами
        if date_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Столбец {date_column} не найден в файле", "columns": df.columns.tolist()}
            )
            
        # Преобразование столбца даты
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка при преобразовании столбца даты: {str(e)}"}
            )
        
        # Проверяем, что все указанные переменные есть в данных
        missing_vars = [var for var in unpacked_vars if var not in df.columns]
        if missing_vars:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Следующие переменные не найдены в данных: {', '.join(missing_vars)}",
                    "available_columns": df.columns.tolist()
                }
            )
        
        # Выполняем только коинтеграционный анализ
        
        # Для анализа методом Энгла-Грейнджера создаем все возможные пары
        eg_results = []
        
        for i in range(len(unpacked_vars)):
            for j in range(i+1, len(unpacked_vars)):
                var1 = unpacked_vars[i]
                var2 = unpacked_vars[j]
                
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
            unpacked_vars,
            det_order=det_order,  # Используем переданное значение
            k_ar_diff=k_ar_diff   # Используем переданное значение
        )
        
        # Определяем общий результат коинтеграции
        any_eg_cointegrated = any([result['is_cointegrated'] for result in eg_results])
        is_cointegrated = any_eg_cointegrated or is_cointegrated_johansen
        
        # Формируем результат
        result = {
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
        
        return result
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе коинтеграции: {str(e)}")
@app.post("/get-time-series-data")
async def get_time_series_data(
    file_id: str = Form(...),
    date_column: str = Form("Дата"),
    column: str = Form(...)
):
    """
    Получение данных временного ряда для построения графиков
    """
    file_path = TEMP_FILES_DIR / file_id

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")

    try:
        # Чтение данных из файла
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # Проверка наличия столбцов
        if date_column not in df.columns or column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Столбцы не найдены"}
            )

        # Преобразование даты и сортировка
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)

        # Формирование данных временного ряда
        time_series_data = []
        for _, row in df.iterrows():
            time_series_data.append({
                "date": row[date_column].strftime("%Y-%m-%d"),
                "value": float(row[column])
            })

        return time_series_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении данных: {str(e)}")
@app.post("/upload-temp-file")
async def upload_temp_file(file: UploadFile = File(...)):
    """
    Загрузка файла для временного хранения (1 час)
    
    - **file**: Excel или CSV файл с данными
    
    Возвращает уникальный идентификатор файла для последующего доступа
    """
    try:
        # Генерируем уникальный ID для файла
        file_id = str(uuid.uuid4())
        file_path = TEMP_FILES_DIR / file_id
        
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Запоминаем время загрузки
        file_timestamps[file_id] = time.time()
        
        # Определяем тип файла
        content_type = "unknown"
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.csv':
            content_type = "text/csv"
        elif file_extension in ['.xlsx', '.xls']:
            content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        return {
            "file_id": file_id,
            "original_filename": file.filename,
            "content_type": content_type,
            "expiration": "1 hour"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")

@app.get("/temp-file/{file_id}")
async def get_temp_file(file_id: str):
    """
    Получение временного файла по его идентификатору
    
    - **file_id**: Идентификатор файла, полученный при загрузке
    """
    file_path = TEMP_FILES_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или срок его хранения истек")
    
    return FileResponse(path=str(file_path))

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_file(
    file: UploadFile = File(...),
    date_column: str = Form("Дата"),
    endogenous_vars: Optional[List[str]] = Form(None)
):
    """
    Анализ стационарности и коинтеграции временных рядов из Excel или CSV файла
    
    - **file**: Excel или CSV файл с данными
    - **date_column**: Имя столбца с датами (по умолчанию "Дата")
    - **endogenous_vars**: Список имен эндогенных переменных для анализа коинтеграции
    """
    # Проверка расширения файла
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        # Чтение данных из файла
        content = await file.read()
        if file_extension == '.csv':
            df = pd.read_csv(BytesIO(content))
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только файлы CSV, XLS или XLSX")
            
        # Проверка наличия столбца с датами
        if date_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Столбец {date_column} не найден в файле", "columns": df.columns.tolist()}
            )
            
        # Преобразование столбца даты
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка при преобразовании столбца даты: {str(e)}"}
            )
            
        # Проведение анализа
        results = analyze_data(df, endogenous_vars)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе данных: {str(e)}")

@app.post("/analyze-with-file-id", response_model=AnalysisResult)
async def analyze_with_file_id(
    file_id: str = Form(...),
    date_column: str = Form("Дата"),
    endogenous_vars: Optional[Union[List[str], str]] = Form(None)
):
    """
    Анализ временных рядов из ранее загруженного файла
    
    - **file_id**: Идентификатор файла, полученный при загрузке
    - **date_column**: Имя столбца с датами (по умолчанию "Дата")
    - **endogenous_vars**: Список имен эндогенных переменных для анализа коинтеграции
    """
    # Распаковываем переменные из возможных форматов
    unpacked_vars = unpack_variables(endogenous_vars)
    
    # Добавим отладочный вывод
    print(f"Received file_id: {file_id}")
    print(f"Received date_column: {date_column}")
    print(f"Received endogenous_vars: {endogenous_vars}")
    print(f"Type of endogenous_vars: {type(endogenous_vars)}")
    print(f"Unpacked vars: {unpacked_vars}")
    
    file_path = TEMP_FILES_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или срок его хранения истек")
    
    try:
        # Определяем тип файла по расширению
        file_extension = os.path.splitext(str(file_path))[1].lower()
        
        # Чтение данных из файла
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            # Пытаемся определить тип файла автоматически
            try:
                df = pd.read_excel(file_path)
            except:
                try:
                    df = pd.read_csv(file_path)
                except:
                    raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
        
        # Проверка наличия столбца с датами
        if date_column not in df.columns:
            return JSONResponse(
                status_code=400,
                content={"error": f"Столбец {date_column} не найден в файле", "columns": df.columns.tolist()}
            )
            
        # Преобразование столбца даты
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Ошибка при преобразовании столбца даты: {str(e)}"}
            )
            
        # Проведение анализа с распакованными переменными
        results = analyze_data(df, unpacked_vars)
        
        # Обработка особых случаев возврата
        if isinstance(results, dict) and 'status' in results:
            if results['status'] == 'required_input':
                # Возвращаем список доступных столбцов и сообщение для пользователя
                return JSONResponse(
                    status_code=200,
                    content={
                        "required_input": True,
                        "message": results['message'],
                        "available_columns": results['available_columns']
                    }
                )
            elif results['status'] == 'error':
                # Возвращаем сообщение об ошибке
                return JSONResponse(
                    status_code=400,
                    content={"error": results['message']}
                )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе данных: {str(e)}")
@app.post("/analyze-columns")
async def analyze_columns(file: UploadFile = File(...)):
    """
    Анализ доступных столбцов в загруженном файле
    
    - **file**: Excel или CSV файл с данными
    """
    # Проверка расширения файла
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        # Чтение данных из файла
        content = await file.read()
        if file_extension == '.csv':
            df = pd.read_csv(BytesIO(content))
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только файлы CSV, XLS или XLSX")
        
        # Определение типов столбцов
        column_types = {}
        numeric_columns = []
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                column_types[column] = "numeric"
                numeric_columns.append(column)
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                column_types[column] = "datetime"
            else:
                try:
                    # Проверяем, можно ли преобразовать в дату
                    pd.to_datetime(df[column])
                    column_types[column] = "potential_datetime"
                except:
                    column_types[column] = "string"
        
        # Возвращаем результаты анализа
        return {
            "columns": df.columns.tolist(),
            "column_types": column_types,
            "numeric_columns": numeric_columns,
            "rows_count": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе столбцов: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Проверка работоспособности API
    """
    return {"status": "ok", "version": "1.0.0","Первый запуск Егор":"Оно работает я проверил"}

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackForm):
    """
    Обработка формы обратной связи (тестовая версия)
    """
    # Выводим данные в консоль для отладки
    print(f"Получено сообщение от: {feedback.name} <{feedback.email}>")
    print(f"Тема: {feedback.subject}")
    print(f"Сообщение: {feedback.message}")
    
    # Всегда возвращаем успех
    return {"status": "success", "message": "Сообщение успешно отправлено"}

# Генерация тестовых данных для демонстрации (если нет реальных данных)
@app.get("/generate-demo-data")
async def generate_demo_data():
    """
    Генерация тестовых данных для демонстрации API
    """
    try:
        # Создаем тестовые временные ряды
        dates = pd.date_range(start='2013-01-01', end='2023-12-01', freq='MS')
        
        # Симулируем нестационарный ряд (случайное блуждание)
        random_walk = np.random.normal(0, 1, len(dates))
        random_walk = np.cumsum(random_walk)
        
        # Симулируем ряд с трендом
        trend = np.linspace(5, 15, len(dates)) + np.random.normal(0, 0.5, len(dates))
        
        # Создаем датафрейм
        df = pd.DataFrame({
            'Дата': dates,
            'Средневзвешенная_ставка': trend,
            'Средняя_задолженность': random_walk + 50 + np.linspace(0, 30, len(dates)),
            'Ключевая_ставка': np.random.normal(8, 2, len(dates)),
            'Денежные_доходы': np.linspace(25, 50, len(dates)) + np.random.normal(0, 2, len(dates))
        })
        
        # Сохраняем в файл
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            temp_path = temp_file.name
            df.to_excel(temp_path, index=False)
        
        # Отправляем демо-файл пользователю
        return FileResponse(
            path=temp_path,
            filename="demo_data.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации демо-данных: {str(e)}")
        
@app.post("/transform-integration-order")
async def transform_integration_order(
    file_id: str = Form(...),
    date_column: str = Form("Дата"),
    transformation_settings: str = Form(...),
    preview: str = Form("false")
):
    """
    Преобразование временных рядов к заданному порядку интеграции
    
    - **file_id**: Идентификатор загруженного файла
    - **date_column**: Имя столбца с датами
    - **transformation_settings**: JSON со структурой {переменная: порядок_интеграции}
    - **preview**: Если "true", то возвращает только данные для предпросмотра без сохранения
    
    Возможные значения порядка интеграции:
    - "none": Оставить как есть
    - "I(0)": Привести к стационарному виду (дифференцирование до стационарности)
    - "I(1)": Привести к первому порядку интеграции
    - "I(2)": Привести к второму порядку интеграции
    """
    # Проверяем наличие файла
    file_path = TEMP_FILES_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден или срок его хранения истек")
    
    try:
        # Разбираем настройки преобразования
        settings = json.loads(transformation_settings)
        is_preview = preview.lower() == "true"
        
        # Определяем тип файла по расширению
        file_extension = os.path.splitext(str(file_path))[1].lower()
        
        # Чтение данных из файла
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            # Пытаемся определить тип файла автоматически
            try:
                df = pd.read_excel(file_path)
            except:
                try:
                    df = pd.read_csv(file_path)
                except:
                    raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
        
        # Проверка наличия столбца с датами
        if date_column not in df.columns:
            return {"error": f"Столбец {date_column} не найден в файле", "columns": df.columns.tolist()}
            
        # Преобразование столбца даты
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            # Сортировка по дате для корректности расчетов
            df = df.sort_values(by=date_column)
        except Exception as e:
            return {"error": f"Ошибка при преобразовании столбца даты: {str(e)}"}
        
        # Данные для предпросмотра трансформации
        preview_data = {"transformed_data": {}}
        
        # Создаем новый DataFrame для измененных данных
        transformed_df = df.copy()
        
        # Преобразуем каждую переменную согласно настройкам
        for variable, transform_to in settings.items():
            # Пропускаем, если оставляем как есть
            if transform_to == "none":
                continue
                
            # Проверяем, есть ли переменная в DataFrame
            if variable not in df.columns:
                continue
                
            # Получаем временной ряд
            series = df[variable].copy()
            
            # Преобразуем согласно выбранному порядку интеграции
            if transform_to == "I(0)":  # Привести к стационарному виду
                # Применяем первые разности
                transformed_series = series.diff()
                column_name = f"{variable}_I0"
                
            elif transform_to == "I(1)":  # Привести к первому порядку интеграции
                if variable.endswith("_I0"):  # Если это уже I(0)
                    # Интегрируем один раз
                    transformed_series = series.cumsum()
                    # Замена NaN начальным значением
                    transformed_series.iloc[0] = series.iloc[0]
                    column_name = variable.replace("_I0", "_I1")
                else:
                    # Оставляем как есть или применяем первые разности, если нужно
                    transformed_series = series
                    column_name = f"{variable}_I1"
                
            elif transform_to == "I(2)":  # Привести к второму порядку интеграции
                if variable.endswith("_I0"):  # Если это I(0)
                    # Интегрируем дважды
                    temp_series = series.cumsum()
                    transformed_series = temp_series.cumsum()
                    # Замена NaN начальными значениями
                    transformed_series.iloc[0] = series.iloc[0]
                    transformed_series.iloc[1] = temp_series.iloc[1]
                    column_name = variable.replace("_I0", "_I2")
                elif variable.endswith("_I1"):  # Если это I(1)
                    # Интегрируем один раз
                    transformed_series = series.cumsum()
                    # Замена NaN начальным значением
                    transformed_series.iloc[0] = series.iloc[0]
                    column_name = variable.replace("_I1", "_I2")
                else:
                    # Оставляем как есть
                    transformed_series = series
                    column_name = f"{variable}_I2"
            
            # Добавляем преобразованный ряд в DataFrame
            transformed_df[column_name] = transformed_series
            
            # Добавляем данные для предпросмотра
            if is_preview:
                preview_data["transformed_data"][variable] = [
                    {"date": date.strftime("%Y-%m-%d"), "value": float(value) if pd.notna(value) else None}
                    for date, value in zip(transformed_df[date_column], transformed_series)
                    if pd.notna(value)
                ]
        
        # Если это предпросмотр, возвращаем только данные для визуализации
        if is_preview:
            return preview_data
        
        # Сохраняем преобразованные данные в новый файл с правильным расширением
        new_file_id = str(uuid.uuid4())
        new_file_path = TEMP_FILES_DIR / f"{new_file_id}{file_extension}"

        
        # Сохраняем файл в том же формате, что и исходный
        if file_extension == '.csv':
            transformed_df.to_csv(new_file_path, index=False)
        else:
            transformed_df.to_excel(new_file_path, index=False)
        
        # Запоминаем время загрузки нового файла
        file_timestamps[new_file_id] = time.time()
        
        # Анализируем столбцы нового файла
        numeric_columns = [col for col in transformed_df.columns if pd.api.types.is_numeric_dtype(transformed_df[col])]
        
        # Готовим данные для обновления localStorage
        updated_columns_data = {
            "columns": transformed_df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "rows_count": len(transformed_df),
            "date_column": date_column,
        }
        
        # Формируем URL для скачивания с сохранением расширения
        download_url = f"http://37.252.23.30:8000/temp-file/{new_file_id}{file_extension}"
        
        # Возвращаем результаты
        return {
            "success": True,
            "message": "Преобразование успешно выполнено",
            "updated_file_id": new_file_id,
            "updated_columns_data": updated_columns_data,
            "download_url": download_url
        }
            
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при преобразовании данных: {str(e)}")
# Установка URL API для формирования URL скачивания
API_URL = "http://37.252.23.30:8000" 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
