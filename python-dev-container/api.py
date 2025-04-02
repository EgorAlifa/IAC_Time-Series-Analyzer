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
from ts_analysis import generate_time_series_plots_for_variable

from ts_analysis import analyze_data, create_word_report

app = FastAPI(title="Анализ стационарности временных рядов")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/")
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
                if "путин" not in article.get('title', '').lower()
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
@app.get("/generate-charts")
async def generate_charts(
    file_id: str = Query(..., description="ID загруженного файла"),
    column: str = Query(..., description="Название колонки"),
    date_column: str = Query("Дата", description="Название столбца с датами")
):
    """
    Генерация графиков для выбранной колонки
    
    - **file_id**: Идентификатор файла
    - **column**: Название колонки для анализа
    - **date_column**: Название столбца с датами (по умолчанию "Дата")
    """
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
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
        
        # Проверка наличия столбца с датами
        if date_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Столбец {date_column} не найден в файле")
        
        # Преобразование столбца даты
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        
        # Проверка наличия колонки
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Колонка {column} не найдена в файле")
        
        # Генерация графиков
        charts = generate_time_series_plots_for_variable(df, column)
        
        return {
            "charts": charts,
            "column": column
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации графиков: {str(e)}")

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
    # Преобразование строки в список, если передана одна переменная
    if isinstance(endogenous_vars, str):
        endogenous_vars = [endogenous_vars]
    
    # Добавим отладочный вывод
    print(f"Received file_id: {file_id}")
    print(f"Received date_column: {date_column}")
    print(f"Received endogenous_vars: {endogenous_vars}")
    print(f"Type of endogenous_vars: {type(endogenous_vars)}")
    
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
            
        # Проведение анализа
        results = analyze_data(df, endogenous_vars)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе данных: {str(e)}")

@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    date_column: str = Form("Дата"),
    endogenous_vars: Optional[List[str]] = Form(None)
):
    """
    Генерация отчета в формате Word на основе анализа стационарности и коинтеграции
    
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
        
        # Создание отчета Word
        report_path = create_word_report(results, df)
        
        # Отправка файла пользователю
        return FileResponse(
            path=report_path,
            filename="Отчет_анализа_стационарности.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации отчета: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
