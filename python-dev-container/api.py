import uuid
import os
import shutil
import threading
import time
from pathlib import Path

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

@app.post("/analyze-with-file-id", response_model=AnalysisResult)
async def analyze_with_file_id(
    file_id: str = Form(...),
    date_column: str = Form("Дата"),
    endogenous_vars: Optional[List[str]] = Form(None)
):
    """
    Анализ временных рядов из ранее загруженного файла
    
    - **file_id**: Идентификатор файла, полученный при загрузке
    - **date_column**: Имя столбца с датами (по умолчанию "Дата")
    - **endogenous_vars**: Список имен эндогенных переменных для анализа коинтеграции
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
