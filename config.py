class ConstValues:
    """
    Конфигурации для YOLOv10Tracker
    """
    # FPS для записи видео
    FPS: int = 24

    # индекс изображений, на которых проверяем модель
    START: int = 1
    STOP: int = 400

    # параметры модели трекинга
    MAX_TRACK_LEN = 30
    IMGSZ: int = 640

    # папки с размеченными изображениями
    PATH_IMG: str = '../src/data/v2/images/'
    # папки с файлом разметки
    PATH_TARGET: str = '../src/data/gt/gt.txt'

    # путь к тестовому видео
    PATH_VIDEO: str = '../src/data/test_video.mp4'
    # путь к видео плохого качества
    PATH_VIDEO_BLUR: str = '../src/data/video_blur.mp4'

    # путь к модели YOLO
    PATH_MODEL: str = '../src/yolov10x_v2_4_best.pt'

    # папка для выгрузки результатов работы
    PATH_RESULTS: str = '../src/data/Results/'

