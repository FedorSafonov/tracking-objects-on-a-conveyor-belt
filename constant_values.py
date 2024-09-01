class ConstantValues:
    # FPS для записи видео
    FPS = 24

    # индекс изображений, на которых проверяем модель
    START = 1
    STOP = 400

    # параметры модели трекинга

    IMGSZ = 640

    # папки с размеченными изображениями
    PATH_IMG = 'src/data/v2/images/'
    PATH_TARGET = 'src/data/gt/gt.txt'

    # папка с коротким видео
    PATH_VIDEO = 'src/data/test_video - Trim.mp4'
    PATH_VIDEO_BLUR = 'src/data/video_blur.mp4'

    # папка с моделью
    PATH_MODEL = 'src/yolov10x_v2_4_best.pt'

    # папка для выгрузки результатов работы
    PATH_RESULTS = 'src/data/Results/'