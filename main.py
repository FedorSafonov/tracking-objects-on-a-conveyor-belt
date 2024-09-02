import glob
import json
import os
import numpy as np
from config import ConstValues
from init_model import YOLOv10Tracker


# Инициализируем модель YOLO v10
test_model = YOLOv10Tracker(model_path=ConstValues.PATH_MODEL)

# Создаем видеопоток с тестированием трекинга объектов
# На вход подаем видеозапись, по умолчанию длина треков 30 кадров
test_model.test_track(video_path=ConstValues.PATH_VIDEO)

# Пример расчета метрики для видео

# Указываем кол-во кадров и суффикс видео
num_fames = 99
name_suffix = 'short'

# Создаем срез для видео
gt_video_short = gt[gt[:, 0] <= num_fames]

# Попробуем загрузить результаты, если они уже были посчитаны, если нет, то создадим их.
try:
    with open(ConstValues.PATH_RESULTS + f"pred_track_{name_suffix}.json", "r") as json_file:
        t_video_short = np.array(json.load(json_file))
    with open(ConstValues.PATH_RESULTS + f"delta_time_mean_{name_suffix}.json", "r") as json_file:
        time_video_short = json.load(json_file)
    print("Данные успешно загружены")

except:
    print("Выполняется расчет")
    t_video_short, time_per_frame_video_short = test_model.tracking_video(
        ConstValues.PATH_VIDEO,
        gt_video_short,
        ConstValues.PATH_RESULTS,
        name_suffix,
        output_video=f"output_video_{name_suffix}.mp4",
    )

# Подсчет скорости обработки одного кадра
# print(f'Время обработки одного кадра: **{time_per_frame_video_short}**')

# Рассчет метрики для видео

strsummary_video_short, summary_hist_video_short, acc_video_short = YOLOv10Tracker.mot_metrics(gt_video_short, t_video_short, stop=num_fames)

print(strsummary_video_short)

YOLOv10Tracker.analyze_metrics(summary_hist_video_short)