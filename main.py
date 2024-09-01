import glob
import json
import os
import numpy as np
from constant_values import ConstantValues
from init_model import YOLOv10Tracker


# Инициализируем модель YOLO v10
test_model = YOLOv10Tracker(model_path='src/yolov10x_v2_4_best.pt')

# Создаем видеопоток с тестированием трекинга объектов
# На вход подаем видеозапись, по умолчанию длина треков 30 кадров
test_model.test_track(video_path='src/data/test_video - Trim.mp4')

# Пример расчета метрики для группы фотографий

# Создаем папку, если ее нет
if not os.path.exists(ConstantValues.PATH_RESULTS):
    os.makedirs(ConstantValues.PATH_RESULTS)
    print(f"Папка '{ConstantValues.PATH_RESULTS}' была создана.")
else:
    print(f"Папка '{ConstantValues.PATH_RESULTS}' уже существует.")

image_paths = sorted(
    glob.glob(f'{ConstantValues.PATH_IMG}{str(i).zfill(6)}.jpg')
    for i in range(ConstantValues.START, ConstantValues.STOP + 1)
)

image_paths = [path for sublist in image_paths for path in sublist]

# Создаем срез для фотографий
gt = np.loadtxt(ConstantValues.PATH_TARGET, delimiter=",")
gt_slice = gt[(gt[:, 0] <= ConstantValues.STOP) & (gt[:, 0] >= ConstantValues.START)]


# Попробуем загрузить результаты, если они уже были посчитаны, если нет, то создадим их.
try:
    with open(ConstantValues.PATH_RESULTS + f"pred_track_{ConstantValues.START}_{ConstantValues.STOP}.json", "r") as json_file:
        t = np.array(json.load(json_file))
    with open(ConstantValues.PATH_RESULTS + f"delta_time_mean_{ConstantValues.START}_{ConstantValues.STOP}.json", "r") as json_file:
        time_per_frame = json.load(json_file)
    print("Данные успешно загружены")

except:
    print("Выполняется расчет")
    t, time_per_frame = test_model.tracking_image(
        image_paths,
        gt_slice,
        ConstantValues.PATH_RESULTS,
        output_video=f"output_video_{ConstantValues.START}_{ConstantValues.STOP}.mp4",
        fps=ConstantValues.FPS
    )

# Рассчет метрики для фотографий

strsummary, summary_hist, acc = YOLOv10Tracker.mot_metrics(gt_slice, t)

print(strsummary)

YOLOv10Tracker.analyze_metrics(summary_hist)

# Пример расчета метрики для видео

# Указываем кол-во кадров и суффикс видео
num_fames = 99
name_suffix = 'short'

# Создаем срез для видео
gt_video_short = gt[gt[:, 0] <= num_fames]

# Попробуем загрузить результаты, если они уже были посчитаны, если нет, то создадим их.
try:
    with open(ConstantValues.PATH_RESULTS + f"pred_track_{name_suffix}.json", "r") as json_file:
        t_video_short = np.array(json.load(json_file))
    with open(ConstantValues.PATH_RESULTS + f"delta_time_mean_{name_suffix}.json", "r") as json_file:
        time_video_short = json.load(json_file)
    print("Данные успешно загружены")

except:
    print("Выполняется расчет")
    t_video_short, time_per_frame_video_short = test_model.tracking_video(
        ConstantValues.PATH_VIDEO,
        gt_video_short,
        ConstantValues.PATH_RESULTS,
        name_suffix,
        output_video=f"output_video_{name_suffix}.mp4",
    )

# Подсчет скорости обработки одного кадра
print(f'Время обработки одного кадра: **{time_per_frame_video_short}**')

# Рассчет метрики для видео

strsummary_video_short, summary_hist_video_short, acc_video_short = YOLOv10Tracker.mot_metrics(gt_video_short, t_video_short, stop=num_fames)

print(strsummary_video_short)

YOLOv10Tracker.analyze_metrics(summary_hist_video_short)