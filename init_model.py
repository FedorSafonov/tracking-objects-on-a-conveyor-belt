import json
from collections import defaultdict
from datetime import timedelta
from time import time

import numpy as np
import motmetrics as mm
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from config import ConstValues


class YOLOv10Tracker:
    def __init__(self, model_path: str = ConstValues.PATH_MODEL):
        """
        Инициализирует YOLOv10Tracker с заданными параметрами.
        :param model_path: Путь к модели YOLO или наименование стандартной модели, которая будет загружена.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"На устройстве доступны ресурсы: {'CPU' if self.device=='cpu' else 'GPU'}")
        self.model = YOLO(model_path)


    def test_track(self, video_path: str, max_track_length: int = ConstValues.MAX_TRACK_LEN,
                   display_window_name: str = "YOLOv10 Tracking"):
        """
        Запускает процесс отслеживания объектов на видео с использованием модели YOLOv10.
        Визуализирует треки объектов на каждом кадре видео.
        """

        cap = cv2.VideoCapture(video_path)

        track_history = defaultdict(lambda: [])

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            results = self.model.track(frame, persist=True)
            boxes = results[0].boxes.xywh

            if results[0].boxes.id is None:
                continue

            track_ids = results[0].boxes.id.int().tolist()
            annotated_frame = results[0].plot()

            # Цикл Обновляет историю треков для текущих объектов
            # boxes: Координаты объектов на текущем кадре
            # track_ids: Идентификаторы треков для текущих объектов
            for box, track_id in zip(boxes, track_ids):

                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))

                # Ограничиваем длину трека объекта
                if len(track) > max_track_length:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False,
                              color=(230, 230, 230), thickness=10)

            cv2.imshow(display_window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def draw_track_and_boxes(self, annotated_frame: np.ndarray,
                             track_history: dict, max_track_length: int = 30,
                             xy: list = [], wh: list = [], track_ids: list = [],
                             gt: bool = True):
        """
        Рисует историю треков объектов и bounding box (bbox) на аннотированном кадре, предоставляя визуализацию отслеживания объектов.

        Параметры:
        - annotated_frame (np.ndarray): Кадр, на который будут нанесены аннотации и треки.
        - track_history (dict): Словарь, содержащий историю треков для каждого объекта, где ключом является ID трека, а значением - список координат центра.
        - xy (list): Список координат верхнего левого угла боксов в формате [x, y].
        - wh (list): Список ширины и высоты боксов в формате [w, h].
        - track_ids (list): Список ID треков, соответствующих каждому бокс-объекту.
        - gt (bool): Флаг, указывающий, является ли это отображение истинным значением (ground truth). Если True, используется красный цвет для аннотаций.

        Функция выполняет следующие действия:
        1. Определяет цвет для аннотаций в зависимости от значения флага `gt`.
        2. Обрабатывает каждый трек, обновляя историю треков для соответствующего объекта.
        3. Рисует линию трека, если имеются хотя бы две его точки.
        4. Рисует bounding box вокруг объекта, если `gt` установлен в True.
        5. Ограничивает длину истории треков до 30 наиболее последних позиций.

        Примечание: Для корректного выполнения функции необходимо, чтобы все необходимые библиотеки (например, OpenCV и NumPy) были предварительно импортированы.
        """
        color = (0, 0, 255) if gt else (230, 230, 230)  # Устанавливаем цвет в зависимости от флага gt

        for xy_track, wh_track, track_id in zip(xy, wh, track_ids):
            x, y, w, h = xy_track[0], xy_track[1], wh_track[0], wh_track[1]

            # Добавляем центральное положение бокса в историю трека
            track = track_history[track_id]
            track.append((float(x + w / 2), float(y + h / 2)))

            # Ограничиваем длину истории треков
            if len(track) > max_track_length:
                track.pop(0)

            if len(track) > 1:  # Рисуем линию только если есть как минимум две точки в истории
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=3)

            if gt:
                # Рисуем bounding box
                cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 5)



    def tracking_image(self, path_images: list, gt: np.array, path_results: str,
                       output_video: str, imgsz: int = ConstValues.IMGSZ,
                       fps: int = ConstValues.FPS,
                       start_frame: int = ConstValues.START,
                       need_show: bool = False, need_save_video: bool = True):
        """
        Обрабатывает набор изображений, выполняя отслеживание объектов с использованием модели YOLO и создает аннотированное видео.

        Параметры:
        - path_images (list): список строк с путями к изображениям (кадрам) для обработки.
        - gt (np.array): массив с истинными значениями (ground truth) объектов.
        - model (model): модель отслеживания объектов.
        - path_results (str): путь для сохранения выходных результатов (видеофайлов и JSON).
        - output_video (str): имя выходного видеофайла (по умолчанию 'output_video.mp4').
        - fps (int): количество кадров в секунду для выходного видео (по умолчанию 24).
        - start_frame (int): начальный номер кадра (по умолчанию ConstValues.START).
        - device (torch.device): устройство, на котором проводятся расчеты.
        - need_show (bool): если True, отображает видео в окне во время обработки (по умолчанию False).
            Работает только при need_save_video=True.
        - need_save_video (bool): если True, то сохраняет аннотированное видео (по умолчанию True)

        Функция выполняет следующие действия:
        1. Инициализирует историю отслеживания для объектов и объектов во временном режиме (ground truth).
        2. Загружает первый кадр из списка изображений и определяет его размеры.
        3. Создает объект для записи видео с использованием указанного кодека и заданного FPS.
        4. Проходит по всем кадрам изображений по их путем:
            a. Загружает текущий кадр.
            b. Выполняет отслеживание объектов с использованием модели (например, YOLO).
            c. Если обнаружены объекты, получает их идентификаторы, координаты и размеры.
            d. Добавляет информацию об отслеживаемых объектах в список результатов.
            e. Визуализирует результаты (размеченные рамки) на текущем кадре.
            f. Если предоставлен ground truth (gt), добавляет его визуализацию.
            g. Добавляет номер кадра на аннотированный кадр.
            h. Сохраняет аннотированный кадр в выходное видео.
            i. Если параметр need_show установлен в True, отображает текущий аннотированный кадр.
            j. Позволяет прерывать выполнение функции при нажатии клавиши "q".
        5. Освобождает ресурсы: закрывает объект записи видео и уничтожает все окна OpenCV.
        6. Преобразует результаты отслеживания в массив и сохраняет их в формате JSON.
        7. Выводит сообщение о успешном сохранении данных в указанный файл JSON.

        Возвращает:
        - t (np.array): массив, содержащий информацию о отслеживаемых объектах.
        - delta_time_mean (float): среднее время расчета кадра.

        Примечание: Для корректного отображения аннотированного видео необходимо иметь установленные библиотеки OpenCV и YOLOv10.
        """

        # Создаем словари для хранения отслеживания и времени расчета
        track_history = defaultdict(lambda: [])
        track_history_gt = defaultdict(lambda: [])
        result_track = []
        time_list = []

        if need_save_video:
            # Читаем первый кадр изображения, чтобы получить его размеры
            frame = cv2.imread(path_images[0])
            height, width, _ = frame.shape

            # Определяем кодек для записи видео в формате MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Создаем объект для записи видео с заданными параметрами (путь, кодек, fps, размер)
            video_writer = cv2.VideoWriter(path_results + output_video, fourcc, fps, (width, height))

        # Проходим по всем изображениям и их индексам
        for frame_id, image_path in tqdm(enumerate(path_images)):

            # фиксируем время начала обработки кадра
            start_time = time()

            # Читаем текущее изображение
            frame = cv2.imread(image_path)

            # Обновляем frame_id, начиная с заданного значения
            frame_id = frame_id + start_frame

            # Выполнение отслеживания с помощью модели YOLOv8
            result = self.model.track(frame, persist=True, verbose=False, imgsz=imgsz, device=self.device)

            # Получаем результат отслеживания для первого результата (предполагается, что это будет YOLOv8)
            result_0 = result[0]

            # Проверяем, есть ли отслеживаемые объекты
            if result_0.boxes.id is None:
                continue  # Если объектов нет, переходим к следующему кадру

            # Получаем идентификаторы треков и координаты боксов объектов
            track_ids = result_0.boxes.id.int().cpu().tolist()  # Преобразуем идентификаторы в список
            xy_img = result_0.boxes.xyxy[:, :2].cpu()  # Получаем координаты верхнего левого угла боксов
            wh_img = result_0.boxes.xywh[:, 2:].cpu()  # Получаем ширину и высоту боксов
            conf = result_0.boxes.conf  # Получаем уверенность модели в распознавании объектов

            # Обрабатываем каждый отслеживаемый объект
            for i in range(len(track_ids)):
                # Добавляем информацию о треке в результат
                result_track.append((
                    frame_id,  # Идентификатор текущего кадра
                    track_ids[i],  # Идентификатор трек-идентификатора
                    xy_img[i, 0].item(),  # X координата верхнего левого угла
                    xy_img[i, 1].item(),  # Y координата верхнего левого угла
                    wh_img[i, 0].item(),  # Ширина бокса
                    wh_img[i, 1].item(),  # Высота бокса
                    1,  # Код состояния (например, 1 для активного)
                    2,  # Код типа объекта или другое значение (например, тип)
                    conf[i].item()  # Уверенность модели (значение от 0 до 1)
                ))

            # Визуализация результатов на кадре
            annotated_frame = result_0.plot()

            self.draw_track_and_boxes(annotated_frame, track_history, xy_img, wh_img, track_ids, gt=False)

            if gt is not None:
                track_ids_gt = gt[gt[:, 0] == frame_id, 1]
                xy_gt = gt[gt[:, 0] == frame_id, 2:4]
                wh_gt = gt[gt[:, 0] == frame_id, 4:6]

                self.draw_track_and_boxes(annotated_frame, track_history_gt, xy_gt, wh_gt, track_ids_gt)

            # Добавление номера кадра в верхний правый угол
            cv2.putText(annotated_frame, f"Frame: {frame_id}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.putText(annotated_frame, 'True', (50, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(annotated_frame, 'Pred', (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 2)

            stop_time = time()
            detla_time = timedelta(seconds=stop_time - start_time)
            time_list.append(detla_time)

            if need_save_video:
                # Сохранение аннотированного кадра в видео
                video_writer.write(annotated_frame)

                # Отображение аннотированного кадра
            if need_show:
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Условие выхода при нажатии "q"
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if need_save_video:
            # Освобождаем ресурсы
            video_writer.release()  # Закрываем писатель видео
            cv2.destroyAllWindows()

        t = np.array(result_track)

        # Преобразование в список для совместимости с JSON
        t_list = t.tolist()

        # Расчитываем среднее время обработки кадра
        delta_time_mean = round(np.mean(time_list).total_seconds(), 4)

        # Сохранение в JSON файл
        with open(path_results + f'pred_track_{ConstValues.START}_{ConstValues.STOP}.json', 'w') as json_file:
            json.dump(t_list, json_file)

        print(f"Данные успешно сохранены в {path_results}'pred_track_{ConstValues.START}_{ConstValues.STOP}.json")

        # Сохранение t_list в другой JSON файл
        with open(path_results + f'delta_time_mean_{ConstValues.START}_{ConstValues.STOP}.json', 'w') as json_file:
            json.dump(delta_time_mean, json_file)

        print(f"Данные успешно сохранены в {path_results}'delta_time_mean_{ConstValues.START}_{ConstValues.STOP}.json")

        return t, delta_time_mean


    def tracking_video(self, video_path: str, gt: np.array,
                       path_results: str = ConstValues.PATH_RESULTS,
                       name_suffix: str = 'output',
                       output_video: str = 'output_video.mp4',
                       imgsz: int = ConstValues.IMGSZ,
                       fps: int = ConstValues.FPS,
                       need_show: int = False, need_save_video: int = True):
        """
        Функция для отслеживания объектов на видео с помощью заданной модели и создания аннотированного видео.

        Параметры:
        - video_path (str): Путь к входному видеофайлу.
        - gt (np.array): Ground truth (проверка истинного значения) в виде массива, или None.
        - model (model): Модель отслеживания объектов.
        - path_results (str): Путь для сохранения выходных результатов.
        - name_suffix (str): Суффикс для имен выходных файлов.
        - output_video (str): Имя выходного видеофайла (по умолчанию 'output_video.mp4').
        - fps (int): Частота кадров для выходного видео (по умолчанию 24).
        - device (torch.device): Устройство (CPU/GPU) для выполнения.
        - need_show (bool): если True, отображает видео в окне во время обработки (по умолчанию False).
            Работает только при need_save_video=True.
        - need_save_video (bool): если True, то сохраняет аннотированное видео (по умолчанию True)

        Функция выполняет следующие действия:
        1. Инициализирует историю отслеживания для объектов и объектов во временном режиме (ground truth).
        2. Создает объект для записи видео с использованием указанного кодека и заданного FPS.
        3. Проходит по всем кадрам изображений по их путем:
            a. Загружает текущий кадр.
            b. Выполняет отслеживание объектов с использованием модели (например, YOLOv8).
            c. Если обнаружены объекты, получает их идентификаторы, координаты и размеры.
            d. Добавляет информацию об отслеживаемых объектах в список результатов.
            e. Визуализирует результаты (размеченные рамки) на текущем кадре.
            f. Если предоставлен ground truth (gt), добавляет его визуализацию.
            g. Добавляет номер кадра на аннотированный кадр.
            h. Сохраняет аннотированный кадр в выходное видео.
            i. Если параметр need_show установлен в True, отображает текущий аннотированный кадр.
            j. Позволяет прерывать выполнение функции при нажатии клавиши "q".
        4. Освобождает ресурсы: закрывает объект записи видео и уничтожает все окна OpenCV.
        5. Преобразует результаты отслеживания в массив и сохраняет их в формате JSON.
        6. Выводит сообщение о успешном сохранении данных в указанный файл JSON.

        Возвращает:
        - t (np.array): Массив с результатами отслеживания.
        - delta_time_mean (float): Среднее время обработки каждого кадра.
        """

        # Создаем словари для хранения отслеживания и времени расчета
        track_history = defaultdict(lambda: [])
        track_history_gt = defaultdict(lambda: [])
        result_track = []
        time_list = []

        # Открываем видео
        cap = cv2.VideoCapture(video_path)

        if need_save_video:
            # Получаем параметры видео (ширина, высота)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Настраиваем видеописатель для сохранения выходного видео
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(path_results + output_video, fourcc, fps, (width, height))

        frame_id = 0  # Идентификатор текущего кадра

        while cap.isOpened():  # Цикл, пока видео открыто
            ret, frame = cap.read()  # Читаем текущий кадр

            frame_id += 1  # Увеличиваем идентификатор кадра

            if not ret:
                break  # Выход из цикла, если не удалось прочитать кадр

            # Обработка кадра
            start_time = time()  # Запоминаем время начала обработки кадра

            result = self.model.track(frame, persist=True, verbose=False, imgsz=imgsz,
                                 device=self.device)  # Запуск модели отслеживания
            result_0 = result[0]

            if result_0.boxes.id is None:
                continue  # Пропускаем кадр, если ID объектов отсутствуют

            # Получаем данные о треках
            track_ids = result_0.boxes.id.int().cpu().tolist()
            xy_v = result_0.boxes.xyxy[:, :2].cpu()
            wh_v = result_0.boxes.xywh[:, 2:].cpu()
            conf = result_0.boxes.conf

            # Сохраняем результаты отслеживания в список
            for i in range(len(track_ids)):
                result_track.append((
                    frame_id,
                    track_ids[i],
                    xy_v[i, 0].item(),
                    xy_v[i, 1].item(),
                    wh_v[i, 0].item(),
                    wh_v[i, 1].item(),
                    1,
                    2,
                    conf[i].item()
                ))

            if need_save_video:
                annotated_frame = result_0.plot()  # Аннотируем кадр с результатами

                # Рисуем треки и bounding boxes
                self.draw_track_and_boxes(annotated_frame, track_history, xy_v, wh_v, track_ids, gt=False)

                # Если есть ground truth, рисуем его
                if gt is not None:
                    track_ids_gt = gt[gt[:, 0] == frame_id, 1]
                    xy_gt = gt[gt[:, 0] == frame_id, 2:4]
                    wh_gt = gt[gt[:, 0] == frame_id, 4:6]

                    self.draw_track_and_boxes(annotated_frame, track_history_gt, xy_gt, wh_gt, track_ids_gt)

                # Добавляем текст на кадр с идентификатором кадра и метками
                cv2.putText(annotated_frame, f"Frame: {frame_id}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255),2)
                cv2.putText(annotated_frame, 'True', (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.putText(annotated_frame, 'Pred', (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (230, 230, 230), 2)

            stop_time = time()  # Запоминаем время окончания обработки кадра
            delta_time = timedelta(seconds=stop_time - start_time)  # Вычисляем время обработки кадра
            time_list.append(delta_time)  # Сохраняем время в список

            if need_save_video:
                # Сохранение аннотированного кадра в выходное видео
                video_writer.write(annotated_frame)

                # Отображение аннотированного кадра
            if need_show:
                cv2.imshow("Tracking", annotated_frame)

                # Условие выхода при нажатии кнопки "q"
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if need_save_video:
            # Освобождаем ресурсы
            cap.release()
            video_writer.release()  # Закрываем писатель видео
            cv2.destroyAllWindows()

        t = np.array(result_track)  # Преобразуем результаты отслеживания в массив NumPy

        # Преобразование в список для совместимости с JSON
        t_list = t.tolist()

        # Рассчитываем среднее время обработки кадра
        delta_time_mean = round(np.mean(time_list).total_seconds(), 4)

        # Сохранение результатов отслеживания в JSON файл
        with open(path_results + f'pred_track_{name_suffix}.json', 'w') as json_file:
            json.dump(t_list, json_file)

        print(f"Данные успешно сохранены в {path_results}'pred_track_{name_suffix}.json")

        # Сохранение t_list в другой JSON файл, если это необходимо
        with open(path_results + f'delta_time_mean_{name_suffix}.json', 'w') as json_file:
            json.dump(delta_time_mean, json_file)

        print(f"Данные успешно сохранены в {path_results}'delta_time_mean_{name_suffix}.json")

        return t, delta_time_mean  # Возвращаем результаты и среднее время обработки кадров


    def mot_metrics(gt: np.array, t: np.array, start: int = ConstValues.START,
                    stop: int = ConstValues.STOP):
        """
        Вычисляет метрики отслеживания объектов (MOT) на основе данных о истинных значениях и результатах отслеживания
        с использованием методов отслеживания. Функция отслеживает изменения метрик по кадрам.

        Параметры:
            gt (np.array): Массив, содержащий истинные значения (ground truth) в формате,
                             где каждая строка представляет объект в формате [frame_id, object_id, x1, y1, x2, y2].
            t (np.array): Массив результатами отслеживания (tracking results) в формате,
                            аналогичном gt_source.

        Функция выполняет следующие действия:
            1. Загружает данные о истинных значениях и результатах отслеживания из файлов CSV.
            2. Инициализирует накопитель метрик для отслеживания.
            3. Проходит по каждому кадру в заданном диапазоне.
            4. Извлекает детекции истинных значений и детекции алгоритма для текущего кадра.
            5. Вычисляет матрицу дистанций (IoU) между истинными значениями и результатами отслеживания.
            6. Обновляет накопленные метрики на основе текущих детекций.
            7. Вычисляет метрики для текущего кадра.
            8. Сохраняет результаты метрик по кадрам для последующего анализа.
            9. Форматирует и отображает сводную строку со значениями метрик.

        Возвращает:
            tuple: Кортеж, состоящий из:
                - strsummary (str): Итоговые значения метрик (Recall, Precision, MOTA, MOTP).
                - summary_hist (list): История изменения метрик.
                - acc (motmetrics.mot.MOTAccumulator): Аккумулятор метрик.

        Примечание:
            Функция использует библиотеки для оценки отслеживания (например, `motmetrics`) и
            предполагает, что данные в файлах корректно отформатированы.
            Изменения в метриках отслеживания сохраняются по кадрам, что позволяет анализировать
            производительность алгоритма во времени.
        """

        # Инициализируем список для хранения сводных данных по кадрам
        summary_hist = []

        # Инициализируем накопитель метрик
        acc = mm.MOTAccumulator(auto_id=True)

        # Списки для хранения кадров
        frames = []

        # Проходим по каждому кадру в заданном диапазоне
        for frame in tqdm(range(start, stop)):
            # Извлекаем детекции для истинных значений и результатов отслеживания
            gt_dets = gt[gt[:, 0] == frame, 1:6]
            t_dets = t[t[:, 0] == frame, 1:6]

            # Вычисляем матрицу дистанций (IoU) между истинными значениями и результатами отслеживания
            C = mm.distances.iou_matrix(gt_dets[:, 1:], t_dets[:, 1:], max_iou=0.5)

            # Обновляем накопленные метрики на основе текущих детекций
            acc.update(
                gt_dets[:, 0].astype("int").tolist(), t_dets[:, 0].astype("int").tolist(), C)

            # Сохраняем текущий кадр
            frames.append(frame)

            # Вычисляем метрики для текущего кадра
            summary = mm.metrics.create().compute(
                acc, metrics=["recall", "precision", "mota", "motp"], name="acc"
            )

            # Создаем объект метрик
            mh = mm.metrics.create()
            # Вычисляем метрики
            summary = mh.compute(
                acc, metrics=["recall", "precision", "mota", "motp"], name="acc"
            )
            # Сохраняем сводные данные для текущего кадра
            summary_hist.append(summary)

        # Отображаем итоговые результаты
        strsummary = mm.io.render_summary(
            summary,
            namemap={
                "recall": "Recall",
                "precision": "Precision",
                "mota": "MOTA",
                "motp": "MOTP",
            },
        )

        # Возвращаем итоговые метрики и сводные данные
        return strsummary, summary_hist, acc


    def analyze_metrics(summary_hist: list, start: int = ConstValues.START):
        """
        Функция для анализа метрик отслеживания объектов и построения графиков их значений по временным кадрам.

        Параметры:
        summary_hist (list): Список DataFrame объектов, где каждый DataFrame содержит метрики
                             (recall, precision, MOTA, MOTP) для соответствующего кадра.
                             Предполагается, что каждая метрика представлена в первой строке
                             каждого DataFrame.

        Функция выполняет следующие действия:
        1. Инициализирует словарь для хранения значений метрик.
        2. Обходит список DataFrame объектов summary_hist, извлекая значения метрик
           для каждого кадра.
        3. Подготавливает временные метки (кадры) для графиков.
        4. Строит и отображает графики для каждой из метрик (recall, precision, MOTA, MOTP).

        Возвращает:
        None: Функция не возвращает значений, но визуализирует результаты в виде графиков для каждого из показателей.

        Примечание:
        Графики отображают изменения значений метрик по кадрам, что позволяет оценить
        производительность алгоритма отслеживания объектов во времени.
        """

        # Задаем названия метрик для анализа
        metrics = ['recall', 'precision', 'mota', 'motp']
        metric_names = ['Recall', 'Precision', 'MOTA', 'MOTP']

        # Инициализируем словарь для хранения данных метрик
        data = {metric: [] for metric in metrics}
        frames = []

        # Проходим по каждому кадру и извлекаем метрики
        for frame, summary in enumerate(summary_hist):
            frames.append(frame + start)  # Добавляем номер кадра в список
            for metric in metrics:
                # Извлекаем значение метрики и добавляем его в соответствующий список
                data[metric].append(summary.iloc[0][metric])

        # Построение графиков
        plt.figure(figsize=(12, 8))

        # Для каждой метрики создаем подграфик
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            plt.plot(frames, data[metric], marker='.', label=metric_names[i])  # Строим график
            plt.title(f'{metric_names[i]} по кадрам')  # Заголовок графика
            plt.xlabel('Кадр')  # Подпись оси x
            plt.ylabel(metric_names[i])  # Подпись оси y
            plt.xticks(frames[::20], rotation=45)  # Устанавливаем метки на оси x
            plt.grid()  # Включаем сетку на графиках

        plt.tight_layout()  # Автоматически настраиваем размещение графиков
        plt.show()  # Отображаем графики
