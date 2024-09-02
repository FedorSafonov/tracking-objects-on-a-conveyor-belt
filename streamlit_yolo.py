import io
import time
from collections import defaultdict
import numpy as np

import cv2
import torch

from ultralytics.utils.checks import check_requirements

from config import ConstValues


def inference(model=None):
    """
    Запуск модели трекинга объектов в реальном времени с использованием Ultralytics YOLOv10
    """
    check_requirements("streamlit>=1.29.0")  # импорт области применения для ускорения загрузки пакетов ultralytics
    import streamlit as st

    from ultralytics import YOLO

    # Используем GPU по возможности
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Скрываем главное меню
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Заголовок приложения
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Визуализация работы модели Ultralytics YOLOv10
                    </h1></div>"""

    # Подзаголовок приложения
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Эксперимент по детектированию объектов в реальном времени с вебкамеры или загруженного файла</h4>
                    </div>"""

    # Указываем html конфигурацию страницы
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

    # Добавляем кастомные настройки HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Добавляем заголовок заказчика
    with st.sidebar:
        logo = "https://renue.ru/images/renue-logo.png"
        st.image(logo, width=100)


    # Добавляем элементы в вертикальное меню
    st.sidebar.title("Пользовательская конфигурация")

    # Добавляем выбор источника видеозаписи
    source = st.sidebar.selectbox(
        "Загрузка видео",
        ("Вебкамера", "Видео"),
    )

    vid_file_name = ""
    if source == "Видео":
        vid_file = st.sidebar.file_uploader("Загрузка видеофайла", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())  # BytesIO Object
            vid_location = "ultralytics.mp4"
            with open(vid_location, "wb") as out:  # Open temporary file as bytes
                out.write(g.read())  # Read bytes into file
            vid_file_name = "ultralytics.mp4"
    elif source == "Вебкамера":
        vid_file_name = 0

    # Выбор доступных моделей. Не используется в текущем проекте
    available_models = ['yolov10x']
    if model:
        available_models.insert(0, model.split(".pt")[0])  # insert model without suffix as *.pt is added later

    st.sidebar.selectbox("Модель", available_models)
    with st.spinner("Модель загружается..."):
        model = YOLO(ConstValues.PATH_MODEL)  # Загрузка модели
        class_names = list(model.names.values())  # Конвертация словаря в список имен классов
    st.success(f"Модель успешно загружена! На устройстве доступны ресурсы: {'CPU' if device == 'cpu' else 'GPU'}")

    # Многовариационное окно выбора классов
    selected_classes = st.sidebar.multiselect("Классы объектов", class_names, default=class_names[:3])
    selected_ind = [class_names.index(option) for option in selected_classes]

    if not isinstance(selected_ind, list):
        selected_ind = list(selected_ind)

    # Параметры для модели трекинга
    conf = float(st.sidebar.slider("Порог классификации объекта", 0.0, 1.0, 0.75, 0.01))
    iou = float(st.sidebar.slider("Параметр IoU", 0.0, 1.0, 0.45, 0.01))
    track_len = int(st.sidebar.slider("Длина трека объекта", 5, 50, 25, 1))

    # Окно видео и отображение реального FPS
    col, col1 = st.columns(2)
    ann_frame = col.empty()
    fps_display = st.sidebar.empty()

    if st.sidebar.button("Запуск"):
        videocapture = cv2.VideoCapture(vid_file_name)  # Вывод видео

        track_history = defaultdict(lambda: [])

        if not videocapture.isOpened():
            st.error("Не найдена вебкамера")

        stop_button = st.button("Стоп")  # Кнопка остановки видео

        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning("Ошибка записи с камеры. Пожалуйста прововерьте подключение камеры.")
                break

            prev_time = time.time()

            # Инициализация модели
            results = model.track(frame, persist=True, conf=conf, iou=iou,
                                  classes=selected_ind, device=device)
            boxes = results[0].boxes.xywh
            if results[0].boxes.id is None:
                continue

            track_ids = results[0].boxes.id.int().tolist()
            annotated_frame = results[0].plot()

            # Отрисовка рамок объектов и их трекинга
            for box, track_id in zip(boxes, track_ids):

                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))

                # Ограничиваем длину трека объекта
                if len(track) > track_len:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False,
                              color=(230, 230, 230), thickness=10)

            # Расчет реального FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Отображение окна видео
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()  # Окончание видео
                torch.cuda.empty_cache()  # Очистка памяти
                st.stop()  # Остановка приложения

            # Отображение FPS
            fps_display.metric("FPS", f"{fps:.2f}")

        # Остановка видео
        videocapture.release()

    # Очистка памяти
    torch.cuda.empty_cache()

    # Закрытие окна
    cv2.destroyAllWindows()

# Запуск кода
if __name__ == "__main__":
    inference()