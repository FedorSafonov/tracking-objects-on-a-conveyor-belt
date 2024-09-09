# Трекинг объектов на конвейерной ленте

- Полный отчёт по проекту можно посмотреть в [report.md](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/report.md)
- [Презентация](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/Presentation_Renue.pdf)

## Описание

### С каждым годом количество мусора, генерируемое человеком растет, поэтому сортировка, переработка, а также автоматизация этого процесса является актуальной задачей.

### Цель

Целью данного проекта является разработка модели потокового трекинга пластиковой тары на конвейере, которая выдает координаты центра обнаруженных объектов для каждого видеокадра. 

### Заказчик

Renue, IT-компания из Екатеринбурга

Компания разрабатывает высоконагруженные информационные системы для крупных российских заказчиков, для бизнеса и государства

### **Инструкции по использованию:**
Для того, чтобы корректно работать с данным проектом, необходимо:
```
- выгрузить файлы config.py, init_model.py, main.py, requirements.txt

- создать виртуальное окружение, в которое необходимо уставить библиотеки, указанные в файле requirements.txt

- в файле config.py необходимо указать пути к тестовому видео и фото, количество фото которые хотим обработать,
  файлу предобученной модели yolov10x_v2_4_best.pt и к папке, в которую будут сохраняться результаты.

- после этого запускаем файл main.py
```

### Исходные данные

1) Предобученная модель детекции с 15 классами распознаваемых объектов:

- PET (transparent) (green)
- PET (transparent) (brown)
- PET (transparent) (blue)
- PET (transparent)
- PET (transparent) (dark blue)
- PET (black)
- PET (white)
- PET (sticker)
- PET (flacon)
- PET (household chemicals)
- PND (household chemicals)
- PND packet
- Other plastic
- Other plastic (transparent)
- Not plastic

2) Датасет (изображения + разметка) в нескольких форматах: MOT, COCO, CVAT.
3) Примеры видеозаписей работы конвейера (в том числе плохого качества).

### Метрика и условия:
- Скорость обработки кадра должна быть не более 100 мс. 
- Метрика оценки модели – MOTA (Multiple Object Tracking Accuracy).

## Выбор модели

В проекте использовалось:
- предобученная модель yolov10x_v2_4_best
- трекеры BoT-Sort и ByteTrack

**Результат:**
- видео с бибоксами объектов, их классом и треком.
- расчитанные метрики работы модели для фото и видео.

## Структура репозитория:

| #    | Наименование файла                | Описание   |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1.   | [README.md](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/README.md) | Представлена основная информация по проекту и его результатах   |
| 2.   | [config.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/config.py) | Класс, в котором задаются константы и пути к файлам   |
| 3.   | [init_model.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/init_model.py) | Класс, который инициализирует работу модели детекции и трекинга объектов и всех функций необходимых для визуализации их работы и расчета метрик   |
| 4.   | [main.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/main.py) | Код запуска всех методов имеющихся в классах    |
| 5.   | [requirements.txt](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/requirements.txt) | Список всех библиотек и их версии, необходимых для установки в виртуальной среде для запуска кода проекта   |
| 6.   | [streamlit_yolo.py](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/streamlit_yolo.py) | Веб приложение в стримлит с моделью заказчика YOLA |
| 7.   | [report.md](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/report.md) | Отчёт о проделанной работе |
| 8.   | [Renue_group_5.ipynb](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/Renue_group_5.ipynb) | Тетрадка с ходом исследования |
| 9.   | [Presentation_Renue.pdf](https://github.com/FedorSafonov/tracking-objects-on-a-conveyor-belt/blob/main/Presentation_Renue.pdf) | Презентация проекта |

## Итоги

* **Основные выводы:**  
В ходе проекта были протестированы 2 трекера: SORT (Simple Online and Realtime Tracking) и ByteTrack с разным размером изображений на выходе (imgsz) и разным количеством кадров. В результате трекер BoT-SORT показал более высокие значения MOTA, однако скорость работы ByteTrack была в 1,7-2,7 раза выше в зависимости от размера изображения. Также отмечено, что скорость и точность работы трекеров значительно зависит от среды, в которой выполняется обработка видео. В целом модель хорошо справляется с разными форматами и разным количеством кадров, показывая высокую метрику (0,909-0,918).
Для заявленной цели – создать модель потокового трекинга пластиковой тары на конвейере, которая выдает координаты центра обнаруженных объектов для каждого видеокадра со скоростью обработки потока не более 100 мс – больше подойдет предобученная модель YOLOv10 с трекером BoT-SORT.
 
* **Проект может быть развит в следующих направлениях:**
   * **Использование других моделей и трекеров:** Deep SORT, FairMOT, DEVA, CSRT.
   * **Обучение моделей на датасете большего размера:**  Обучение моделей на большем датасете может улучшить их качество и обобщающую способность.
   * **Изменить положение записывающей камеры:**
     - Переместить камеру дальше от места подачи мусора, чтобы исключить "влетание" мусора в кадр и дальнейшее его передвижение по конвейерной ленте.
     - Поднять камеру выше над транспортерной лентой. При том же угле обзора это расширит поле зрения камеры и позволит модели дольше сопровождать объекты, что, в свою очередь, даст выигрыш в качестве трекинга.  
   * **Использование Streamlit/FastAPI для более удобного трекинга и детекции объектов и анализа результатов**

## Cтатус: 
Завершён.

## Стэк:
Python, Ultralytics, Motmetrics, OpenCV, PyTorch, Numpy, SciPy, time, timedelta, PIL, OS, Glob

## Команда проекта
- [Федор Сафонов (TeamLead)](https://github.com/FedorSafonov)
- [Анна Йорданова](https://github.com/A-Yordanova)
- [Юрий Кашин](https://github.com/yakashin)
- [Александр Вотинов](https://github.com/VotinovAlS)
- [Гульшат Зарипова](https://github.com/gulshart)
- [Сергей Пашкин](https://github.com/DrSartoriuss)
- [Александр Глазунов](https://github.com/pzae)

