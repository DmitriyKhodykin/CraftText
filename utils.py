"""
Модуль со вспомогательными функциями.
"""


import cv2


def evaluate_document_quality(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    
    # Вычисление DPI
    dpi = image.shape[0] / image.shape[1]
    
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Вычисление Отношения сигнал/шум (SNR)
    snr = cv2.meanStdDev(gray)[0][0] / cv2.meanStdDev(gray)[1][0]

    # Вычисление контрастности изображения
    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()  

    # Вычисление яркости изображения
    brightness = cv2.mean(gray)[0]

    # Вычисление шумов изображения
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Оценка качества скан-копии на основе статистических метрик
    quality_score = (contrast + brightness - blur) / 3

    return quality_score, dpi, snr
