# Проект преобразования скана российского паспорт (первый разворот) в структурированную форму

![Пример](/static/example.png)

### Извлечение данных из машиночитаемой зоны паспорта

* Для заполнения позиций знаков 6 - 44 верхней строки машиночитаемой записи используется способ 
кодирования информации "модернизированный клер", при котором буквам русского алфавита 
соответствуют определенные буквы латинского алфавита и арабские цифры (см. списки букв в коде).

* Для заполнения позиций знаков нижней строки машиночитаемой записи используется цифровой способ 
кодирования информации, кроме позиций знаков 11 - 13 и 21.

* Машиночитаемые данные располагаются слева направо в две строки (верхняя и нижняя) фиксированной длины. 
Машиночитаемые данные вносятся начиная с левой позиции знаков. 

* Позиции знаков 10, 20, 28, 43, 44 нижней строки машиночитаемой записи содержат контрольные цифры.

* Если вводимые машиночитаемые данные не занимают все позиции знаков, для заполнения оставшихся позиций 
используется знак-заполнитель.