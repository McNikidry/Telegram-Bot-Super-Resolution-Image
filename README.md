# Telegram bot 


Телеграмм бот для работы с изображениям. Бот включает в себя увеличение размерности изображения, удаление гауссовских шумов, гауссовского размытия, раскраска изображения из ЧБ.

#### Статьи, которые были использованы для реализации

| Название статьи | Ссылка |
| ------ | ------ |
| Deep Koalarization | https://arxiv.org/abs/1712.03400 |
|  | |
| |  |
|  |  |

#### Дополнение
В качестве дополнения имеется файл
```sh
detection.py,
```
который ищет фигуры на ЧБ изображении и раскрашивает в определенные цвета. Выполнено c использованием OpenCV.

Для улучшения результатов в `Deep Koalarization` использовалось добавление `BatchNormalization`, что отсутствует в оригинальной статье.

Для обучения использовался датасет - `Div_2k`
#### Авторы работы
 [Соловьев Никита](https://github.com/McNikidry)
 [Харитонов Кирилл](https://github.com/KirillKharitonov)




