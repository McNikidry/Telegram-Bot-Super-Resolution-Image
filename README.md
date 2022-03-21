# Telegram bot 


Телеграмм бот для работы с изображениям. Бот включает в себя увеличение размерности изображения, удаление гауссовских шумов, гауссовского размытия, раскраска изображения из ЧБ.
Бот написан с использование библиотеки [telegram](https://python-telegram-bot.readthedocs.io/en/stable/)

#### Статьи, которые были использованы для реализации

| Название статьи | Ссылка |
| ------ | ------ |
| Deep Koalarization | https://arxiv.org/abs/1712.03400 |
| Real Image Denoising with Feature Attention | https://arxiv.org/pdf/1904.07396.pdf |
| Beyond a Gaussian Denoiser: Residual Learning of
Deep CNN for Image Denoising | https://arxiv.org/pdf/1608.03981.pdf |
| Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network | https://arxiv.org/pdf/1609.04802.pdf |

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




