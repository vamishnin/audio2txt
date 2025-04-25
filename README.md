# audio2txt
Speech-to-text + speaker diarisation pipeline.

Usage
-----
Заполнить блок Configuration. Положить в папку со скриптом файл в формате wav. \
Запустить.
```shell
python transcribe.py
```


Environment
-----------
HF_AUTH_TOKEN должен быть прописан в `.env`. Токен можно получить после регистрации на \
https://huggingface.co и подписавшись на модель https://huggingface.co/pyannote/segmentation-3.0 \
Также, нужно иметь в виду, что под сохранение модели на диск потребуется 5-7 ГБ места.