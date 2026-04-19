## تشغيل سريع

[![Open Notebook](https://img.shields.io/badge/Open%20Notebook-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/kareemkamal10/NAMAA-Egyptian-Voice/blob/main/NAMAA_Colab_T4.ipynb)

## المحتويات

- `app.py`: تشغيل المشروع عبر CLI بنمط جلسات (تفاعلي أو مباشر).
- `scripts/setup_env.py`: سكريبت Python مستقل لإنشاء virtualenv وتثبيت المتطلبات.
- `NAMAA_Colab_T4.ipynb`: النوتبوك الأساسية.
- `NAMAA_Colab_T4_TEST.ipynb`: نوتبوك تجريبية للتعديلات قبل النقل للنسخة الأساسية.
- `requirements.txt`: الاعتمادات المثبتة بإصدارات محددة.

## إعداد البيئة (مستقل عن Colab)

من داخل المشروع:

```bash
python scripts/setup_env.py --venv-path .venv --project-dir .
```

خيارات مفيدة:

```bash
python scripts/setup_env.py --venv-path .venv --project-dir . --recreate
python scripts/setup_env.py --venv-path /content/namaa-venv --project-dir /content/NAMAA-Egyptian-Voice
```

## CLI بنمط الجلسة

### جلسة تفاعلية

```bash
python app.py --new --outputname output
```

سيطلب منك:
- النص (مطلوب)
- الريفرنس الصوتي (اختياري)
- `exaggeration`, `cfg_pace`, `seed`, `temperature` (اختياري بقيم افتراضية)

### جلسة مباشرة

```bash
python app.py --new --prompt "انا سبت الشغل" --reference ./female_voice.wav --exaggeration 0.5 --cfg-pace 0.5 --seed 0 --temperature 0.8 --outputname demo
```

### ملاحظات الإخراج

- `--outputname` يحدد اسم الملف الأساسي.
- لو الاسم موجود، يتم إنشاء اسم جديد تلقائيًا مثل: `output_1.wav`, `output_2.wav`.
- المسار الافتراضي للإخراج: `outputs/`.

## التشغيل على Colab

1. افتح النوتبوك من الزر بالأعلى.
2. اختر Runtime نوع GPU.
3. شغّل خلية الإعداد ثم خلايا الاستخدام.
4. راجع ملفات التتبع داخل مجلد `logs/` عند الحاجة.
