# Document-Grounded Evaluation Report

**Date:** 2026-07-11T17:55:33.088035+00:00
**Model:** gemma4:e4b-it-qat
**Doc Relevance Score:** 0.808
**Overall Pass Rate:** 58.3% (7/12)

## By Category

| Category | Pass Rate | Source Hit | Passage Recall | Must Contain | Avg Latency |
|---|---:|---:|---:|---:|---:|
| direct | 70% | 1.0 | 0.8 | 0.9 | 69.79s |
| paraphrase | 0% | 1.0 | N/A | 0.5 | 126.3s |

## Failure Breakdown

- **retrieval_miss**: 4
- **generation_incomplete_or_wrong**: 1

## Failed Cases

### golestan_direct_004 (direct) — retrieval_miss
**Q:** رمز عبور اولیه گلستان چیست؟
**Search query:** رمز عبور اولیه سیستم گلستان چیست؟
**Answer:** رمز عبور اولیه گلستان به‌صورت پیشفرض کد ملی دانشجو است.

---
**منابع:**
سامانه گلستان.pdf...
**Top retrieval:** سامانه گلستان.pdf (score=1.000)

### golestan_direct_010 (direct) — generation_incomplete_or_wrong
**Q:** ثبت نهایی انتخاب واحد یعنی چه؟
**Search query:** ثبت نهایی انتخاب واحد دقیقاً شامل چه مراحلی است؟
**Answer:** در منابع بازیابی‌شده اطلاعات کافی برای پاسخ به این بخش وجود ندارد.

---
**منابع:**
سامانه گلستان.pdf...
**Top retrieval:** سامانه گلستان.pdf (score=0.833)

### golestan_direct_016 (direct) — retrieval_miss
**Q:** تداخل دروس یعنی چه؟
**Search query:** تعریف تداخل دروس چیست؟
**Answer:** تداخل دروس به همپوشانی زمانی کلاس‌ها یا امتحانات دو درس گفته می‌شود.

---
**منابع:**
سامانه گلستان.pdf...
**Top retrieval:** سامانه گلستان.pdf (score=1.000)

### golestan_para_001 (paraphrase) — retrieval_miss
**Q:** اولین بار که وارد گلستان می‌شوم، رمزم چیست؟
**Search query:** رمز ورود برای اولین استفاده در سامانه گلستان چیست؟
**Answer:** رمز عبور اولیه در سامانه گلستان به‌صورت پیش‌فرض کد ملی دانشجو است.

---
**منابع:**
سامانه گلستان.pdf...
**Top retrieval:** سامانه گلستان.pdf (score=0.833)

### golestan_para_002 (paraphrase) — retrieval_miss
**Q:** سیستم به من اجازه ثبت‌نام دروس را نمی‌دهد؛ دلیلش چیست؟
**Search query:** چرا سیستم اجازه ثبت‌نام در دروس را برای من نمی‌دهد؟
**Answer:** دلایل عدم نمایش برخی دروس در سیستم عبارتند از: سیاست و محدودیت گروه آموزشی، ارائه درس فقط برای ورودی‌های خاص، یا رعایت نشدن پیشنیاز یا همنیاز.

---
**منابع:**
سامانه گلستان.pdf...
**Top retrieval:** سامانه گلستان.pdf (score=1.000)
