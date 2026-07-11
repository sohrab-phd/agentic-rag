# Phase 1 — Query Rewrite Audit

**Cases audited:** 7 (paraphrase + multi-intent)
**Failed in pipeline A:** 6
**Retrieval regressions caused by rewrite:** 0
**Answer regressions where rewrite retrieval was OK but B passed:** 2
**Would pass with original-query grounded answer:** 2
**A/B retrieval improved with original query:** 0/7
**A/B answer improved with original query:** 2/7

## Drift Tags (failed cases)

- **missing_entities**: 6
- **dropped_keywords**: 6
- **intent_collapse**: 1

## Failed Case Details

### golestan_para_001 (paraphrase)
**Original:** اولین بار که وارد گلستان می‌شوم، رمزم چیست؟
**Rewritten:** رمز ورود برای اولین استفاده در سامانه گلستان چیست؟
**Drift tags:** missing_entities, dropped_keywords
**Pipeline A pass:** False (retrieval_miss)
**B retrieval passage_recall:** False
**B grounded answer pass:** False

### golestan_para_002 (paraphrase)
**Original:** سیستم به من اجازه ثبت‌نام دروس را نمی‌دهد؛ دلیلش چیست؟
**Rewritten:** چرا سیستم اجازه ثبت‌نام در دروس را برای من نمی‌دهد؟, دلایل عدم امکان ثبت‌نام در دروس از طریق سیستم چیست؟
**Drift tags:** missing_entities, dropped_keywords
**Pipeline A pass:** False (retrieval_miss)
**B retrieval passage_recall:** False
**B grounded answer pass:** False

### golestan_para_003 (paraphrase)
**Original:** مسیر ثبت‌نام دروس در گلستان کجاست؟
**Rewritten:** مسیر و مراحل ثبت‌نام دروس در سامانه گلستان چیست؟
**Drift tags:** missing_entities, dropped_keywords
**Pipeline A pass:** False (retrieval_miss)
**B retrieval passage_recall:** False
**B grounded answer pass:** False

### golestan_para_005 (paraphrase)
**Original:** حداکثر تعداد واحد مجاز برای کارشناسی چقدر است؟
**Rewritten:** حداکثر تعداد واحدهای مجاز برای انجام کارشناسی چقدر است؟
**Drift tags:** missing_entities, dropped_keywords
**Pipeline A pass:** False (generation_incomplete_or_wrong)
**B retrieval passage_recall:** True
**B grounded answer pass:** True

### multi_001 (multi_intent)
**Original:** رمز اولیه گلستان چیست و اگر فراموش کنم چه کنم؟
**Rewritten:** رمز اولیه گلستان چیست؟, اگر رمز گلستان را فراموش کردم، چگونه می‌توان آن را بازیابی کرد؟
**Drift tags:** missing_entities, dropped_keywords
**Pipeline A pass:** False (generation_incomplete_or_wrong)
**B retrieval passage_recall:** True
**B grounded answer pass:** True

### multi_002 (multi_intent)
**Original:** استخراج رابطه چیست و یادگیری انتقالی چه نقشی دارد؟
**Rewritten:** رابطه در زمینه مورد بحث چیست؟, یادگیری انتقالی چه نقشی ایفا می‌کند؟
**Drift tags:** missing_entities, dropped_keywords, intent_collapse
**Pipeline A pass:** False (retrieval_miss)
**B retrieval passage_recall:** False
**B grounded answer pass:** False
