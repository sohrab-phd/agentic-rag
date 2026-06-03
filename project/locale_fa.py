"""User-facing Farsi strings for nodes and UI."""

CLARIFICATION_FALLBACK = "برای درک سوال شما به اطلاعات بیشتری نیاز دارم."

FORCE_SEARCH_MESSAGE = (
    "برای پاسخ به این پرسش، اولین قدم باید فراخوانی 'search_child_chunks' باشد."
)

NO_ANSWER = "امکان تولید پاسخ وجود نداشت."

NO_ANSWERS_GENERATED = "پاسخی تولید نشد."

NO_RETRIEVED_DATA = "داده‌ای از اسناد بازیابی نشد."

CONVERSATION_HISTORY = "تاریخچه گفتگو:\n"
ROLE_USER = "کاربر"
ROLE_ASSISTANT = "دستیار"

CONVERSATION_CONTEXT = "زمینه گفتگو:\n"
USER_QUERY = "پرسش کاربر:\n"

COMPRESSED_CONTEXT_HEADER = "[زمینه فشرده از تحقیقات قبلی]\n\n"

COMPRESSED_RESEARCH_CONTEXT = "## زمینه تحقیق فشرده (از تکرارهای قبلی)\n\n"
RETRIEVED_DATA_HEADER = "## داده بازیابی‌شده (تکرار فعلی)\n\n"
DATA_SOURCE = "--- منبع داده {i} ---\n"

USER_QUESTION_LABEL = "پرسش کاربر:\n"
CONVERSATION_TO_COMPRESS = "\n\nگفتگو برای فشرده‌سازی:\n\n"
PRIOR_COMPRESSED_CONTEXT = "[زمینه فشرده قبلی]\n"
TOOL_CALL_ONLY = "(فقط فراخوانی ابزار)"
TOOL_RESULT = "[نتیجه ابزار — {name}]\n"

ALREADY_EXECUTED_BLOCK = "\n\n---\n**قبلاً اجرا شده (تکرار نکن):**\n"
PARENT_CHUNKS_RETRIEVED = "بخش‌های والد بازیابی‌شده:\n"
SEARCH_QUERIES_RUN = "کوئری‌های جستجوی اجراشده:\n"

ORIGINAL_QUESTION = "پرسش اصلی کاربر: {query}\nپاسخ‌های بازیابی‌شده:{answers}"
ANSWER_N = "\nپاسخ {i}:\n"

FALLBACK_USER_PROMPT = (
    "پرسش کاربر: {question}\n\n"
    "{context}\n\n"
    "دستور:\nبهترین پاسخ ممکن را فقط با داده‌های بالا بده."
)

# --- Gradio UI ---
UI_TITLE = "دستیار RAG"
UI_TAB_DOCUMENTS = "اسناد"
UI_TAB_CHAT = "گفتگو"
UI_NO_DOCUMENTS = "📭 سندی در پایگاه دانش موجود نیست"
UI_UPLOAD_TITLE = "## افزودن سند جدید"
UI_UPLOAD_DESC = "فایل PDF یا Markdown بارگذاری کنید. موارد تکراری نادیده گرفته می‌شوند."
UI_FILE_LABEL = "فایل PDF یا Markdown را اینجا رها کنید"
UI_ADD_BTN = "افزودن اسناد"
UI_CURRENT_DOCS = "## اسناد فعلی در پایگاه دانش"
UI_REFRESH_BTN = "بروزرسانی"
UI_CLEAR_BTN = "حذف همه"
UI_ADDED_INFO = "✅ افزوده شد: {added} | رد شد: {skipped}"
UI_CLEARED_INFO = "🗑️ همه اسناد حذف شدند"
UI_CHAT_PLACEHOLDER = (
    "<strong>هر سوالی دارید بپرسید!</strong><br>"
    "<em>جستجو می‌کنم، استدلال می‌کنم و بهترین پاسخ را می‌دهم :)</em>"
)
UI_PROCESSING = "در حال پردازش {name}"
UI_REINDEXED = "پایگاه دانش با مدل embedding جدید بازسازی شد ({count} سند)."
RETRIEVAL_DIM_MISMATCH = (
    "خطای بازیابی: بردارهای ذخیره‌شده با مدل embedding فعلی هم‌خوان نیستند. "
    "از تب اسناد «حذف همه» را بزنید و فایل‌ها را دوباره بارگذاری کنید، یا برنامه را یک‌بار ری‌استارت کنید."
)
