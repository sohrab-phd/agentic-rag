from typing import List
from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(
        description="آیا پرسش کاربر واضح و قابل پاسخ است."
    )
    questions: List[str] = Field(
        description="فهرست پرسش‌های بازنویسی‌شده و مستقل به فارسی."
    )
    clarification_needed: str = Field(
        description="در صورت نامشخص بودن پرسش، توضیح روشن‌سازی به فارسی."
    )
