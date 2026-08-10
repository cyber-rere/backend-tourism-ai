"""
main.py
========
Backend لتطبيق سياحي ذكي (MVP هاكاثون) 
النسخة النهائية (Code Freeze) - تدعم:
- الربط مع واجهات السياق החقيقية (API Contract).
- اللغتين العربية والإنجليزية.
- حصر المدن (الرياض، الدرعية، جدة).
- أماكن مخصصة (Hooks) لرفع قاعدة البيانات (Supabase/PostgreSQL).
"""

import os
import json
import logging
import requests
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import google.generativeai as genai

# --------------------------------------------------------------------------
# 0) الإعداد العام (Config)
# --------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("الرجاء إضافة GEMINI_API_KEY في ملف .env")

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview") # يفضل استخدام 1.5 للسرعة والتقليل من الهلوسة

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("narrative-itinerary")

app = FastAPI(
    title="Saudi Smart Tourism — Narrative Itinerary API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# 1) إعدادات الربط مع واجهة السياق (Frontend APIs)
# --------------------------------------------------------------------------
TOURISM_API_URL = os.getenv("TOURISM_API_URL", "http://localhost:3000/api/tourism-context")

# حصر المدن المطلوبة في الهاكاثون فقط
CITY_COORDINATES = {
    "الرياض": {"lat": 24.7136, "lon": 46.6753},
    "الدرعية": {"lat": 24.7337, "lon": 46.5746},
    "جدة": {"lat": 21.4858, "lon": 39.1925},
}

# --------------------------------------------------------------------------
# 2) هياكل البيانات (Pydantic Models)
# --------------------------------------------------------------------------
class Coordinates(BaseModel):
    lat: float = Field(..., description="خط العرض (Latitude)")
    lng: float = Field(..., description="خط الطول (Longitude)")

class Source(BaseModel):
    name: str = Field(..., description="اسم المصدر (مثال: UNESCO)")

class StoryChapter(BaseModel):
    chapter_title: str = Field(..., description="عنوان الفصل القصصي")
    landmark_name: str = Field(..., description="اسم المعلم السياحي")
    coordinates: Coordinates = Field(..., description="إحداثيات المعلم")
    narrative_text: str = Field(..., description="النص السردي التاريخي")
    visit_reason: str = Field(..., description="سبب زيارة هذه المحطة الآن تحديداً")
    sources: List[Source] = Field(default_factory=list, description="مصادر المعلومات التاريخية")
    duration_hours: float = Field(..., description="المدة المقترحة لزيارة هذا المعلم بالساعات")

class NarrativeItinerary(BaseModel):
    trip_title: str = Field(..., description="عنوان الرحلة القصصية الكاملة")
    chapters: List[StoryChapter] = Field(..., description="قائمة فصول الرحلة")

class ItineraryRequest(BaseModel):
    # استخدام Literal هنا يرفض تلقائياً أي مدينة غير الرياض، جدة، الدرعية ويرجع خطأ 422 للواجهات
    current_location: Literal["الرياض", "جدة", "الدرعية"] = Field(..., description="المدينة")
    interests: List[str] = Field(..., description="اهتمامات السائح")
    available_hours: int = Field(..., description="عدد الساعات المتاحة للجولة")
    budget: Literal["اقتصادية", "متوسطة", "مرتفعة"] = Field(default="متوسطة")
    accessibility_needs: bool = Field(default=False)
    language: Literal["ar", "en"] = Field(default="ar", description="لغة الرد المطلوبة")

# --------------------------------------------------------------------------
# 3) تعليمات النظام (System Prompts)
# --------------------------------------------------------------------------
GROUNDING_STAGE_SYSTEM_PROMPT = """
أنت باحث تاريخي دقيق ومتحقق من الحقائق (Fact-checker). استخدم بحث Google
للتأكد من وجود معالم سياحية/تاريخية حقيقية وفعلية في المنطقة المطلوبة فقط.
اذكر مصدر كل حقيقة تاريخية بإيجاز.
"""

STORYTELLER_SYSTEM_PROMPT = """
أنت "راوٍ تاريخي" ومخطط لوجستي محترف لتطبيق سياحة في السعودية. مهمتك بناء مسار سياحي "قصصي" مترابط الفصول، بحيث:
1. ترتبط كل محطة سردياً بالمحطة التي قبلها.
2. تعتمد فقط على المعالم المؤكدة في السياق.
3. تُكيّف التوقيت بناءً على الطقس وأوقات الصلاة.
4. يتناسب عدد المحطات مع الوقت المتاح.
5. تكتب النص كاملاً باللغة المحددة لك.
6. إذا كانت قائمة "الاهتمامات" فارغة، افترض أن السائح مهتم بـ "أهم المعالم العامة".
7. كن مختصراً، لا تكرر الجمل، واجعل السرد لكل محطة لا يتجاوز 4 أسطر.
أعد الناتج بصيغة JSON فقط دون أي إضافات.
"""

# --------------------------------------------------------------------------
# 4) منطق العمل (Business Logic)
# --------------------------------------------------------------------------
def fetch_realtime_context(location: str) -> str:
    """المرحلة 1: جلب بيانات الطقس والصلاة من واجهة الـ API المتفق عليها."""
    coords = CITY_COORDINATES.get(location)
    if not coords:
        return "لا توجد بيانات."

    try:
        response = requests.get(
            TOURISM_API_URL,
            params={
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "destLat": coords["lat"],
                "destLon": coords["lon"],
                "prayerMethodId": 4
            },
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            temp = data.get("weather", {}).get("temperature", "غير متوفر")
            condition = data.get("weather", {}).get("weatherCondition", {}).get("label", "غير متوفر")
            next_prayer = data.get("prayerTimes", {}).get("nextPrayer", {}).get("displayName", "غير متوفر")
            time_rem = data.get("prayerTimes", {}).get("nextPrayer", {}).get("timeRemainingMinutes", "غير متوفر")
            
            return f"الطقس: {temp}°C ({condition}). الصلاة القادمة: {next_prayer} بعد {time_rem} دقيقة."
    except Exception as e:
        logger.warning(f"تعذر الاتصال بـ Tourism API: {e}")
    
    return "لا توجد أوقات صلاة قريبة تتعارض مع الجولة والطقس معتدل."

def run_grounding_stage(location: str, interests: List[str]) -> str:
    """المرحلة 2: Grounding — التحقق من المعالم والحقائق التاريخية عبر بحث Google."""
    prompt = (
        f"اذكر 3 إلى 5 معالم سياحية/تاريخية حقيقية وموجودة فعلياً في: "
        f"{location}, المملكة العربية السعودية، وذات صلة بالاهتمامات التالية: "
        f"{', '.join(interests)}.\n"
        "لكل معلم اذكر: الاسم الدقيق، الإحداثيات التقريبية (lat, lng)، "
        "وحقيقة تاريخية واحدة موثقة ومختصرة عنه، واذكر مصدر كل معلومة."
    )
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            # التعديل هنا: استخدام الصيغة الدقيقة التي طلبتها رسالة الخطأ
            tools=[{"google_search_retrieval": {}}],
            system_instruction=GROUNDING_STAGE_SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        logger.info("تم تنفيذ مرحلة Grounding عبر بحث Google بنجاح.")
        return response.text
        
    except Exception as exc:
        logger.warning(f"Grounding tool failed: {exc}, سيتم التجاوز للاحتياط.")
        # إرجاع نص ثابت كخطة بديلة لحماية السيرفر من خطأ 500 في حال انتهاء الكوتا
        return "يُرجى الاعتماد على المعالم التاريخية العامة الموثقة في المنطقة."

def run_story_generation_stage(location: str, interests: List[str], hours: int, realtime_context: str, grounded_facts: str, language: str) -> NarrativeItinerary:
    """المرحلة 3: Structured Output"""
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=STORYTELLER_SYSTEM_PROMPT)
    lang_name = "العربية" if language == "ar" else "English"

    prompt = f"""
    المعطيات:
    - الموقع: {location}
    - الاهتمامات: {', '.join(interests)}
    - المدة المتاحة: {hours} ساعة
    - لغة الرد المطلوبة حصراً: {lang_name}

    الظروف الواقعية: {realtime_context}
    المعالم المؤكدة: {grounded_facts}
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=NarrativeItinerary,
            temperature=0.6,
        ),
    )
    return NarrativeItinerary.model_validate_json(response.text)

# --------------------------------------------------------------------------
# 5) نقطة ربط قاعدة البيانات (DB Hook) للزميلة
# --------------------------------------------------------------------------
def save_itinerary_to_db(itinerary: NarrativeItinerary, request: ItineraryRequest):
    """
    مرحباً بزميلة الـ Database! 👋
    هذه الدالة مخصصة لكِ. يتم استدعاؤها بمجرد أن يولد الذكاء الاصطناعي المسار بنجاح.
    
    البيانات المتاحة لكِ هنا وتطابق السكيما الخاصة بك:
    - request.current_location / request.interests (لجدول users و trips)
    - itinerary.trip_title (لجدول trips)
    - itinerary.chapters (قائمة تحتوي على الفصول لجدول story_chapters و places)
    
    ضعي كود الـ Supabase/SQL الخاص بك هنا لإدخال البيانات.
    """
    logger.info("تم الوصول لنقطة حفظ قاعدة البيانات.")
    # مثال: db.table("trips").insert({"destination": request.current_location, ...})
    pass

# --------------------------------------------------------------------------
# 6) نقطة النهاية (Endpoint)
# --------------------------------------------------------------------------
@app.post("/generate-itinerary", response_model=NarrativeItinerary)
def generate_itinerary(request: ItineraryRequest):
    try:
        context = fetch_realtime_context(request.current_location)
        facts = run_grounding_stage(request.current_location, request.interests)
        itinerary = run_story_generation_stage(
            location=request.current_location,
            interests=request.interests,
            hours=request.available_hours,
            realtime_context=context,
            grounded_facts=facts,
            language=request.language
        )
        
        # استدعاء دالة قاعدة البيانات فوراً بعد التوليد
        save_itinerary_to_db(itinerary, request)
        
        return itinerary

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="خلل مؤقت في تنسيق البيانات. الرجاء المحاولة مجدداً.")
    except Exception as exc:
        msg = "تم استنفاد الطلبات (Quota limit)، نرجو المحاولة بعد قليل." if "429" in str(exc) else "حدث خطأ غير متوقع."
        raise HTTPException(status_code=500, detail=msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)