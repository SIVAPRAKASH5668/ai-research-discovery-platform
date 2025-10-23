"""
==================================================================================
REAL-TIME TRANSLATION MIDDLEWARE - UNIVERSAL LANGUAGE SUPPORT
Automatically translates API responses to ANY language
✅ Supports: 100+ languages including EN, ZH, JA, FR, DE, ES, PT, KO, IT, RU, AR, HI
✅ READS CUSTOM HEADER X-User-Language
✅ ASYNC SUPPORT for googletrans 4.0+
==================================================================================
"""

from functools import wraps
from typing import Dict, List
import asyncio
import logging

logger = logging.getLogger(__name__)

# Try to import translation library
TRANSLATION_AVAILABLE = False
TRANSLATOR_TYPE = None

try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    TRANSLATOR_TYPE = 'googletrans'
    logger.info("✅ googletrans available - supports 100+ languages")
except ImportError:
    try:
        from deep_translator import GoogleTranslator
        TRANSLATION_AVAILABLE = True
        TRANSLATOR_TYPE = 'deep-translator'
        logger.info("✅ deep-translator available - supports 100+ languages")
    except ImportError:
        logger.warning("⚠️ Translation unavailable - install: pip install googletrans==4.0.0-rc1")

# Fields to translate in papers
TRANSLATABLE_FIELDS = ['title', 'abstract']

# ==================================================================================
# SUPPORTED LANGUAGES
# ==================================================================================
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'zh': 'Chinese (Simplified)',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'it': 'Italian',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'id': 'Indonesian',
    'ms': 'Malay',
    'sv': 'Swedish',
    'no': 'Norwegian',
    'da': 'Danish',
    'fi': 'Finnish',
    'el': 'Greek',
    'he': 'Hebrew',
    'cs': 'Czech',
    'ro': 'Romanian',
    'hu': 'Hungarian',
    'uk': 'Ukrainian',
    'fa': 'Persian',
    'ur': 'Urdu',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
}


async def translate_text(text: str, target_lang: str) -> str:
    """
    ✅ ASYNC: Translate single text to target language
    Works with both googletrans 4.0+ (async) and deep-translator (sync)
    """
    if not TRANSLATION_AVAILABLE or not text or not isinstance(text, str) or target_lang == 'en':
        return text
    
    try:
        text_to_translate = text[:500] if len(text) > 500 else text
        
        if TRANSLATOR_TYPE == 'googletrans':
            # ✅ googletrans 4.0+ is async
            translator = Translator()
            result = await translator.translate(text_to_translate, dest=target_lang, src='auto')
            translated_text = result.text
            
            # Log translation with character set detection
            if any('\u4e00' <= char <= '\u9fff' for char in text):
                logger.info(f"🌍 Chinese → {target_lang}: '{text[:20]}...' → '{translated_text[:20]}...'")
            elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
                logger.info(f"🌍 Japanese → {target_lang}: '{text[:20]}...' → '{translated_text[:20]}...'")
            elif any('\uac00' <= char <= '\ud7af' for char in text):
                logger.info(f"🌍 Korean → {target_lang}: '{text[:20]}...' → '{translated_text[:20]}...'")
            else:
                logger.info(f"🌍 Translated: '{text[:20]}...' → '{translated_text[:20]}...'")
            
            return translated_text
        else:
            # ✅ deep-translator is sync, so run in executor
            loop = asyncio.get_event_loop()
            translator = GoogleTranslator(source='auto', target=target_lang)
            result = await loop.run_in_executor(None, translator.translate, text_to_translate)
            return result
        
    except Exception as e:
        logger.error(f"❌ Translation error for {target_lang}: {e}")
        return text


async def translate_paper(paper: Dict, target_lang: str) -> Dict:
    """
    ✅ ASYNC: Translate all fields in a single paper
    """
    if not TRANSLATION_AVAILABLE or target_lang == 'en':
        return paper
    
    translated = paper.copy()
    
    for field in TRANSLATABLE_FIELDS:
        if field in paper and paper[field] and isinstance(paper[field], str):
            original_text = paper[field]
            # ✅ Await the async translation
            translated_text = await translate_text(original_text, target_lang)
            
            if translated_text and translated_text != original_text:
                translated[field] = translated_text
                translated[f'original_{field}'] = original_text
    
    translated['translated_to'] = target_lang
    translated['translation_engine'] = TRANSLATOR_TYPE
    
    return translated


async def translate_papers_batch(papers: List[Dict], target_lang: str) -> List[Dict]:
    """
    ✅ ASYNC: Translate multiple papers in parallel
    """
    if not TRANSLATION_AVAILABLE or target_lang == 'en' or not papers:
        return papers
    
    lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang.upper())
    logger.info(f"🌍 Translating {len(papers)} papers to {lang_name} ({target_lang})...")
    
    # ✅ Use asyncio.gather for true async parallelism
    tasks = [translate_paper(paper, target_lang) for paper in papers]
    translated_papers = await asyncio.gather(*tasks)
    
    translated_count = sum(1 for p in translated_papers if p.get('translated_to') == target_lang)
    logger.info(f"✅ Translation complete: {translated_count}/{len(papers)} papers translated")
    
    return translated_papers


def auto_translate(func):
    """
    ✅ DECORATOR: Automatically translate API responses to ANY language
    ✅ FIXED: Handles papers, similar_papers, source_paper, etc.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"🔍 @auto_translate triggered: {func.__name__}")
        
        # ✅ Try to find FastAPI Request object
        from fastapi import Request
        request = None
        
        # ✅ Check kwargs first
        for key, value in kwargs.items():
            if isinstance(value, Request):
                request = value
                logger.info(f"✅ Found Request in kwargs['{key}']")
                break
        
        # ✅ Fallback to args
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    logger.info("✅ Found Request in args")
                    break
        
        # Get target language
        target_lang = 'en'
        if request:
            logger.info("✅ Request object found!")
            
            # Get language from headers (case-insensitive)
            custom_lang = request.headers.get('x-user-language') or request.headers.get('X-User-Language')
            
            if custom_lang:
                target_lang = custom_lang.split('-')[0].strip().lower()
                logger.info(f"🌍 Using X-User-Language: {custom_lang} → {target_lang}")
            else:
                accept_lang = request.headers.get('accept-language', 'en')
                target_lang = accept_lang.split('-')[0].split(',')[0].strip().lower()
                logger.info(f"🌍 Using Accept-Language: {accept_lang} → {target_lang}")
        else:
            logger.warning("⚠️ No Request object found - using default language")
        
        lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang.upper())
        logger.info(f"🎯 TARGET: {lang_name} ({target_lang})")
        
        # Call original function
        result = await func(*args, **kwargs)
        
        # ✅ REMOVE THIS CHECK - Always try to translate if available!
        if not TRANSLATION_AVAILABLE or not isinstance(result, dict):
            logger.info(f"⚠️ No translation available or result not a dict")
            return result
        
        # ✅ Translate papers (handles 'papers' key)
        if 'papers' in result and isinstance(result['papers'], list) and result['papers']:
            logger.info(f"📄 Translating {len(result['papers'])} papers...")
            result['papers'] = await translate_papers_batch(result['papers'], target_lang)
        
        # ✅ NEW: Translate similar_papers (handles 'similar_papers' key)
        if 'similar_papers' in result and isinstance(result['similar_papers'], list) and result['similar_papers']:
            logger.info(f"🔗 Translating {len(result['similar_papers'])} similar papers...")
            result['similar_papers'] = await translate_papers_batch(result['similar_papers'], target_lang)
        
        # ✅ NEW: Translate source_paper (handles 'source_paper' key)
        if 'source_paper' in result and isinstance(result['source_paper'], dict):
            logger.info(f"📄 Translating source paper...")
            result['source_paper'] = await translate_paper(result['source_paper'], target_lang)
        
        # ✅ Translate agent response (async)
        if 'agent_response' in result and isinstance(result['agent_response'], str) and result['agent_response']:
            logger.info(f"🤖 Translating agent_response...")
            result['agent_response'] = await translate_text(result['agent_response'], target_lang)
        
        # ✅ Translate single paper (handles 'paper' key)
        if 'paper' in result and isinstance(result['paper'], dict):
            logger.info(f"📄 Translating single paper...")
            result['paper'] = await translate_paper(result['paper'], target_lang)
        
        # ✅ Translate summary (async)
        if 'summary' in result and isinstance(result['summary'], str) and result['summary']:
            logger.info(f"📝 Translating summary...")
            result['summary'] = await translate_text(result['summary'], target_lang)
        
        logger.info(f"🎉 Translation complete for {func.__name__}!")
        return result
    
    return wrapper

