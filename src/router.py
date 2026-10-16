"""Question router for HR, IT, and Finance domains."""

import re
import unicodedata

DOMAIN_KEYWORDS = {
    "hr": {
        # English keywords
        "leave",
        "vacation",
        "holiday",
        "employee",
        "employees",
        "attendance",
        "absence",
        "salary",
        "payroll",
        "recruitment",
        "resignation",
        "benefit",
        "insurance",
        "human resources",
        "annual leave",
        "sick leave",
        "probation",
        "work from home",
        "remote work",
        "working hours",
        "overtime",
        "performance review",
        "maternity leave",
        "paternity leave",
        "promotion",
        "notice period",
        "employee benefits",
        "full-time",
        "part-time",
        "contract employee",
        "annual entitlement",
        "leave balance",
        "carryover",
        "manager approval",
        "employment policy",
        # Arabic keywords
        "إجازة",
        "اجازة",
        "الإجازة",
        "الاجازة",
        "إجازات",
        "اجازات",
        "إجازة سنوية",
        "إجازة مرضية",
        "عطلة",
        "عطلات",
        "موظف",
        "موظفة",
        "موظفين",
        "الموظفين",
        "الموظفات",
        "الموارد البشرية",
        "شؤون العاملين",
        "الحضور",
        "الغياب",
        "الانصراف",
        "التأخير",
        "الراتب",
        "راتب",
        "المرتبات",
        "الرواتب",
        "الأجور",
        "التوظيف",
        "الاستقالة",
        "استقالة",
        "التأمين",
        "التأمينات",
        "المزايا",
        "عقد العمل",
        "عقود العمل",
        "الأداء الوظيفي",
        "تقييم الأداء",
        "ساعات العمل",
        "إجازة أمومة",
        "إجازة أبوة",
        "الترقية",
        "الترقيات",
        "الجزاءات",
        "فترة الاختبار",
        "العمل من المنزل",
        "العمل عن بعد",
        "ساعات إضافية",
        "تقييم الموظفين",
        "دوام كامل",
        "دوام جزئي",
        "رصيد الإجازات",
        "ترحيل الإجازات",
        "موافقة المدير",
        "سياسة الإجازات",
    },
    "it": {
        # English keywords
        "computer",
        "laptop",
        "software",
        "hardware",
        "network",
        "internet",
        "password",
        "login",
        "email",
        "printer",
        "system",
        "server",
        "bug",
        "access",
        "vpn",
        "permissions",
        "technical support",
        "troubleshooting",
        "authenticator",
        "authentication",
        "two-factor authentication",
        "multi-factor authentication",
        "2fa",
        "mfa",
        "sla",
        "service level agreement",
        "cybersecurity",
        "firewall",
        "database",
        "backup",
        "installation",
        "update",
        "outage",
        "connectivity",
        "operating system",
        "account locked",
        "reset password",
        "access denied",
        "security token",
        "software installation",
        "network connection",
        "account security",
        "technical issue",
        "system error",
        # Arabic keywords
        "كمبيوتر",
        "حاسوب",
        "لابتوب",
        "لاب توب",
        "حاسب آلي",
        "البرمجيات",
        "البرامج",
        "الأجهزة",
        "جهاز",
        "الشبكة",
        "شبكة",
        "الشبكات",
        "الإنترنت",
        "انترنت",
        "باسورد",
        "كلمة المرور",
        "كلمة السر",
        "تسجيل الدخول",
        "البريد الإلكتروني",
        "البريد الالكتروني",
        "إيميل",
        "ايميل",
        "الطابعة",
        "طابعة",
        "النظام",
        "الأنظمة",
        "سيستم",
        "الخادم",
        "الخوادم",
        "السيرفر",
        "عطل",
        "أعطال",
        "مشكلة تقنية",
        "مشكلات تقنية",
        "صلاحيات",
        "الأذونات",
        "الدعم الفني",
        "دعم فني",
        "استكشاف الأخطاء",
        "حل المشكلات التقنية",
        "الاتصال بالشبكة",
        "الاتصال بالإنترنت",
        "تحديث النظام",
        "تثبيت البرامج",
        "أمن المعلومات",
        "الأمن السيبراني",
        "النسخ الاحتياطي",
        "المصادقة الثنائية",
        "التحقق بخطوتين",
        "المصادقة متعددة العوامل",
        "تطبيق المصادقة",
        "توثيق الحساب",
        "اتفاقية مستوى الخدمة",
        "قاعدة البيانات",
        "قواعد البيانات",
        "جدار الحماية",
        "إعادة تعيين كلمة المرور",
        "الحساب مغلق",
        "رفض الوصول",
        "انقطاع الشبكة",
        "رمز التحقق",
        "مشكلة في النظام",
        "خطأ في النظام",
        "أمان الحساب",
    },
    "finance": {
        # English keywords
        "invoice",
        "invoices",
        "reimbursement",
        "expense",
        "expenses",
        "budget",
        "payment",
        "refund",
        "purchase",
        "purchases",
        "cost",
        "financial",
        "finance",
        "tax",
        "receipt",
        "accounting",
        "billing",
        "transaction",
        "transactions",
        "accounts payable",
        "accounts receivable",
        "petty cash",
        "purchase order",
        "travel expense",
        "travel expenses",
        "per diem",
        "hotel",
        "taxi",
        "client dinner",
        "customer dinner",
        "business dinner",
        "client meal",
        "meal allowance",
        "accommodation",
        "transportation expense",
        "travel allowance",
        "expense claim",
        "expense report",
        "vendor payment",
        "financial statement",
        "payment request",
        "currency",
        "exchange rate",
        "tax deduction",
        "refund request",
        "business trip expenses",
        "travel reimbursement",
        "expense policy",
        "reimbursement policy",
        # Arabic keywords
        "فاتورة",
        "الفاتورة",
        "فواتير",
        "الفواتير",
        "استرداد المصروفات",
        "تعويض المصروفات",
        "المصروفات",
        "مصروفات",
        "المصروفات المالية",
        "ميزانية",
        "الميزانية",
        "الميزانيات",
        "الدفع",
        "المدفوعات",
        "دفعة",
        "سداد",
        "استرداد",
        "استرجاع الأموال",
        "المشتريات",
        "شراء",
        "التكلفة",
        "التكاليف",
        "مالي",
        "مالية",
        "الشؤون المالية",
        "الضرائب",
        "ضريبة",
        "الإيصال",
        "إيصال",
        "الإيصالات",
        "الحسابات",
        "المحاسبة",
        "الفوترة",
        "معاملة مالية",
        "تحويل مالي",
        "المستحقات",
        "السلف",
        "العهد المالية",
        "المطالبات المالية",
        "الخصومات",
        "الإيرادات",
        "الأرباح",
        "الخسائر",
        "تسوية مالية",
        "السداد",
        "عشاء العملاء",
        "عشاء العميل",
        "وجبة عمل",
        "وجبات العمل",
        "الفندق",
        "فندق",
        "حجز فندق",
        "الإقامة",
        "سيارة أجرة",
        "تاكسي",
        "مصاريف السفر",
        "تكاليف السفر",
        "بدل يومي",
        "بدل الانتقال",
        "بدل السفر",
        "طلب تعويض",
        "طلب استرداد",
        "أمر شراء",
        "مستحقات الموردين",
        "سعر الصرف",
        "خصم ضريبي",
        "رحلة عمل",
        "رحلات العمل",
        "سياسة المصروفات",
        "سياسة التعويضات",
        "مصاريف الإقامة",
        "مصاريف الانتقال",
    },
}


# العبارات المحددة تحصل على وزن أكبر من الكلمات العامة.
PHRASE_WEIGHT = 3
SINGLE_WORD_WEIGHT = 1


def _normalize(text: str) -> str:
    """توحيد النص لتقليل اختلافات الكتابة قبل المطابقة."""
    text = str(text or "").lower()

    # إزالة التشكيل والتطويل.
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = text.replace("ـ", "")

    # توحيد بعض أشكال الحروف العربية.
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")

    # توحيد علامات الترقيم والشرطات.
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _token_variants(token: str) -> set[str]:
    """إنشاء صيغ بديلة للكلمة العربية عند وجود حروف ملتصقة."""
    variants = {token}

    # عدم تطبيق قواعد الحروف العربية على الكلمات الإنجليزية.
    if not re.search(r"[\u0600-\u06FF]", token):
        return variants

    # إزالة حرف عطف أو جر ملتصق بالكلمة.
    prefixes = {"و", "ف", "ب", "ك", "ل"}

    for prefix in prefixes:
        if token.startswith(prefix) and len(token) > 2:
            variants.add(token[1:])

    # التعامل مع حرف ملتصق يليه التعريف.
    for variant in list(variants):  # noqa: PERF101 - copy: the set grows inside the loop
        if variant.startswith("ال") and len(variant) > 3:
            variants.add(variant[2:])

        if len(variant) > 3 and variant[0] in prefixes and variant[1:3] == "ال":
            variants.add(variant[3:])

    return variants


def _phrase_matches(
    question_tokens: list[str],
    keyword_tokens: list[str],
) -> bool:
    """التحقق من وجود كلمات العبارة المفتاحية بالترتيب."""
    if not keyword_tokens or len(keyword_tokens) > len(question_tokens):
        return False

    for start in range(len(question_tokens) - len(keyword_tokens) + 1):
        matched = True

        for offset, keyword_token in enumerate(keyword_tokens):
            question_token = question_tokens[start + offset]

            if not (_token_variants(question_token) & _token_variants(keyword_token)):
                matched = False
                break

        if matched:
            return True

    return False


def _score_text(text: str) -> dict[str, int]:
    """حساب درجة التطابق لكل مجال باستخدام الكلمات والعبارات."""
    normalized_text = _normalize(text)
    question_tokens = normalized_text.split()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            keyword_tokens = _normalize(keyword).split()

            if not keyword_tokens:
                continue

            if _phrase_matches(question_tokens, keyword_tokens):
                if len(keyword_tokens) > 1:
                    scores[domain] += PHRASE_WEIGHT
                else:
                    scores[domain] += SINGLE_WORD_WEIGHT

    return scores


def route(
    question: str,
    history: list[dict] | None = None,
) -> tuple[str, float]:
    """إرجاع المجال الأنسب ودرجة الثقة في التصنيف."""
    if not _normalize(question):
        return "hr", 0.0

    scores = _score_text(question)

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_domain, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score > 0:
        margin = best_score - second_score

        confidence = min(
            0.99,
            0.60 + 0.05 * best_score + 0.10 * margin,
        )

        confidence = max(0.0, min(1.0, confidence))

        return best_domain, confidence

        # إبقاء الثقة ضمن الحدود المسموح بها في إعدادات المشروع.
        confidence = max(0.0, min(1.0, confidence))

        return best_domain, confidence

    # استخدام آخر سؤال للمستخدم عند عدم وضوح السؤال الحالي.
    if history:
        for message in reversed(history):
            if not isinstance(message, dict):
                continue

            if message.get("role") != "user":
                continue

            previous_text = message.get("content", "")
            previous_scores = _score_text(previous_text)

            previous_domain, previous_score = max(
                previous_scores.items(),
                key=lambda item: item[1],
            )

            if previous_score > 0:
                return previous_domain, 0.65

            break

    # لا توجد كلمات كافية لتحديد المجال.
    return "hr", 0.0
