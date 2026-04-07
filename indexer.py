from __future__ import annotations

import os
import sqlite3
import numpy as np
import Stemmer

from models import Chunk

_stemmer = Stemmer.Stemmer("russian")


def stem_russian(text: str) -> str:
    words = text.lower().split()
    stemmed = _stemmer.stemWords(words)
    return " ".join(stemmed)


_KNOWN_AUTHORS = {
    "tolstoy_aelita": "Алексей Толстой — Аэлита",
    "tsiolkovsky_grezy": "К.Э. Циолковский — Грёзы о Земле и небе",
    "tsiolkovsky_prichina": "К.Э. Циолковский — Причина Космоса",
    "tsiolkovsky_puteshestvennik": "К.Э. Циолковский — Путешественник в мировые пространства",
    "tsiolkovsky_vne_zemli": "К.Э. Циолковский — Вне Земли",
    "verne_20000": "Жюль Верн — Двадцать тысяч лье под водой",
    "verne_2889": "Жюль Верн — В 2889 году",
    "verne_s_zemli_na_lunu": "Жюль Верн — С Земли на Луну",
    "verne_vokrug_luny": "Жюль Верн — Вокруг Луны",
    "verne_vverh_dnom": "Жюль Верн — Вверх дном",
    "wells_pervye_lyudi_na_lune": "Герберт Уэллс — Первые люди на Луне",
    "wells_mashina_vremeni": "Герберт Уэллс — Машина времени",
    "wells_spyaschiy": "Герберт Уэллс — Когда Спящий проснётся",
    "wells_voyna_mirov": "Герберт Уэллс — Война миров",
    "zamyatin_my": "Евгений Замятин — Мы",
    "odoevsky_4338": "В.Ф. Одоевский — 4338-й год",
    "odoevsky_gorod": "В.Ф. Одоевский — Город без имени",
    "bogdanov_krasnaya_zvezda": "А.А. Богданов — Красная звезда",
    "belyaev_zvezda_kec": "А.Р. Беляев — Звезда КЭЦ",
    "slughi-milosierdiia-enn-lieki": "Энн Леки — Слуги милосердия",
    "slughi-pravosudiia-enn-lieki": "Энн Леки — Слуги правосудия",
    "po-tu-storonu-snov-pitier-f.-gamilton": "Питер Ф. Гамильтон — По ту сторону снов",
    "tiemporalnaia-biezdna-pitier-f.-gamilton": "Питер Ф. Гамильтон — Темпоральная бездна",
    "zviezda-pandory-pitier-gamilton": "Питер Гамильтон — Звезда Пандоры",
    "dikhronavty-si-griegh-ighan": "Грег Иган — Дихронавты",
    "liestnitsa_shilda_roman_-_griegh_ighan": "Грег Иган — Лестница Шильда",
    "striely-vriemieni-lp-griegh-ighan": "Грег Иган — Стрелы времени",
    "zavodnaya-raketa-greg-igan": "Грег Иган — Заводная ракета",
    "shiest_izmierienii_prostranstva_-_alastier_rieinolds": "Аластер Рейнольдс — Шесть измерений пространства",
    "zviezdnyi_lied_-_alastier_rieinolds": "Аластер Рейнольдс — Звёздный лёд",
    "fazy_ghravitatsii_-_den_simmons": "Дэн Симмонс — Фазы гравитации",
    "ziemliu-kientavram-si-ievghienii-volkov": "Евгений Волков — Землю кентаврам",
    "odin-dien-na-olimpii-si-aliona-koshchieieva": "Алёна Кощеева — Один день на Олимпии",
    "kosmichieskii_splin_-_alieksandr_chichulin": "Александр Чичулин — Космический сплин",
    "adaptatsiia-si-dik-drammier": "Дик Драммер — Адаптация",
    "novaia-era_-anomaliia-si-vadim-kushnir": "Вадим Кушнир — Новая эра: Аномалия",
    "oskolki-vospominanii-si-marsifam": "Марсифам — Осколки воспоминаний",
    "prishieliets-i-zakon-lp-robiert-soiier": "Роберт Сойер — Пришелец и закон",
    "prostoliudin-si-alieksandr-gromov": "Александр Громов — Простолюдин",
    "tarzanarium-arkhimieda-si-alieksiei-katsai": "Алексей Кацай — Тарзанариум Архимеда",
    "troinoi_pryzhok_si_-_filipp_li": "Филипп Ли — Тройной прыжок",
    "ziemlianin-5.0-si-vladislav-alief": "Владислав Алиев — Землянин 5.0",
    "Nebesnoe_Voinstvo__№01_Sed'moi": "Сергей Лукьяненко — Небесное воинство: Седьмой",
    "Nebesnoe_Voinstvo__№02_Devjatii": "Сергей Лукьяненко — Небесное воинство: Девятый",
    "Tjajelaja_Real'nost'__№01_Igri_Blagorodnix": "Тяжёлая реальность: Игры благородных",
    "Tjajelaja_Real'nost'__№02_Kletka": "Тяжёлая реальность: Клетка",
    "Tjajelaja_Real'nost'__№03_Nestandartnoe_Mishlenie": "Тяжёлая реальность: Нестандартное мышление",
    "sobranie-sochineniy-tom-5-19671968": "Аркадий и Борис Стругацкие — Собрание сочинений, том 5",
    # --- optimist: научные и PDF ---
    "032-038": "Костылева В.Р. и др. — Новая эра российского космостроения",
    "13-23": "Новая эра в освоении Вселенной (советская космонавтика)",
    "202-493all-1": "Резунков Ю.А. — Лазерные технологии для космоса",
    "2023-FM-Session-5-NASA-Deep-Space-Logistics-Sebastian-I-4": "NASA — Artemis Campaign: Deep Space Logistics",
    "250501": "Минэкономразвития — Развитие космической отрасли",
    "33-41_korablev_no1": "Кораблёв О.И. — Таинственный Марс: атмосфера как зеркало жизни",
    "3fa153e9efbb42aa7c48fbfa269e9b2d": "НПО им. С.А. Лавочкина — Вестник, 2017",
    "55-64_zubko_no1": "Зубко — Что такое Облако Оорта? (Земля и Вселенная, 2025)",
    "9789264264014-en": "OECD — Space and Innovation, 2016",
    "IntroductionToDynamics": "Овчинников М.Ю. — Введение в динамику космического полёта",
    "Kosmos_SSAU_marketrepot_2022": "Самарский университет — Рынок коммерческого космоса, 2022",
    "L-0003930916-pdf": "Macdonald, Badescu — The International Handbook of Space Technology",
    "Ponomarev_A_K_Romanov_A_A_Tyulin_A_E_8acdc73f0a": "Пономарев А.К. и др. — Фотонные технологии в космическом приборостроении",
    "Publication_Final_English_June2021": "UN OOSA — Guidelines for Long-term Sustainability of Outer Space",
    "R_Kosmos_D-2019 (1)": "ИКИ РАН — Отчёт по теме «Космос-Д», 2019",
    "Space_Weather_Forecasting_Guide_latest": "Rodriguez L. et al. — Space Weather Forecasting Guide",
    "Vselennaya_chitat": "Альпина нон-фикшн — Краткий путеводитель по Вселенной, 2018",
    "WEF_Clear_Orbit_Secure_Future_2026": "World Economic Forum — Clear Orbit, Secure Future (Space Debris), 2026",
    "_- Космологические модели": "Нагирнер Д.И. — Космологические модели (СПбГУ)",
    "dtlstict2021d1_en": "UNCTAD — Exploring Space Technologies for Sustainable Development, 2021",
    "ecn162020d3_ru": "ООН — Космические технологии в целях устойчивого развития, 2020",
    "getPDF": "Лацинник А.А. — Космос и COTS-технология (Технополис ЭРА, 2023)",
    "ijtsrd70468": "Sadiku M. et al. — Emerging Technologies in Space Exploration, 2024",
    "klyuchevye-texnologii-budushhego-puteshestvie-v-xxv-vek": "Никитин В. — Ключевые технологии будущего: путешествие в XXV век",
    "kosmicheskoe-buduschee-uvidet-i-voplotit": "Бурцева Н.Л. — Космическое будущее: увидеть и воплотить",
    "modern-development-of-general-relativity-for-astronomers-M": "Алексеев С.О. — Современное развитие ОТО для астрономов (МГУ)",
    "mseR": "Послания исследователей космоса будущим поколениям",
    "observational-basis-of-cosmology-M-2": "Сажина О.С. — Наблюдательные основы космологии (МГУ)",
    "our-common-agenda-policy-brief-outer-space-ru": "ООН — Будущность управления космической деятельностью, 2023",
    "paper6": "Катькалов В.Б., Морозова М.Л. — Обслуживаемый космос: достижения и перспективы",
    "perspektivnye_napravleniya_i_konkursy_iki_shrink": "ИКИ РАН — Перспективные направления ракетно-космической отрасли",
    "prvselennaya": "В мире науки — Космология: Первая микросекунда Вселенной, 2009",
    "st_space-088R": "ООН УВКП — Повестка дня «Космос-2030»",
    "wst0tshgk9w4ppfjcxtjf0z0wjnkkp9v": "i.moscow — Космос не ждёт: развитие частного космоса",
    "zov": "Шапошникова Л.В. — Философия космической реальности",
    "Приложение_космос": "Рекомендации сессии: космические технологии и ЦУР стран ШОС",
    # --- pessimist: научные и PDF ---
    "01-Bolshakov_Kuznetsov": "Большаков Б.Е., Кузнецов О.Л. — Устойчивое развитие и космическое будущее",
    "013__7___________": "Философские проблемы астрономии и космологии (лекция)",
    "1": "Чечельницкий А.М. — Волновая Вселенная и жизнь: этнос и космос",
    "108-111": "Савлукова Е.В. — Космологические модели Вселенной: философские основания",
    "12-38_mitrofanov_no1": "Митрофанов И.Г. — Исследования Марса: вчера, сегодня и завтра",
    "16985": "Национальная политика США в области космической деятельности",
    "2019romanov": "Романов А.А. — Основы космических информационных систем (ИКИ РАН)",
    "2021_03_03": "Кречет В.Г. и др. — Космологические модели без сингулярности",
    "569-1240-1-SM": "Стефанович Д.В., Ермаков А.С. — «Новый космос» двойного назначения: опыт США",
    "R_Kosmos_D-2019": "ИКИ РАН — Отчёт по теме «Космос-Д», 2019",
    "Space_Environment_Report_latest": "ESA Space Debris Office — Annual Space Environment Report",
    "VESELOV_2016_4": "Веселов — 2016",
    "_2169": "Научная статья о космосе",
    "angel": "Ангел (рассказ)",
    "d80c7490a9a30ee5bfa9703f4f4c0c4d": "Научная статья о космосе",
    "j5isc9l0hm9ftq34g6jam8mey26eb5w1": "Научная статья о космосе",
    "platforma-kosmos-prezentaciya-final-the-end": "Платформа «Космос» — презентация",
    "sbornik-aviacziya-i-kosmos-2023": "Сборник «Авиация и космос», 2023",
    "skvoz-vremya-sbornik": "Сквозь время (сборник фантастики)",
    "sp1249web": "ESA — Space Report SP-1249",
    "space-technologies-and-climate-change": "Space Technologies and Climate Change",
    "temnmir": "Тёмная материя (научно-популярный текст)",
    "tgb_text_03_23-030-034": "Научная статья о космосе, 2023",
    "v1": "Научная статья о космосе",
    "vliyanie-kosmosa-na-suschestvovanie-chelovechestva": "Влияние космоса на существование человечества",
    "итог_2017_Альманах": "Альманах космических исследований, 2017",
}


def _extract_author(text: str, filename: str) -> str:
    """Try to extract 'Author — Title' from the first ~500 chars of a file."""
    # Check known authors first
    key = os.path.splitext(filename)[0]
    if key in _KNOWN_AUTHORS:
        return _KNOWN_AUTHORS[key]

    head = text[:500]
    lines = [l.strip() for l in head.split("\n") if l.strip()]

    # Pattern 1: fiction files typically have Title on line 1, Author on line 2
    # e.g. "Аврора\nКим Стэнли Робинсон"
    # or "Эксцессия\nИэн М. Бэнкс\nКультура #3"
    if len(lines) >= 2:
        l1, l2 = lines[0], lines[1]
        # Check if line2 looks like an author name (2-4 capitalized words, no digits/special)
        import re as _re
        ru_name = _re.match(r'^[А-ЯЁA-Z][а-яёa-z]+\.?\s+[А-ЯЁA-Z][\w\.\s-]{1,40}$', l2)
        if ru_name and len(l1) < 80 and not _re.search(r'\d{3,}|УДК|DOI|ISSN|http', l1):
            return f"{l2} — {l1}"

    # Pattern 2: scientific — look for "Фамилия И.О." pattern anywhere in first 500 chars
    import re as _re
    sci_author = _re.search(
        r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.(?:\s*,\s*[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)*)',
        head
    )
    if sci_author:
        # Find a title-like line nearby
        title_match = _re.search(r'([А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z\s:–\-]{10,70})', head)
        title = title_match.group(1).strip() if title_match else ""
        author = sci_author.group(1).strip()
        return f"{author}. {title}" if title else author

    # Pattern 3: "Annotation" prefix (fb2 converted) — use filename
    fn_clean = os.path.splitext(filename)[0]
    fn_clean = fn_clean.replace("-", " ").replace("_", " ").replace(".", " ")
    if len(fn_clean) > 5:
        return fn_clean

    return ""


def load_texts(directory: str) -> list[dict]:
    results = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            author = _extract_author(text, filename)
            results.append({"text": text, "source_file": filename, "author": author})
    return results


def chunk_texts(
    texts: list[dict], chunk_size: int = 1000, overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 0
    for item in texts:
        raw = item["text"]
        source = item["source_file"]
        author = item.get("author", "")
        if len(raw) <= chunk_size:
            chunks.append(Chunk(id=chunk_id, text=raw, source_file=source, collection="", author=author))
            chunk_id += 1
            continue
        start = 0
        while start < len(raw):
            end = start + chunk_size
            chunk_text = raw[start:end]
            chunks.append(Chunk(id=chunk_id, text=chunk_text, source_file=source, collection="", author=author))
            chunk_id += 1
            start += chunk_size - overlap
    return chunks


def build_fts_index(chunks: list[Chunk]) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute(
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(stemmed_text, tokenize='unicode61')"
    )
    for chunk in chunks:
        stemmed = stem_russian(chunk.text)
        db.execute("INSERT INTO chunks_fts(rowid, stemmed_text) VALUES (?, ?)", (chunk.id, stemmed))
    db.commit()
    return db


def build_dense_index(
    chunks: list[Chunk], model_name: str = "intfloat/multilingual-e5-small"
) -> tuple[np.ndarray, object]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [f"passage: {chunk.text}" for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32), model


def _cache_path(directory: str, suffix: str) -> str:
    return os.path.join(directory, f"_cache_{suffix}")


class IndexedCollection:
    def __init__(
        self,
        name: str,
        chunks: list[Chunk],
        fts_db: sqlite3.Connection,
        embeddings: np.ndarray,
        embed_model: object,
    ):
        self.name = name
        self.chunks = chunks
        self.fts_db = fts_db
        self.embeddings = embeddings
        self.embed_model = embed_model

    def save_cache(self, directory: str):
        """Save pre-built embeddings and chunks to disk for fast startup."""
        import json
        np.save(_cache_path(directory, "embeddings.npy"), self.embeddings)
        chunks_data = [{"id": c.id, "text": c.text, "source_file": c.source_file, "collection": c.collection, "author": c.author} for c in self.chunks]
        with open(_cache_path(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

    @classmethod
    def load_cache(cls, name: str, directory: str, model_name: str = "intfloat/multilingual-e5-small") -> IndexedCollection | None:
        """Load pre-built index from cache. Returns None if cache is missing or stale."""
        import json
        emb_path = _cache_path(directory, "embeddings.npy")
        chunks_path = _cache_path(directory, "chunks.json")
        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            return None

        # Check if any .txt file is newer than cache
        cache_mtime = os.path.getmtime(emb_path)
        for f in os.listdir(directory):
            if f.endswith(".txt") and os.path.getmtime(os.path.join(directory, f)) > cache_mtime:
                return None  # stale cache

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [Chunk(**d) for d in chunks_data]
        embeddings = np.load(emb_path)
        fts_db = build_fts_index(chunks)

        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(model_name)

        return cls(name, chunks, fts_db, embeddings, embed_model)

    @classmethod
    def build(
        cls,
        name: str,
        directory: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str = "intfloat/multilingual-e5-small",
    ) -> IndexedCollection:
        # Try loading from cache first
        cached = cls.load_cache(name, directory, model_name)
        if cached is not None:
            return cached

        texts = load_texts(directory)
        chunks = chunk_texts(texts, chunk_size, chunk_overlap)
        for chunk in chunks:
            chunk.collection = name
        fts_db = build_fts_index(chunks)
        embeddings, embed_model = build_dense_index(chunks, model_name)
        col = cls(name, chunks, fts_db, embeddings, embed_model)
        col.save_cache(directory)
        return col
