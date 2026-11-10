"""
Explainable recommendations and voice scripts, in English and Tamil.

How it works (and what it is not): CARS is a transparent weighted score, not a
trained model. This module turns the *measured* feature values of one gap into
evidence statements with fixed rules - e.g. "vegetation-like pixels cover 64 %
of the gap" is only said when that fraction was actually measured. Nothing is
inferred that the features do not contain: the image cues are colour/texture
heuristics, so statements say "vegetation-like" or "shadow-like", never "a tree"
or "a building".

Every statement is produced in both languages from the same values, so the
on-screen text and the spoken guidance always state the same facts.
"""

from __future__ import annotations

from typing import Any

from config import FEATURES, PipelineConfig

FEATURE_NAMES = {
    "en": {"A": "road alignment", "O": "occlusion evidence", "W": "width consistency",
           "G": "local geometry", "N": "network connectivity", "E": "surrounding environment", "F": "road fragments"},
    "ta": {"A": "சாலைத் திசை ஒத்திசைவு", "O": "மறைப்பு ஆதாரம்", "W": "அகல ஒத்திசைவு",
           "G": "உள்ளூர் வடிவியல்", "N": "வலையமைப்பு இணைப்பு", "E": "சுற்றுச்சூழல்", "F": "சாலைத் துண்டுகள்"},
}
COVER_NAMES = {
    "en": {"road_like": "road-like surface", "vegetation": "vegetation-like", "shadow": "shadow-like",
           "textured": "textured", "water": "water-like", "unexplained": "other cover"},
    "ta": {"road_like": "சாலை போன்ற பரப்பு", "vegetation": "பச்சைத் தாவரம் போன்றது", "shadow": "நிழல் போன்றது",
           "textured": "அமைப்புடைய பரப்பு", "water": "நீர் போன்றது", "unexplained": "பிற பரப்பு"},
}
TERM_NAMES = {
    "en": {"length": "length", "non_road": "non-road appearance", "guide": "distance from the expected course",
           "direction": "turning", "other_road": "other roads"},
    "ta": {"length": "நீளம்", "non_road": "சாலையல்லாத தோற்றம்", "guide": "எதிர்பார்க்கப்படும் பாதையிலிருந்து விலகல்",
           "direction": "திருப்பம்", "other_road": "பிற சாலைகள்"},
}
HEADLINES = {
    "recommended": {"en": "Connection recommended", "ta": "இணைப்பு பரிந்துரைக்கப்படுகிறது"},
    "uncertain": {"en": "Your judgement is needed", "ta": "உங்கள் முடிவு தேவை"},
    "not_recommended": {"en": "Connection not recommended", "ta": "இணைப்பு பரிந்துரைக்கப்படவில்லை"},
}


def _n(v: Any, d: int = 0) -> str:
    if v is None:
        return "?"
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v)


def _pct(v: float | None) -> str:
    return "?" if v is None else f"{round(100 * v)}"


def _item(feature: str, polarity: str, en: str, ta: str) -> dict[str, str]:
    return {"feature": feature, "polarity": polarity, "en": en, "ta": ta}


# --------------------------------------------------------------------------- #
# Evidence for one gap
# --------------------------------------------------------------------------- #
def gap_evidence(c) -> list[dict[str, str]]:
    f, s = c.features, c.scores
    items: list[dict[str, str]] = []

    # F - the two road fragments (image-based verification) ------------------
    fr = getattr(c, "fragments_info", None) or {}
    if fr.get("A") and fr.get("B"):
        def desc(k, lang):
            x = fr[k]
            lk = "n/a" if x.get("road_likelihood") is None else f"{x['road_likelihood']:.2f}"
            if lang == "en":
                return f"fragment {k} (F{x['component']}) is {_n(x.get('length_px'))} px long and {_n(x.get('width_px'))} px wide, road-colour likelihood {lk}"
            return f"துண்டு {k} (F{x['component']}): நீளம் {_n(x.get('length_px'))} பிக்சல், அகலம் {_n(x.get('width_px'))} பிக்சல், சாலை நிற ஒற்றுமை {lk}"
        verified = fr["A"].get("verified") and fr["B"].get("verified")
        if verified:
            items.append(_item("F", "supports", f"Both sides are identifiable road structures in the image: {desc('A', 'en')}; {desc('B', 'en')}.",
                               f"இரண்டு பக்கங்களும் படத்தில் அடையாளம் காணக்கூடிய சாலைகள்: {desc('A', 'ta')}; {desc('B', 'ta')}."))
        else:
            items.append(_item("F", "uncertain", f"The fragments could not be checked against the image: {desc('A', 'en')}; {desc('B', 'en')}.",
                               f"சாலைத் துண்டுகளைப் படத்துடன் சரிபார்க்க முடியவில்லை: {desc('A', 'ta')}; {desc('B', 'ta')}."))
    if getattr(c, "priority", "") == "occlusion":
        items.append(_item("F", "context", "The gap lies in a possible occlusion region, so it is reviewed early.",
                           "இந்த இடைவெளி மறைப்பு இருக்கக்கூடிய பகுதியில் உள்ளது; எனவே இது முன்னதாக ஆய்வு செய்யப்படுகிறது."))

    # A - alignment ---------------------------------------------------------
    a = s.get("A")
    dev = f.get("end_deviation_deg")
    if a is None:
        items.append(_item("A", "uncertain", "The direction of the road ends could not be measured.",
                           "சாலை முனைகளின் திசையை அளவிட முடியவில்லை."))
    elif c.kind == "endpoint_to_road":
        meet = f.get("meeting_angle_deg")
        if a >= 0.75:
            items.append(_item("A", "supports",
                               f"The road end points at the side of another road and would meet it at {_n(meet)}°, like an ordinary junction.",
                               f"இந்தச் சாலை முனை மற்றொரு சாலையின் பக்கத்தை நோக்கி உள்ளது; இணைந்தால் {_n(meet)}° கோணத்தில் சந்திக்கும் — இது சாதாரண சந்திப்பு போன்றது."))
        else:
            items.append(_item("A", "against",
                               f"The road end does not point squarely at the other road (it deviates {_n(dev)}°, meeting angle {_n(meet)}°).",
                               f"இந்தச் சாலை முனை மற்றொரு சாலையைச் சரியாக நோக்கவில்லை (விலகல் {_n(dev)}°, சந்திப்புக் கோணம் {_n(meet)}°)."))
    else:
        bend, turn = f.get("bend_inconsistency_deg"), f.get("turn_angle_deg")
        if a >= 0.75:
            if turn is not None and turn >= 15 and (bend or 0) <= 15:
                items.append(_item("A", "supports",
                                   f"The two road ends follow one smooth curve: together they turn {_n(turn)}°, with little mismatch ({_n(bend)}°).",
                                   f"இரண்டு சாலை முனைகளும் ஒரே மென்மையான வளைவைப் பின்பற்றுகின்றன: மொத்தம் {_n(turn)}° திரும்புகின்றன; பொருந்தாமை {_n(bend)}° மட்டுமே."))
            else:
                items.append(_item("A", "supports",
                                   f"The two road ends point toward each other, only {_n(dev)}° off the gap line, as one continuous road would.",
                                   f"இரண்டு சாலை முனைகளும் ஒன்றையொன்று நோக்கி உள்ளன; இடைவெளிக் கோட்டிலிருந்து {_n(dev)}° மட்டுமே விலகல் — ஒரே தொடர்ச்சியான சாலை போல."))
        elif a >= 0.5:
            items.append(_item("A", "neutral",
                               f"The road ends are only partly aligned ({_n(dev)}° off the gap line, bend mismatch {_n(bend)}°).",
                               f"சாலை முனைகள் ஓரளவு மட்டுமே ஒத்துப்போகின்றன (இடைவெளிக் கோட்டிலிருந்து {_n(dev)}° விலகல், வளைவுப் பொருந்தாமை {_n(bend)}°)."))
        else:
            items.append(_item("A", "against",
                               f"The road ends have inconsistent orientations: they deviate {_n(dev)}° from the gap line and would need an S-shaped bend of {_n(bend)}° to meet.",
                               f"சாலை முனைகளின் திசைகள் ஒத்துப்போகவில்லை: இடைவெளிக் கோட்டிலிருந்து {_n(dev)}° விலகியுள்ளன; இணைக்க {_n(bend)}° அளவிலான S-வடிவ வளைவு தேவைப்படும்."))

    # W - width --------------------------------------------------------------
    w = s.get("W")
    ws, wt = f.get("width_source"), f.get("width_target")
    if w is None:
        items.append(_item("W", "uncertain", "The road width could not be measured.", "சாலை அகலத்தை அளவிட முடியவில்லை."))
    elif c.kind == "endpoint_to_road":
        pol = "supports" if w >= 0.5 else "against"
        items.append(_item("W", pol,
                           f"The side road is {_n(ws)} px wide and the road it would join is {_n(wt)} px wide.",
                           f"பக்கச் சாலையின் அகலம் {_n(ws)} பிக்சல்; அது இணையும் சாலையின் அகலம் {_n(wt)} பிக்சல்."))
    elif w >= 0.75:
        items.append(_item("W", "supports", f"Both fragments have similar widths ({_n(ws)} and {_n(wt)} px).",
                           f"இரண்டு சாலைத் துண்டுகளும் ஏறக்குறைய ஒரே அகலம் கொண்டவை ({_n(ws)} மற்றும் {_n(wt)} பிக்சல்)."))
    elif w >= 0.4:
        items.append(_item("W", "neutral", f"The widths differ somewhat ({_n(ws)} and {_n(wt)} px).",
                           f"அகலங்களில் சிறிது வேறுபாடு உள்ளது ({_n(ws)} மற்றும் {_n(wt)} பிக்சல்)."))
    else:
        items.append(_item("W", "against",
                           f"The widths differ strongly ({_n(ws)} and {_n(wt)} px), which suggests two different kinds of road.",
                           f"அகலங்கள் பெரிதும் வேறுபடுகின்றன ({_n(ws)} மற்றும் {_n(wt)} பிக்சல்); இவை வேறு வகைச் சாலைகளாக இருக்கலாம்."))

    # O - occlusion (inside the gap) ----------------------------------------
    o = s.get("O")
    if o is None:
        if "corridor_pixels" in f:
            items.append(_item("O", "uncertain", "The gap is too small to inspect what covers it.",
                               "இடைவெளி மிகச் சிறியது; அதை மறைப்பது எது என்று ஆய்வு செய்ய முடியவில்லை."))
        else:
            items.append(_item("O", "uncertain", "There is no image to check what covers the gap.",
                               "இடைவெளியை மறைப்பது எது என்று சரிபார்க்கப் படம் இல்லை."))
    else:
        veg, sh, rl = f.get("frac_vegetation", 0), f.get("frac_shadow", 0), f.get("frac_road_like", 0)
        tex, wat, un = f.get("frac_textured", 0), f.get("frac_water", 0), f.get("frac_unexplained", 0)
        said = False
        if veg >= 0.25:
            items.append(_item("O", "supports",
                               f"Vegetation-like (green) pixels cover {_pct(veg)}% of the gap, which is evidence that the road may continue beneath vegetation.",
                               f"இடைவெளியின் {_pct(veg)}% பகுதியில் பச்சைத் தாவரம் போன்ற பிக்சல்கள் உள்ளன; எனவே சாலை தாவரங்களின் கீழ் தொடரக்கூடும்."))
            said = True
        if sh >= 0.25:
            items.append(_item("O", "supports",
                               f"Shadow-like (dark) pixels cover {_pct(sh)}% of the gap, so the road may be hidden in shadow.",
                               f"இடைவெளியின் {_pct(sh)}% பகுதி நிழல் போல இருண்டு காணப்படுகிறது; சாலை நிழலில் மறைந்திருக்கலாம்."))
            said = True
        if rl >= 0.25:
            items.append(_item("O", "supports",
                               f"{_pct(rl)}% of the gap still looks like road surface, so the road mask probably missed visible road there.",
                               f"இடைவெளியின் {_pct(rl)}% பகுதி இன்னும் சாலை மேற்பரப்பு போலவே தெரிகிறது; road mask அங்குள்ள சாலையைத் தவறவிட்டிருக்கலாம்."))
            said = True
        if wat >= 0.2:
            items.append(_item("O", "against", f"{_pct(wat)}% of the gap looks like water.",
                               f"இடைவெளியின் {_pct(wat)}% பகுதி நீர் போலத் தெரிகிறது."))
            said = True
        if tex >= 0.25:
            items.append(_item("O", "neutral",
                               f"{_pct(tex)}% of the gap is textured; it could be a building or clutter, but this cue is not verified.",
                               f"இடைவெளியின் {_pct(tex)}% பகுதி அமைப்புடைய தோற்றம் கொண்டது; இது கட்டிடமாகவோ பிற பொருளாகவோ இருக்கலாம், ஆனால் உறுதிப்படுத்தப்படவில்லை."))
            said = True
        if un >= 0.5:
            items.append(_item("O", "against",
                               f"{_pct(un)}% of the gap shows cover that resembles neither road, vegetation nor shadow, so nothing in the image explains why the road would stop.",
                               f"இடைவெளியின் {_pct(un)}% பகுதியில் சாலை, தாவரம் அல்லது நிழல் போல இல்லாத பரப்பு உள்ளது; சாலை ஏன் நின்றுவிடுகிறது என்பதற்குப் படத்தில் எந்த ஆதாரமும் இல்லை."))
            said = True
        if not said:
            items.append(_item("O", "neutral", "The gap has mixed cover with no clear occluder.",
                               "இடைவெளியில் தெளிவான மறைப்பு எதுவும் இல்லை; பல வகைப் பரப்புகள் கலந்துள்ளன."))

    # E - surrounding environment (descriptive; part of the context shown) ----
    if "env_pixels" in f:
        parts = sorted(((f.get(f"env_{k}", 0), k) for k in COVER_NAMES["en"]), reverse=True)
        top = [(v, k) for v, k in parts if v >= 0.15][:3]
        if top:
            en = ", ".join(f"{_pct(v)}% {COVER_NAMES['en'][k]}" for v, k in top)
            ta = ", ".join(f"{_pct(v)}% {COVER_NAMES['ta'][k]}" for v, k in top)
            items.append(_item("E", "context", f"Around the gap: {en}.", f"இடைவெளியைச் சுற்றி: {ta}."))

    # G - local geometry ---------------------------------------------------------
    g = s.get("G")
    r_w = f.get("min_turn_radius_widths")
    jump = f.get("curvature_jump")
    if g is None:
        items.append(_item("G", "uncertain", "The shape of the connecting curve could not be measured.",
                           "இணைக்கும் வளைவின் வடிவத்தை அளவிட முடியவில்லை."))
    elif r_w is None:
        items.append(_item("G", "supports" if g >= 0.75 else "neutral", "A straight connection is enough; no sharp turn is needed.",
                           "நேரான இணைப்பு போதுமானது; கூர்மையான திருப்பம் தேவையில்லை."))
    elif g >= 0.75:
        items.append(_item("G", "supports",
                           f"The connecting curve turns no tighter than a radius of {_n(r_w, 1)} road widths, which a real road can do.",
                           f"இணைக்கும் வளைவின் மிகக் கூர்மையான திருப்பம் {_n(r_w, 1)} சாலை அகல ஆரம் கொண்டது; உண்மையான சாலைக்கு இது சாத்தியமே."))
    else:
        en = f"The connection would need a turn with a radius of only {_n(r_w, 1)} road widths, tighter than roads normally bend."
        ta = f"இணைக்க {_n(r_w, 1)} சாலை அகல ஆரம் மட்டுமே கொண்ட கூர்மையான திருப்பம் தேவைப்படும்; சாலைகள் பொதுவாக இவ்வளவு கூர்மையாக வளைவதில்லை."
        if jump is not None and jump > 0.25:
            en += " It would also change the fragments' curvature abruptly."
            ta += " இது சாலைத் துண்டுகளின் வளைவையும் திடீரென மாற்றும்."
        items.append(_item("G", "against", en, ta))

    # N - network -------------------------------------------------------------
    if f.get("duplicate_edge"):
        items.append(_item("N", "against", "These road ends are already joined.", "இந்தச் சாலை முனைகள் ஏற்கனவே இணைக்கப்பட்டுள்ளன."))
    elif not f.get("same_component"):
        items.append(_item("N", "supports",
                           f"It would join two separate road fragments (components {f.get('component_source')} and {f.get('component_target')}), so the connection may improve network continuity.",
                           f"இது தனித்தனியான இரண்டு சாலைத் துண்டுகளை இணைக்கும் (பகுதிகள் {f.get('component_source')} மற்றும் {f.get('component_target')}); இதனால் வலையமைப்பின் தொடர்ச்சி மேம்படலாம்."))
    else:
        items.append(_item("N", "neutral",
                           f"Both ends already belong to one connected network (the detour is {_n(f.get('detour_ratio'), 1)} times longer).",
                           f"இரண்டு முனைகளும் ஏற்கனவே ஒரே வலையமைப்பில் உள்ளன (மாற்று வழி {_n(f.get('detour_ratio'), 1)} மடங்கு நீளம்)."))
    if (f.get("crossing_pixels") or 0) > 2:
        items.append(_item("N", "against", "The connection would cross another road.",
                           "இந்த இணைப்பு மற்றொரு சாலையைக் கடந்து செல்லும்."))
    k = f.get("competing_candidates") or 0
    if k:
        items.append(_item("N", "neutral", f"{k} other candidate(s) compete for the same road end.",
                           f"இதே சாலை முனைக்காக வேறு {k} இணைப்பு(கள்) போட்டியிடுகின்றன."))
    return items


def numeric_breakdown(c) -> dict[str, Any]:
    """The numerical CARS calculation, kept separate from the language explanation."""
    d = c.cars_details
    rows = []
    for k in FEATURES:
        rows.append({"feature": k, "value": c.scores.get(k), "value_used": d.get("values_used", {}).get(k),
                     "weight": d.get("weights_used", {}).get(k), "contribution": d.get("contributions", {}).get(k),
                     "missing": k in d.get("missing_filled", [])})
    formula = " + ".join(f"{(r['weight'] or 0):.2f}×{(r['value_used'] if r['value_used'] is not None else 0):.2f}"
                         for r in rows if r["weight"] is not None)
    return {"rows": rows, "formula": f"CARS = {formula} = {c.cars:.3f}", "cars": c.cars,
            "missing_policy": d.get("missing_policy")}


def conclusion(c, cfg: PipelineConfig) -> dict[str, str]:
    rec = c.recommendation
    thr = cfg.cars.acceptance_threshold
    if rec == "recommended":
        return {"en": "The evidence supports reconnecting these fragments. Approve the connection to reconstruct the road.",
                "ta": "இந்தச் சாலைத் துண்டுகளை இணைப்பதற்கு ஆதாரங்கள் உள்ளன. சாலையை மீண்டும் உருவாக்க இணைப்புக்கு ஒப்புதல் அளிக்கவும்."}
    if rec == "uncertain":
        missing = c.cars_details.get("missing_filled", [])
        if c.rec_basis == "conflict":
            return {"en": f"{c.rec_detail} competes for the same road end with a higher score. Approve at most one of them.",
                    "ta": f"இதே சாலை முனைக்காக அதிக மதிப்பெண் கொண்ட {c.rec_detail} போட்டியிடுகிறது. இரண்டில் ஒன்றை மட்டும் ஏற்கவும்."}
        if missing:
            en = ", ".join(FEATURE_NAMES["en"][m] for m in missing)
            ta = ", ".join(FEATURE_NAMES["ta"][m] for m in missing)
            return {"en": f"Some evidence could not be measured ({en}), so the decision is left to you.",
                    "ta": f"சில ஆதாரங்களை அளவிட முடியவில்லை ({ta}); எனவே முடிவு உங்களிடம் விடப்படுகிறது."}
        return {"en": f"The score is just below the threshold of {thr:.2f}, so the decision is left to you.",
                "ta": f"மதிப்பெண் வரம்பு மதிப்பான {thr:.2f}-க்குச் சற்றுக் கீழே உள்ளது; எனவே முடிவு உங்களிடம் விடப்படுகிறது."}
    if c.features.get("duplicate_edge"):
        return {"en": "These ends are already joined, so no new connection is needed.",
                "ta": "இந்த முனைகள் ஏற்கனவே இணைக்கப்பட்டுள்ளன; புதிய இணைப்பு தேவையில்லை."}
    return {"en": "There is insufficient evidence that both fragments belong to the same road. Connecting them may introduce an incorrect road connection.",
            "ta": "இரண்டு துண்டுகளும் ஒரே சாலையைச் சேர்ந்தவை என்பதற்குப் போதுமான ஆதாரம் இல்லை. இவற்றை இணைத்தால் தவறான சாலை இணைப்பு உருவாகலாம்."}


def influences(c) -> dict[str, list[str]]:
    """Features that raised or lowered CARS most, relative to an even 0.5 score."""
    d = c.cars_details
    w, v = d.get("weights_used", {}), d.get("values_used", {})
    effect = {k: w[k] * (v[k] - 0.5) for k in w}
    ups = [k for k, e in sorted(effect.items(), key=lambda x: -x[1]) if e > 0.02][:2]
    downs = [k for k, e in sorted(effect.items(), key=lambda x: x[1]) if e < -0.02][:2]
    return {"raised": ups, "lowered": downs}


def explain_gap(c, cfg: PipelineConfig) -> dict[str, Any]:
    items = gap_evidence(c)
    concl = conclusion(c, cfg)
    head = HEADLINES[c.recommendation]
    thr = cfg.cars.acceptance_threshold
    low = max(0.0, thr - cfg.cars.review_margin)
    if c.cars >= thr:
        meaning = {"en": "This supports reconnecting the road.", "ta": "இது சாலையை இணைப்பதை ஆதரிக்கிறது."}
    elif c.cars >= low:
        meaning = {"en": "This is just below the threshold.", "ta": "இது வரம்புக்குச் சற்றுக் கீழே உள்ளது."}
    else:
        meaning = {"en": "This does not support reconnecting the road.", "ta": "இது சாலையை இணைப்பதை ஆதரிக்கவில்லை."}
    inf = influences(c)
    infl = {}
    for lang in ("en", "ta"):
        up = ", ".join(FEATURE_NAMES[lang][k] for k in inf["raised"])
        down = ", ".join(FEATURE_NAMES[lang][k] for k in inf["lowered"])
        if lang == "en":
            infl[lang] = " ".join(x for x in (f"Raised most by: {up}." if up else "", f"Lowered most by: {down}." if down else "") if x)
        else:
            infl[lang] = " ".join(x for x in (f"மதிப்பெண்ணை அதிகம் உயர்த்தியவை: {up}." if up else "",
                                             f"அதிகம் குறைத்தவை: {down}." if down else "") if x)
    voice = {
        "reconstruct": {"en": f"Gap {c.id} approved. Reconstructing this road connection with the graph-based trellis method.",
                        "ta": f"இடைவெளி {c.id}-க்கு ஒப்புதல் அளிக்கப்பட்டது. வரைபட அடிப்படையிலான trellis முறையில் இந்தச் சாலை இணைப்பு மீண்டும் உருவாக்கப்படுகிறது."},
        "analysis": {"en": f"Analysing gap {c.id}. " + " ".join(i["en"] for i in items),
                     "ta": f"இடைவெளி {c.id} ஆய்வு செய்யப்படுகிறது. " + " ".join(i["ta"] for i in items)},
        "cars": {"en": f"The CARS score for gap {c.id} is {c.cars:.2f}, and the threshold is {thr:.2f}. {meaning['en']} {infl['en']} {head['en']}. {concl['en']}",
                 "ta": f"இடைவெளி {c.id}-க்கான CARS மதிப்பெண் {c.cars:.2f}; ஏற்பு வரம்பு {thr:.2f}. {meaning['ta']} {infl['ta']} {head['ta']}. {concl['ta']}"},
        "decision": ({"en": f"Inspect the original image, the enlarged gap, the road mask, the local graph and the proposed path for gap {c.id}, then approve or reject the connection.",
                      "ta": f"இடைவெளி {c.id}-க்கான அசல் படம், பெரிதாக்கப்பட்ட இடைவெளி, road mask, உள்ளூர் வரைபடம், முன்மொழியப்பட்ட பாதை ஆகியவற்றைப் பார்த்து, இணைப்பை ஏற்கவும் அல்லது நிராகரிக்கவும்."}
                     if c.recommendation != "not_recommended" else
                     {"en": f"Gap {c.id} is not recommended for connection. {concl['en']} You can reject it, or leave it pending and return later.",
                      "ta": f"இடைவெளி {c.id} இணைக்கப் பரிந்துரைக்கப்படவில்லை. {concl['ta']} இதை நிராகரிக்கலாம், அல்லது நிலுவையில் விட்டுப் பின்னர் திரும்பலாம்."}),
    }
    voice["review"] = {"en": " ".join((voice["analysis"]["en"], voice["cars"]["en"], voice["decision"]["en"])),
                       "ta": " ".join((voice["analysis"]["ta"], voice["cars"]["ta"], voice["decision"]["ta"]))}
    return {"headline": head, "points": items, "conclusion": concl, "meaning": meaning, "influences": inf,
            "influence_text": infl, "numeric": numeric_breakdown(c), "voice": voice}


# --------------------------------------------------------------------------- #
# Reconstruction explanations
# --------------------------------------------------------------------------- #
FAIL_TEXT = {
    "too_long": ("The cheapest route would be {ratio} times the gap, longer than the allowed {limit}.",
                 "மிகக் குறைந்த செலவுடைய வழி இடைவெளியின் {ratio} மடங்கு நீளமாக இருக்கும்; இது அனுமதிக்கப்பட்ட {limit}-ஐ விட அதிகம்."),
    "too_costly": ("The cheapest route costs {mean} per pixel, above the path cost threshold of {limit}.",
                   "மிகக் குறைந்த செலவுடைய வழிக்கு ஒரு பிக்சலுக்கு {mean} செலவாகும்; இது பாதைச் செலவு வரம்பான {limit}-ஐ விட அதிகம்."),
    "other_road": ("The cheapest route would run over an unrelated road.",
                   "மிகக் குறைந்த செலவுடைய வழி தொடர்பில்லாத மற்றொரு சாலையின் மேல் செல்லும்."),
    "no_path": ("No continuous route fits between the two road ends.",
                "இரண்டு சாலை முனைகளுக்கு இடையே தொடர்ச்சியான வழி எதுவும் பொருந்தவில்லை."),
    "degenerate": ("The search could not be set up for this gap.", "இந்த இடைவெளிக்குத் தேடலை அமைக்க முடியவில்லை."),
    "not_connected": ("After drawing the path, the two fragments were still not connected.",
                      "பாதையை வரைந்த பிறகும் இரண்டு துண்டுகளும் இணையவில்லை."),
    "endpoint_used": ("Road end {end} is already connected by {by}.", "சாலை முனை {end} ஏற்கனவே {by} மூலம் இணைக்கப்பட்டுள்ளது."),
    "already_joined": ("These road ends are already joined.", "இந்தச் சாலை முனைகள் ஏற்கனவே இணைக்கப்பட்டுள்ளன."),
}


def explain_reconstruction(c, outcome: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    res = outcome.get("path")
    m = res.metrics if res is not None else {}
    status = outcome["status"]
    b, a = outcome["components_before"]["mask"], outcome["components_after"]["mask"]
    start = c.xai["voice"]["reconstruct"] if c.xai else {"en": "", "ta": ""}
    if status != "reconstructed":
        code = outcome.get("reason_code", "no_path")
        en_t, ta_t = FAIL_TEXT.get(code, FAIL_TEXT["no_path"])
        params = {"ratio": _n(m.get("length_ratio"), 2), "limit": _n(cfg.path.max_length_ratio if code == "too_long"
                                                                    else cfg.path.max_mean_cost, 2),
                  "mean": _n(m.get("mean_cost"), 2), "end": outcome.get("end", "?"),
                  "by": outcome.get("blocked_by", "?")}
        reason_en, reason_ta = en_t.format(**params), ta_t.format(**params)
        result = {"en": f"Gap {c.id} could not be reconstructed. {reason_en} The road mask was not changed.",
                  "ta": f"இடைவெளி {c.id}-ஐ மீண்டும் உருவாக்க முடியவில்லை. {reason_ta} சாலை மாஸ்க் மாற்றப்படவில்லை."}
        return {"start": start, "result": result, "points": [], "succeeded": False}

    br = res.breakdown
    points = [{
        "en": (f"The graph-based trellis method placed {m['stations']} stations along the expected road course, each with "
               f"{m['positions']} lateral positions ({m['nodes']} nodes, {m['edges']} edges), and found the lowest-cost continuous "
               f"route by dynamic programming: total cost {m['total_cost']:.1f} ({m['mean_cost']:.2f} per pixel)."),
        "ta": (f"வரைபட அடிப்படையிலான trellis முறை எதிர்பார்க்கப்படும் சாலைப் பாதையில் {m['stations']} நிலையங்களை அமைத்தது; ஒவ்வொன்றிலும் "
               f"{m['positions']} பக்கவாட்டு நிலைகள் ({m['nodes']} கணுக்கள், {m['edges']} விளிம்புகள்). Dynamic programming மூலம் "
               f"மிகக் குறைந்த செலவுடைய தொடர்ச்சியான வழி கண்டறியப்பட்டது: மொத்தச் செலவு {m['total_cost']:.1f} (ஒரு பிக்சலுக்கு {m['mean_cost']:.2f}).")},
        {"en": (f"The path is {m['length']:.0f} px long ({m['length_ratio']:.2f} times the straight gap) and stays on average "
                f"{m['offset_mean']:.1f} px, at most {m['offset_max']:.1f} px, from the smooth course that continues both road directions."),
         "ta": (f"இந்தப் பாதை {m['length']:.0f} பிக்சல் நீளம் (நேரான இடைவெளியின் {m['length_ratio']:.2f} மடங்கு); இரண்டு சாலைத் திசைகளையும் "
                f"தொடரும் மென்மையான பாதையிலிருந்து சராசரியாக {m['offset_mean']:.1f} பிக்சல், அதிகபட்சம் {m['offset_max']:.1f} பிக்சல் மட்டுமே விலகுகிறது.")}]
    terms_en = ", ".join(f"{TERM_NAMES['en'][k]} {br[k]:.1f}" for k in TERM_NAMES["en"])
    terms_ta = ", ".join(f"{TERM_NAMES['ta'][k]} {br[k]:.1f}" for k in TERM_NAMES["ta"])
    points.append({"en": f"Cost breakdown: {terms_en}.", "ta": f"செலவுப் பிரிவு: {terms_ta}."})
    cov = [(v, k) for k, v in res.cover.items() if v >= 0.1 and k != "unexplained"]
    if cov:
        cov.sort(reverse=True)
        points.append({"en": "Along the new section: " + ", ".join(f"{_pct(v)}% {COVER_NAMES['en'][k]}" for v, k in cov) + ".",
                       "ta": "புதிய பகுதியில்: " + ", ".join(f"{_pct(v)}% {COVER_NAMES['ta'][k]}" for v, k in cov) + "."})
    ref = res.reference
    if ref and ref["total_cost"] > m["total_cost"] * 1.005:
        diff = {k: ref["terms"][k] - br[k] for k in br}
        top = max(diff, key=diff.get)
        more = 100 * (ref["total_cost"] / max(m["total_cost"], 1e-6) - 1)
        points.append({"en": f"Following the expected course exactly would cost {ref['total_cost']:.1f} ({more:.0f}% more), mainly because of {TERM_NAMES['en'][top]}.",
                       "ta": f"எதிர்பார்க்கப்படும் பாதையை அப்படியே பின்பற்றினால் {ref['total_cost']:.1f} செலவாகும் ({more:.0f}% அதிகம்); முக்கியக் காரணம்: {TERM_NAMES['ta'][top]}."})
    else:
        points.append({"en": "The chosen route is the expected course itself, which was already the cheapest.",
                       "ta": "தேர்ந்தெடுக்கப்பட்ட வழி எதிர்பார்க்கப்படும் பாதையே; அதுவே மிகக் குறைந்த செலவுடையது."})
    points.append({"en": f"The new segment is drawn {outcome['thickness']} px wide, matching the measured road width.",
                   "ta": f"புதிய பகுதி அளவிடப்பட்ட சாலை அகலத்திற்கு ஏற்ப {outcome['thickness']} பிக்சல் அகலத்தில் வரையப்பட்டது."})
    result = {"en": f"The reconstructed path was added and the road network was updated. Road components: {b} before, {a} now.",
              "ta": f"மீண்டும் உருவாக்கப்பட்ட பாதை சேர்க்கப்பட்டு, சாலை வலையமைப்பு புதுப்பிக்கப்பட்டது. சாலைப் பகுதிகள்: முன்பு {b}, இப்போது {a}."}
    return {"start": start, "result": result, "points": points, "succeeded": True}


def decision_voice(c, decision: str) -> dict[str, str]:
    if decision == "rejected":
        return {"en": f"Gap {c.id} rejected. It will not be reconstructed, and the road network stays unchanged.",
                "ta": f"இடைவெளி {c.id} நிராகரிக்கப்பட்டது. இது மீண்டும் உருவாக்கப்படாது; சாலை வலையமைப்பு மாறாமல் உள்ளது."}
    return {"en": f"Gap {c.id} skipped. It stays pending.", "ta": f"இடைவெளி {c.id} தவிர்க்கப்பட்டது. இது நிலுவையில் உள்ளது."}


# --------------------------------------------------------------------------- #
# Stage announcements (built from actual results)
# --------------------------------------------------------------------------- #
def stage_voices(info: dict[str, Any]) -> dict[str, dict[str, str]]:
    v: dict[str, dict[str, str]] = {
        "upload": {"en": "Image uploaded successfully. The analysis is starting.",
                   "ta": "படம் வெற்றிகரமாகப் பதிவேற்றப்பட்டது. பகுப்பாய்வு தொடங்குகிறது."}}
    if "width" in info:
        v["original"] = {"en": f"This is the original image, {info['width']} by {info['height']} pixels. It is kept unchanged.",
                         "ta": f"இது அசல் படம், {info['width']} × {info['height']} பிக்சல். இது மாற்றப்படாமல் பாதுகாக்கப்படுகிறது."}
    if "road_pct" in info:
        v["mask"] = {"en": (f"The road mask has been generated. White pixels are detected roads and black pixels are the background. "
                            f"Roads cover {info['road_pct']} percent of the image."),
                     "ta": (f"சாலை மாஸ்க் உருவாக்கப்பட்டது. வெள்ளை பிக்சல்கள் கண்டறியப்பட்ட சாலைகள்; கருப்பு பிக்சல்கள் பின்னணி. "
                            f"படத்தின் {info['road_pct']} சதவீதம் சாலையாக உள்ளது.")}
    if "nodes" in info:
        v["graph"] = {"en": f"The road graph has {info['nodes']} nodes and {info['edges']} edges, in {info['components']} connected parts.",
                      "ta": f"சாலை வரைபடத்தில் {info['nodes']} கணுக்களும் {info['edges']} விளிம்புகளும் உள்ளன; இவை {info['components']} இணைந்த பகுதிகளாக உள்ளன."}
    if "fragments" in info:
        v["fragments"] = {"en": f"{info['fragments']} road fragments have open ends where a road may be interrupted.",
                          "ta": f"{info['fragments']} சாலைத் துண்டுகளுக்குத் திறந்த முனைகள் உள்ளன; அங்கே சாலை இடைமறிக்கப்பட்டிருக்கலாம்."}
    if "gaps" in info:
        if info["gaps"]:
            v["gaps"] = {"en": (f"{info['gaps']} possible road gaps were identified, and {info['excluded']} other endpoint pairs were excluded. "
                                "Each gap will now be analysed separately."),
                         "ta": (f"{info['gaps']} சாத்தியமான சாலை இடைவெளிகள் கண்டறியப்பட்டன; மேலும் {info['excluded']} முனை ஜோடிகள் விலக்கப்பட்டன. "
                                "ஒவ்வொரு இடைவெளியும் இப்போது தனித்தனியாக ஆய்வு செய்யப்படும்.")}
        else:
            v["gaps"] = {"en": "No genuine road gaps were identified.", "ta": "உண்மையான சாலை இடைவெளிகள் எதுவும் கண்டறியப்படவில்லை."}
    return v


def final_voice(s: dict[str, Any]) -> dict[str, str]:
    if s["pending"] and not s["complete"]:
        return {"en": f"The review is not finished: {s['pending']} gaps are still waiting for a decision.",
                "ta": f"ஆய்வு இன்னும் முடியவில்லை: {s['pending']} இடைவெளிகள் இன்னும் முடிவுக்காகக் காத்திருக்கின்றன."}
    fail_en = f", and {s['failed']} could not be reconstructed" if s["failed"] else ""
    fail_ta = f"; {s['failed']} உருவாக்க முடியவில்லை" if s["failed"] else ""
    return {"en": (f"The review is complete. Of {s['total']} gaps, {s['approved']} were approved, {s['rejected']} rejected and "
                   f"{s['pending']} are pending. {s['reconstructed']} were reconstructed successfully{fail_en}. "
                   f"Road components went from {s['components_before']} to {s['components_after']}."),
            "ta": (f"ஆய்வு முடிந்தது. மொத்தம் {s['total']} இடைவெளிகளில் {s['approved']} ஏற்கப்பட்டன, {s['rejected']} நிராகரிக்கப்பட்டன, "
                   f"{s['pending']} நிலுவையில் உள்ளன. {s['reconstructed']} வெற்றிகரமாக மீண்டும் உருவாக்கப்பட்டன{fail_ta}. "
                   f"சாலைப் பகுதிகள் {s['components_before']}-இலிருந்து {s['components_after']} ஆக மாறின.")}


def view_voices(s: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        "compare": {"en": f"Compare the original road network with the reconstructed one. {s['reconstructed']} new connections are shown in green.",
                    "ta": f"அசல் சாலை வலையமைப்பையும் மீண்டும் உருவாக்கப்பட்டதையும் ஒப்பிடுங்கள். {s['reconstructed']} புதிய இணைப்புகள் பச்சை நிறத்தில் காட்டப்பட்டுள்ளன."},
        "map": {"en": "Explore the map. Use the layer buttons to show or hide roads, gaps and reconstructed paths, and select a gap to see its analysis.",
                "ta": "வரைபடத்தை ஆராயுங்கள். சாலைகள், இடைவெளிகள், மீண்டும் உருவாக்கப்பட்ட பாதைகளைக் காட்டவோ மறைக்கவோ அடுக்குப் பொத்தான்களைப் பயன்படுத்துங்கள்; ஒரு இடைவெளியைத் தேர்ந்தெடுத்து அதன் ஆய்வைப் பாருங்கள்."},
    }
