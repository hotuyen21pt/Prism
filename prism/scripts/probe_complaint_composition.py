#!/usr/bin/env python3
"""
Probe: Compositional Aspect-Complaint Tracking (CACT) trên pool Booking.

Đo share-of-complaints theo tháng cho từng aspect, so sánh chuỗi THÔ với chuỗi
ĐÃ HIỆU CHỈNH THÀNH PHẦN (direct standardisation trên country_bloc × traveller_type),
rồi tách mùa vụ và ước lượng xu hướng trên thang CLR.

Đây là proxy bằng keyword (lexicon sinh từ gold aspect_term) — dùng để kiểm chứng
tính khả thi TRƯỚC khi đầu tư vào ASQP extractor. Pipeline thật chỉ thay bước
"đếm lượt nhắc bằng keyword" bằng "đếm quad có trọng số tin cậy".

Chạy:  python3 scripts/probe_complaint_composition.py
"""
import json, re, math, pickle, collections, os, unicodedata
from pathlib import Path
import statistics as s

ROOT   = Path(__file__).resolve().parents[1]
HAMOS  = Path(os.environ.get('PRISM_HAMOS_ROOT', ROOT.parent / 'hamos-mabsa'))
POOL   = Path(os.environ.get('PRISM_POOL_JSONL', ROOT / 'data/raw/hotel_booking_unlabeled.jsonl'))
QUADS  = Path(os.environ.get('PRISM_GOLD_QUADS', HAMOS / 'data/annotations/quads.jsonl'))
OUT    = Path(os.environ.get('PRISM_PROBE_OUT', ROOT / 'outputs/probe/outputs_cact_probe.pkl'))
WINDOW = ('2022-03', '2025-02')      # cắt 2022-02 (thưa) và 2025-03 (crawl dở)
MIN_STRATUM = 30                      # ô strata dưới ngưỡng này bị bỏ qua trong kỳ đó

CODES = ['FAC_ROOM','FAC_VIEW_LOCATION','AM_POOL','FAC_BUILDING','SER_ATTITUDE','AM_FOOD',
         'FAC_BATH','AM_ROOM_UTIL','FAC_ENV','AM_WIFI','FAC_CLIMATE','AM_TRANSPORT','SER_SUPPORT']

SEED = {  # seed thủ công độ chính xác cao, lấy từ aspect_term tần suất cao nhất của gold
 'FAC_ROOM':['room','rooms','bed','beds','phòng','giường','mattress','nệm'],
 'FAC_VIEW_LOCATION':['view','location','beach','vị trí','biển','trung tâm','địa điểm'],
 'AM_POOL':['pool','swimming pool','hồ bơi','bể bơi'],
 'FAC_BUILDING':['hotel','resort','khách sạn','homestay','villa','property'],
 'SER_ATTITUDE':['staff','reception','nhân viên','lễ tân','owner','host','chủ nhà'],
 'AM_FOOD':['breakfast','food','đồ ăn','bữa sáng','ăn sáng','restaurant','nhà hàng','coffee'],
 'FAC_BATH':['bathroom','shower','toilet','phòng tắm','nước nóng','nhà vệ sinh','bồn tắm'],
 'AM_ROOM_UTIL':['towel','towels','kitchen','amenities','tiện nghi','tủ lạnh','khăn tắm','tv'],
 'FAC_ENV':['noise','smell','garden','không gian','cách âm','mùi','ồn'],
 'AM_WIFI':['wifi','wi-fi','internet','mạng'],
 'FAC_CLIMATE':['air conditioning','aircon','máy lạnh','điều hoà','điều hòa','quạt','a/c'],
 'AM_TRANSPORT':['shuttle','taxi','parking','xe','đưa đón','bãi đỗ','grab'],
 'SER_SUPPORT':['check in','check-in','checkin','nhận phòng','trả phòng','check out']}

WEST = {'pháp','úc','đức','vương quốc anh','mỹ','tây ban nha','hà lan','ý','bỉ','canada',
        'thụy sĩ','thuỵ sĩ','áo','đan mạch','thụy điển','thuỵ điển','na uy','ireland',
        'new zealand','ba lan','séc','bồ đào nha'}
ASIA = {'nhật bản','hàn quốc','trung quốc','thái lan','singapore','malaysia','đài loan',
        'hồng kông','indonesia','philippines','ấn độ','campuchia','lào'}

norm = lambda x: unicodedata.normalize('NFC', (x or '').lower())

def bloc(c):
    c = norm(c).strip()
    if c == 'việt nam': return 'VN'
    if c in WEST:       return 'WEST'
    if c in ASIA:       return 'ASIA'
    return 'OTH'

def build_lexicon():
    """term -> code, chỉ giữ term có code trội >=60% và >=3 lần trong gold."""
    tc = collections.defaultdict(collections.Counter)
    for line in open(QUADS):
        q = json.loads(line)
        a = q.get('aspect_term')
        if not a: continue
        a = norm(a).strip()
        if len(a) < 3 or len(a.split()) > 3: continue
        tc[a][q['taxonomy_code']] += 1
    lex = collections.defaultdict(set)
    for term, cc in tc.items():
        code, n = cc.most_common(1)[0]
        if code in CODES and n >= 3 and n / sum(cc.values()) >= 0.6:
            lex[code].add(term)
    for c, ts in SEED.items():
        lex[c].update(ts)
    return {c: re.compile(r'(?<![\w])(?:' +
            '|'.join(sorted((re.escape(t) for t in ts), key=len, reverse=True)) +
            r')(?![\w])') for c, ts in lex.items()}

def scan(pat):
    neg  = collections.defaultdict(collections.Counter)   # month -> code
    negS = collections.defaultdict(collections.Counter)   # (month, stratum) -> code
    strn = collections.defaultdict(collections.Counter)   # month -> stratum
    nrev = collections.Counter()
    for line in open(POOL):
        d = json.loads(line)
        m = re.search(r'ngày (\d+) tháng (\d+) năm (\d{4})', d.get('review_date') or '')
        if not m: continue
        k = f"{m.group(3)}-{int(m.group(2)):02d}"
        if not (WINDOW[0] <= k <= WINDOW[1]): continue
        ng = norm(d.get('review_negative'))
        if not ng.strip(): continue
        g = (bloc(d.get('country')), d.get('state') or 'NA')
        nrev[k] += 1; strn[k][g] += 1
        for c, p in pat.items():
            if p.search(ng):
                neg[k][c] += 1; negS[(k, g)][c] += 1
    return neg, negS, strn, nrev

def clr(p):
    ps = {c: max(v, 1e-6) for c, v in p.items()}
    g = math.exp(sum(math.log(v) for v in ps.values()) / len(ps))
    return {c: math.log(v / g) for c, v in ps.items()}

def deseason_trend(series, months):
    """Khử hiệu ứng tháng-trong-năm rồi OLS. Trả (slope/năm, t-stat, biên độ mùa vụ)."""
    n = len(series); xs = list(range(n))
    moy = collections.defaultdict(list)
    for i, k in enumerate(months): moy[k[5:7]].append(series[i])
    mm = {m: s.mean(v) for m, v in moy.items()}; gm = s.mean(series)
    de = [series[i] - (mm[months[i][5:7]] - gm) for i in range(n)]
    mx, my = s.mean(xs), s.mean(de)
    b  = sum((x-mx)*(y-my) for x, y in zip(xs, de)) / sum((x-mx)**2 for x in xs)
    res = [y - (my + b*(x-mx)) for x, y in zip(xs, de)]
    se = math.sqrt(sum(r*r for r in res)/(n-2)/sum((x-mx)**2 for x in xs))
    return b*12, (b/se if se else 0.0), s.pstdev([mm[m] for m in sorted(mm)])

def main():
    pat = build_lexicon()
    neg, negS, strn, nrev = scan(pat)
    months = sorted(neg)

    ref = collections.Counter()
    for k in months:
        for g, v in strn[k].items(): ref[g] += v
    tot = sum(ref.values()); ref = {g: v/tot for g, v in ref.items()}

    def raw(k):
        T = sum(neg[k][c] for c in CODES) or 1
        return {c: neg[k][c]/T for c in CODES}

    def adjusted(k):
        acc = {c: 0.0 for c in CODES}
        for g, w in ref.items():
            if strn[k].get(g, 0) < MIN_STRATUM: continue
            cnt = negS.get((k, g), {})
            t = sum(cnt.get(c, 0) for c in CODES)
            if not t: continue
            for c in CODES: acc[c] += w * cnt.get(c, 0) / t
        T = sum(acc.values()) or 1
        return {c: acc[c]/T for c in CODES}

    R = {k: raw(k) for k in months}; A = {k: adjusted(k) for k in months}
    print(f"{len(months)} tháng · {sum(nrev.values()):,} review có text negative · {len(ref)} strata\n")
    print(f"{'code':20s} | {'RAW /năm':>9s} {'t':>6s} | {'ADJ /năm':>9s} {'t':>6s} | {'mùa vụ':>6s} | kết luận")
    print("-"*92)
    rows = []
    for c in CODES:
        br, tr, _  = deseason_trend([clr(R[k])[c] for k in months], months)
        ba, ta, sa = deseason_trend([clr(A[k])[c] for k in months], months)
        rows.append((c, br, tr, ba, ta, sa))
    for c, br, tr, ba, ta, sa in sorted(rows, key=lambda r: -abs(r[3])):
        if abs(ta) > 2.5 and ba*br > 0:                v = "XU HƯỚNG THẬT"
        elif abs(tr) > 2.5 and abs(ta) <= 2.5:         v = "GIẢ (do thành phần)"
        elif abs(ta) > 2.5 and abs(tr) <= 2.5:         v = "BỊ CHE, lộ ra sau hiệu chỉnh"
        elif abs(ta) > 2.5 and ba*br < 0:              v = "ĐẢO DẤU sau hiệu chỉnh"
        else:                                          v = "phẳng"
        print(f"{c:20s} | {br:+9.3f} {tr:+6.1f} | {ba:+9.3f} {ta:+6.1f} | {sa:6.2f} | {v}")
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump({'neg':dict(neg),'negS':dict(negS),'strn':dict(strn),'nrev':dict(nrev),
                 'months':months,'raw':R,'adj':A,'ref':ref}, open(OUT,'wb'))
    print(f"\n→ {OUT}")

if __name__ == '__main__':
    main()
