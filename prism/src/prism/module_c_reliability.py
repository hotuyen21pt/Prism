"""
Module C — Cross-Modal Reliability Calibration  [lõi novelty · có điểm go/no-go].

C1  Học verifier V(image, category) -> [0,1] trên 9.219 ảnh gold
    (mức REVIEW × CATEGORY — phạm vi duy nhất mà alignment cho phép:
     95,6% review chỉ có 1 ảnh, 42,6% review 1-ảnh trải >1 taxonomy code).
C2  Áp V lên các review pool có ảnh (đã tải ảnh — xem download_pool_photos.py).
C3  Cầu nối: fit g(đặc trưng chỉ-text) ≈ P[V=1] trên tập có ảnh (kèm IPW theo
    has_photo propensity), rồi áp cho toàn bộ quad — kể cả review không ảnh.

    w_q = 1[tax hợp lệ] · 1[span hợp lệ] · r̂(q)

GO/NO-GO (chốt ngưỡng TRƯỚC khi chạy — sửa tại đây nếu nhóm quyết khác):
    AUC của V trên gold test  >= 0.70
    AUC(V) - AUC(baseline không ảnh) >= 0.05   [chống rò rỉ category-prior:
        one-hot category tự dự đoán được y vì FACILITY có mặt ở hầu hết review;
        AUC cao mà không vượt baseline nghĩa là ảnh KHÔNG đóng góp gì]
    Spearman(r̂, human audit) >= 0.30
Trượt bất kỳ ngưỡng nào -> bỏ Module C, dùng w_q = conf_seq THÔ
(pipeline vẫn chạy, bài lùi về A+B+D). Fallback KHÔNG có temperature scaling —
repo chưa implement bước đó, đừng mô tả nó là "đã calibrate".

CƯỠNG CHẾ trong code (không chỉ ghi vào report):
  - verdict được ghi VÀO pickle, không chỉ vào json
  - bridge NO-GO ghi ra bridge_NOGO.pkl và XOÁ bridge.pkl cũ
  - apply_weights chỉ dùng bridge khi verdict == "GO", và kiểm parity đặc trưng
    (feature_names + n_features_in_) trước khi predict
  - thiếu audit / khớp <30 cặp cũng là NO-GO, không phải "bỏ qua im lặng"

Chạy: python3 -m prism.module_c_reliability \
          --stage {train_verifier,apply_verifier,bridge,apply} --cohort T-unbiased
(apply_verifier = C2: gắn v_image cho pool quads có ảnh đã tải —
 xem scripts/download_pool_photos.py)
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random

from . import config as C
from . import utils as U

log = U.get_logger("prism.C")

GO_NOGO = {"verifier_auc_min": 0.70, "verifier_delta_auc_min": 0.05,
           "bridge_spearman_min": 0.30}

# Đặc trưng của bridge — ĐỊNH NGHĨA MỘT LẦN, dùng ở cả fit_bridge và apply_weights.
# Trước đây vector này được viết tay ở hai chỗ: thêm/đổi thứ tự một đặc trưng mà
# quên chỗ kia thì sklearn vẫn nhận đủ cột, KHÔNG raise, chỉ dịch sai cột -> r̂ rác
# -> w rác -> mọi số π/ν của paper sai trong khi pipeline chạy xanh hoàn toàn.
BRIDGE_FEATURE_NAMES = ["conf_seq", "p_posterior", "phi_pos", "log_len", "prov_flip"]


def bridge_features(q: dict) -> list[float]:
    x = [float(q["conf_seq"]), float(q["p_posterior"]),
         1.0 if q["phi"] == "POS" else 0.0,
         math.log1p(q["n_words"]),
         1.0 if q["provenance_flip"] else 0.0]
    assert len(x) == len(BRIDGE_FEATURE_NAMES), "bridge_features lệch BRIDGE_FEATURE_NAMES"
    return x


def propensity_features(q: dict) -> list[float]:
    """Đặc trưng cho P(has_photo | ...) dùng trong IPW.

    q["score"] có thể là 0.0 THẬT (parse_score cho phép 0-10) nên phải so
    `is None`: `q["score"] or DEFAULT` biến điểm 0 thành 8,7 — đúng cái đuôi
    quan trọng nhất cho drift phàn nàn.
    """
    score = q.get("score")
    return [math.log1p(q["n_words"]),
            C.DEFAULT_SCORE if score is None else float(score)]


# ---------------------------------------------------------------- C1: verifier
def build_verifier_dataset() -> list[dict]:
    """
    Cặp (image, category, y): y=1 nếu review của ảnh có >=1 quad thuộc category đó.
    Negative: category không xuất hiện trong review (sample cân bằng 1:1).
    Split THEO HOTEL, tái dùng đúng hotel-disjoint split của gold.
    """
    img_of: dict[str, list[str]] = collections.defaultdict(list)
    for r in U.read_jsonl(C.GOLD_IMAGES):
        img_of[r["source_review_id"]].append(r["local_path"])

    cats_of: dict[str, set] = collections.defaultdict(set)
    for q in U.read_jsonl(C.GOLD_QUADS):
        rid = q["quad_id"].rsplit("_Q", 1)[0]
        cats_of[rid].add(q["aspect_category"])

    split_of = {}
    for name in ("train", "dev", "test"):
        for row in U.read_jsonl(C.SPLIT_DIR / f"{name}.jsonl"):
            split_of[row["source_review_id"]] = name

    rng = random.Random(C.RANDOM_SEED)
    rows = []
    for rid, paths in img_of.items():
        pos = sorted(cats_of.get(rid, set()))
        neg_pool = [c for c in C.CATEGORIES if c not in pos]
        for p in paths:
            for c in pos:
                rows.append({"image": p, "category": c, "y": 1,
                             "split": split_of.get(rid, "train"), "rid": rid})
            for c in rng.sample(neg_pool, min(len(pos), len(neg_pool))):
                rows.append({"image": p, "category": c, "y": 0,
                             "split": split_of.get(rid, "train"), "rid": rid})
    if not rows:
        raise SystemExit(
            f"verifier dataset RỖNG — không đọc được ảnh/quad gold. Kiểm tra "
            f"PRISM_HAMOS_ROOT={C.HAMOS_ROOT} (cần {C.GOLD_IMAGES} và {C.GOLD_QUADS})")
    log.info("verifier dataset: %d cặp (pos %.1f%%)", len(rows),
             100 * sum(r["y"] for r in rows) / len(rows))
    return rows


def train_verifier() -> None:
    """
    CLIP zero-shot làm khởi điểm + logistic head học được trên train split.
    Encoder ĐÓNG BĂNG (A8: ghi version) — chỉ head là tham số học.
    """
    import numpy as np
    import torch
    import open_clip
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = build_verifier_dataset()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model_name, pretrained = "ViT-B-32", "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model = model.to(device).eval()

    # prompt tiếng Anh mô tả 6 category khách sạn
    PROMPTS = {
        "FACILITY":  "a photo of a hotel room, building, bathroom or view",
        "AMENITY":   "a photo of a hotel pool, food, breakfast or amenities",
        "SERVICE":   "a photo of hotel staff or reception service",
        "EXPERIENCE":"a photo showing the overall hotel experience and atmosphere",
        "LOYALTY":   "a photo a returning hotel guest would take",
        "BRANDING":  "a photo showing hotel branding and luxury identity",
    }
    with torch.no_grad():
        txt = open_clip.tokenize([PROMPTS[c] for c in C.CATEGORIES]).to(device)
        tfeat = model.encode_text(txt)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # cache image embeddings (mỗi ảnh 1 lần)
    uniq = sorted({r["image"] for r in rows})
    feats: dict[str, "np.ndarray"] = {}
    n_fail = 0
    for i, rel in enumerate(uniq):
        p = C.HAMOS_ROOT / "data" / rel
        if not p.exists():
            p = C.HAMOS_ROOT / rel          # layout phẳng: hamos-mabsa/images/...
        try:
            im = preprocess(Image.open(p).convert("RGB"))
        except Exception:
            n_fail += 1
            continue
        with torch.no_grad():
            f = model.encode_image(im.unsqueeze(0).to(device))
            feats[rel] = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        if i % 500 == 0:
            log.info("  embed ảnh %d/%d", i, len(uniq))
    if not feats:
        raise SystemExit(
            f"KHÔNG mở được ảnh gold nào ({n_fail}/{len(uniq)} lỗi). Kiểm tra "
            f"IMAGE_DIR dưới PRISM_HAMOS_ROOT={C.HAMOS_ROOT}")
    if n_fail:
        log.warning("%d/%d ảnh gold không mở được -> bỏ khỏi tập verifier",
                    n_fail, len(uniq))

    cat_idx = {c: i for i, c in enumerate(C.CATEGORIES)}
    def xy(split_names):
        X, y = [], []
        for r in rows:
            if r["split"] not in split_names or r["image"] not in feats:
                continue
            sim = float(feats[r["image"]] @ tfeat[cat_idx[r["category"]]].cpu().numpy())
            onehot = [0.0] * len(C.CATEGORIES); onehot[cat_idx[r["category"]]] = 1.0
            X.append([sim] + onehot); y.append(r["y"])
        return np.array(X), np.array(y)

    Xtr, ytr = xy({"train", "dev"})
    Xte, yte = xy({"test"})
    head = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    auc = roc_auc_score(yte, head.predict_proba(Xte)[:, 1])

    # baseline KHÔNG ảnh (chỉ one-hot category): kiểm soát rò rỉ category-prior.
    # Nếu verifier không vượt baseline một khoảng delta_auc_min thì ảnh vô dụng
    # dù AUC tuyệt đối cao — go/no-go phải dựa trên delta, không chỉ AUC thô.
    head_base = LogisticRegression(max_iter=1000).fit(Xtr[:, 1:], ytr)
    auc_base = roc_auc_score(yte, head_base.predict_proba(Xte[:, 1:])[:, 1])
    delta = auc - auc_base
    verdict = ("GO" if auc >= GO_NOGO["verifier_auc_min"]
               and delta >= GO_NOGO["verifier_delta_auc_min"] else "NO-GO")
    log.info("VERIFIER AUC=%.4f  baseline(no-image)=%.4f  Δ=%.4f  ->  %s",
             auc, auc_base, delta, verdict)

    # AUC tách theo category (trong từng category chỉ còn tín hiệu ảnh)
    auc_by_cat = {}
    te_rows = [r for r in rows if r["split"] == "test" and r["image"] in feats]
    for cat in C.CATEGORIES:
        sub = [r for r in te_rows if r["category"] == cat]
        ys = [r["y"] for r in sub]
        if len(set(ys)) < 2:
            auc_by_cat[cat] = None
            continue
        sims = [float(feats[r["image"]] @ tfeat[cat_idx[cat]].cpu().numpy())
                for r in sub]
        auc_by_cat[cat] = round(float(roc_auc_score(ys, sims)), 4)
    log.info("AUC theo category: %s", auc_by_cat)

    import pickle
    with open(C.RELIAB_DIR / "verifier.pkl", "wb") as f:
        pickle.dump({"head": head, "model_name": model_name,
                     "pretrained": pretrained, "prompts": PROMPTS,
                     "go_nogo": verdict, "auc_test": float(auc),
                     "delta_auc": float(delta)}, f)
    if verdict != "GO":
        log.warning("VERIFIER NO-GO (AUC=%.4f Δ=%.4f) — apply_verifier vẫn chạy được "
                    "để khảo sát, nhưng v_image KHÔNG dùng cho số chính thức; "
                    "bridge sẽ bị chấm NO-GO theo.", auc, delta)
    U.write_json(C.RELIAB_DIR / "verifier_report.json", {
        "auc_test": float(auc), "auc_no_image_baseline": float(auc_base),
        "delta_auc": float(delta), "auc_by_category": auc_by_cat,
        "n_train": len(ytr), "n_test": len(yte),
        "go_nogo": verdict, "thresholds": GO_NOGO,
        "encoder_version": f"open_clip/{model_name}/{pretrained}",
    })


# ----------------------------------------------------------- C2: apply verifier
def apply_verifier(quad_file, image_index_file, out_file) -> None:
    """
    C2 — gắn v_image = V(ảnh, category của quad) cho quad thuộc review CÓ ảnh
    pool đã tải về (scripts/download_pool_photos.py tạo image_index_file:
    {review_uid: đường dẫn ảnh}). Quad không có ảnh giữ nguyên, không gắn v_image.
    """
    import numpy as np
    import pickle
    import torch
    import open_clip
    from PIL import Image

    vp = C.RELIAB_DIR / "verifier.pkl"
    if not vp.exists():
        log.error("chưa có verifier.pkl — chạy stage train_verifier trước")
        return
    with open(vp, "rb") as f:
        saved = pickle.load(f)
    head, prompts = saved["head"], saved["prompts"]
    if saved.get("go_nogo") not in (None, "GO"):
        log.warning("verifier.pkl có verdict=%s — v_image gắn ra chỉ để khảo sát",
                    saved["go_nogo"])

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        saved["model_name"], pretrained=saved["pretrained"])
    model = model.to(device).eval()
    with torch.no_grad():
        txt = open_clip.tokenize([prompts[c] for c in C.CATEGORIES]).to(device)
        tfeat = model.encode_text(txt)
        tfeat = (tfeat / tfeat.norm(dim=-1, keepdim=True)).cpu().numpy()
    cat_idx = {c: i for i, c in enumerate(C.CATEGORIES)}

    index = json.loads(U.Path(image_index_file).read_text())
    # Cache LRU nhỏ, KHÔNG cache toàn bộ: bản cũ giữ mọi embedding review đã gặp,
    # tức vài trăm MB cho corpus mà gần như không dùng lại.
    # Vì sao 256 chứ không phải 1: quad của cùng review thường liền nhau, nhưng
    # hai unit (POS/NEG) của một review có thể bị TÁCH RA khi module_b_infer chạy
    # với --limit (reservoir sampling có shuffle). Cache 1 phần tử sẽ embed lại ảnh
    # đó lần thứ hai; 256 phần tử (~0,5 MB) là đủ và vẫn có chặn trên.
    CACHE_MAX = 256
    cache: "collections.OrderedDict[str, object]" = collections.OrderedDict()
    stats = collections.Counter()

    def embed(uid: str):
        if uid in cache:
            cache.move_to_end(uid)
            return cache[uid]
        while len(cache) >= CACHE_MAX:
            cache.popitem(last=False)
        try:
            im = preprocess(Image.open(index[uid]).convert("RGB"))
        except Exception as e:
            stats["decode_fail"] += 1
            if stats["decode_fail"] <= 10:
                log.warning("không mở được ảnh %s: %s", index[uid], e)
            cache[uid] = None
            return None
        with torch.no_grad():
            f = model.encode_image(im.unsqueeze(0).to(device))
            cache[uid] = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        return cache[uid]

    def rows():
        for q in U.read_jsonl(quad_file):
            uid = q["review_uid"]
            if uid in index:
                f = embed(uid)
                if f is not None:
                    ci = cat_idx[q["aspect_category"]]
                    sim = float(f @ tfeat[ci])
                    onehot = [0.0] * len(C.CATEGORIES); onehot[ci] = 1.0
                    q["v_image"] = round(float(
                        head.predict_proba([[sim] + onehot])[0, 1]), 4)
                    stats["with_vimage"] += 1
                else:
                    stats["quad_lost_to_bad_image"] += 1
            else:
                stats["no_image_in_index"] += 1
            yield q

    n = U.write_jsonl(out_file, rows())
    log.info("v_image cho %d/%d quad -> %s", stats["with_vimage"], n, out_file)
    # Phân tách rõ ba lý do thiếu v_image — trước đây ảnh hỏng bị `except` ăn im lặng
    # nên không phân biệt được "ảnh hỏng" với "review không có ảnh".
    log.info("  không có ảnh trong index: %d · ảnh KHÔNG decode được: %d quad "
             "(%d file lỗi)", stats["no_image_in_index"],
             stats["quad_lost_to_bad_image"], stats["decode_fail"])
    if stats["decode_fail"]:
        log.warning("%d file ảnh không decode được — chạy lại download_pool_photos "
                    "(nó đã kiểm content-type nên sẽ tải lại đúng file hỏng)",
                    stats["decode_fail"])


# ------------------------------------------------------------------ C3: bridge
def fit_bridge(quad_file, audit_file=None) -> None:
    """
    Fit g: đặc trưng-chỉ-text -> P[V=1] trên quad thuộc review CÓ ảnh (đã chạy
    verifier), kèm IPW theo has_photo propensity ([đo] có ảnh 56,7 từ / 8,90 điểm
    vs không ảnh 36,5 / 8,68 — bỏ IPW là sai).
    Nếu có audit_file (300 quad người gán đúng/sai): đo Spearman(r̂, đúng/sai).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr

    # Chỉ giữ TRƯỜNG CẦN DÙNG, không giữ cả dòng: trường `text` chiếm phần lớn dung
    # lượng mỗi quad và fit_bridge không dùng nó. Nạp cả dòng (bản cũ) tốn ~4,3 GB
    # cho corpus (~3,6M quad × ~1,2 KB) — sát hạn mức RAM của Kaggle.
    _KEEP = ("conf_seq", "p_posterior", "phi", "n_words", "provenance_flip",
             "score", "has_photo", "v_image",
             "review_uid", "taxonomy_code", "opinion_term")   # 3 cái cuối: quad_uid
    quads = [{k: q[k] for k in _KEEP if k in q} for q in U.read_jsonl(quad_file)]
    with_v = [q for q in quads if "v_image" in q]
    log.info("nạp %d quad (%d có v_image) từ %s", len(quads), len(with_v), quad_file)
    if not with_v:
        log.error("chưa có v_image — chạy stage apply-verifier trên pool có ảnh trước")
        return

    # IPW: propensity P(has_photo | length, score) trên toàn bộ quads
    yp = np.array([1 if q["has_photo"] else 0 for q in quads])
    if len(set(yp.tolist())) < 2:
        log.error("propensity không fit được: has_photo đồng nhất (%d/%d True) trong "
                  "%s — cần file chứa CẢ quad có ảnh và không ảnh (pool_quads_vimg "
                  "của cả cohort, không phải tập con đã lọc)",
                  int(yp.sum()), len(yp), quad_file)
        return
    Xp = np.array([propensity_features(q) for q in quads])
    prop = LogisticRegression(max_iter=1000).fit(Xp, yp)
    w_ipw = 1.0 / np.clip(prop.predict_proba(
        np.array([propensity_features(q) for q in with_v]))[:, 1], 0.02, 1.0)

    # Nhãn MỀM: v_image là xác suất liên tục, ép về 0/1 tại 0,5 vừa mất hết độ tin
    # cậy (0,51 và 0,99 thành cùng một nhãn) vừa crash "needs samples of at least
    # 2 classes" khi mọi v_image rơi cùng một phía — head fit trên tập cân bằng
    # 1:1 còn pool thì không. Cách chuẩn: nhân đôi mỗi hàng thành (y=1, w=v) và
    # (y=0, w=1-v), tương đương cross-entropy trên nhãn mềm.
    Xb_one = np.array([bridge_features(q) for q in with_v])
    v = np.array([float(q["v_image"]) for q in with_v])
    Xb = np.vstack([Xb_one, Xb_one])
    yb = np.concatenate([np.ones(len(with_v)), np.zeros(len(with_v))])
    sw_soft = np.concatenate([v, 1.0 - v])
    bridge = LogisticRegression(max_iter=1000).fit(
        Xb, yb, sample_weight=sw_soft * np.concatenate([w_ipw, w_ipw]))
    bridge_noipw = LogisticRegression(max_iter=1000).fit(Xb, yb, sample_weight=sw_soft)

    def feat(q):
        return bridge_features(q)

    report = {"n_with_image": len(with_v), "n_total": len(quads),
              "coef": bridge.coef_.tolist(),
              "feature_names": list(BRIDGE_FEATURE_NAMES),
              "v_image_mean": float(v.mean()), "v_image_min": float(v.min()),
              "v_image_max": float(v.max())}
    if audit_file and U.Path(audit_file).exists():
        audit = {a["quad_uid"]: a["correct"] for a in U.read_jsonl(audit_file)
                 if a.get("correct") is not None}
        # pool quads KHÔNG mang sẵn quad_uid — phải dựng lại bằng đúng hàm
        # U.quad_uid mà make_audit_samples đã dùng, nếu không sẽ không khớp cặp nào
        pairs = [(float(bridge.predict_proba([feat(q)])[0, 1]), audit[U.quad_uid(q)])
                 for q in quads if U.quad_uid(q) in audit]
        if len(pairs) >= 30:
            rho, pval = spearmanr([p for p, _ in pairs], [c for _, c in pairs])
            verdict = "GO" if rho >= GO_NOGO["bridge_spearman_min"] else "NO-GO"
            report.update({"audit_spearman": float(rho), "audit_p": float(pval),
                           "audit_n": len(pairs), "go_nogo": verdict})
            log.info("BRIDGE vs human audit: ρ=%.3f (n=%d) -> %s", rho, len(pairs), verdict)
        else:
            log.error("audit chỉ khớp %d/%d cặp (<30) — KHÔNG đủ để chấm go/no-go. "
                      "Khoá join là U.quad_uid: nếu 0 cặp thì audit được sinh bằng "
                      "phiên bản quad_uid khác, phải sinh lại mẫu audit.",
                      len(pairs), len(audit))
            report["go_nogo"] = "NO-GO (thiếu audit)"
    else:
        log.error("không thấy audit_file %s — chưa chấm được ngưỡng Spearman", audit_file)
        report["go_nogo"] = "NO-GO (thiếu audit)"

    # Verdict đi VÀO pickle, và pickle chỉ mang tên "bridge.pkl" khi GO.
    # Trước đây bridge NO-GO vẫn được ghi ra đúng tên đó và apply_weights dùng nó
    # chỉ vì file tồn tại -> hợp đồng "NO-GO thì fallback conf_seq" không bao giờ chạy.
    import pickle
    payload = {"bridge": bridge, "bridge_noipw": bridge_noipw, "propensity": prop,
               "go_nogo": report["go_nogo"],
               "feature_names": list(BRIDGE_FEATURE_NAMES),
               "thresholds": dict(GO_NOGO)}
    is_go = report["go_nogo"] == "GO"
    out = C.RELIAB_DIR / ("bridge.pkl" if is_go else "bridge_NOGO.pkl")
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    if not is_go:
        stale = C.RELIAB_DIR / "bridge.pkl"
        if stale.exists():
            stale.unlink()      # không để bản GO cũ sống sót và bị dùng nhầm
        log.warning("verdict=%s -> ghi %s (KHÔNG phải bridge.pkl). Module C bị bỏ, "
                    "apply sẽ fallback w=conf_seq; bài lùi về A+B+D.",
                    report["go_nogo"], out.name)
    U.write_json(C.RELIAB_DIR / "bridge_report.json", report)


def apply_weights(quad_file, out_file) -> None:
    """Gắn w_q cho mọi quad. Fallback w=conf_seq THÔ nếu bridge NO-GO/thiếu.

    (Fallback KHÔNG có temperature scaling — repo chưa implement bước đó; đừng mô
    tả nó là "conf_seq đã calibrate".)
    """
    import pickle
    bridge = None
    verdict = "NO-GO (không có bridge.pkl)"
    bp = C.RELIAB_DIR / "bridge.pkl"
    if bp.exists():
        with open(bp, "rb") as f:
            saved = pickle.load(f)
        verdict = saved.get("go_nogo", "GO (pickle cũ, không có verdict)")
        if verdict == "GO":
            bridge = saved["bridge"]
            # Bảo vệ parity đặc trưng: pickle mang theo tên cột lúc fit, và
            # sklearn biết số cột nó chờ. Lệch -> dừng, đừng sinh w rác.
            names = saved.get("feature_names")
            if names and list(names) != list(BRIDGE_FEATURE_NAMES):
                raise SystemExit(f"bridge.pkl fit trên {names} nhưng code hiện tại "
                                 f"dựng {BRIDGE_FEATURE_NAMES} — fit lại bridge")
            if getattr(bridge, "n_features_in_", None) != len(BRIDGE_FEATURE_NAMES):
                raise SystemExit(f"bridge.pkl chờ {bridge.n_features_in_} đặc trưng, "
                                 f"code dựng {len(BRIDGE_FEATURE_NAMES)} — fit lại bridge")
        else:
            log.warning("bridge.pkl có verdict=%s -> BỎ bridge, fallback conf_seq", verdict)

    if bridge is None:
        log.warning("Module C NO-GO/thiếu (%s) -> w = conf_seq thô. Ghi cờ "
                    "w_source=conf_seq vào từng dòng để Module D truy vết được.", verdict)

    src = "bridge" if bridge is not None else "conf_seq"

    def rows():
        for q in U.read_jsonl(quad_file):
            q.setdefault("quad_uid", U.quad_uid(q))   # truy vết row-level
            hard = 1.0 if q["taxonomy_code"] in C.CODE2CAT else 0.0
            if bridge is not None:
                r_hat = float(bridge.predict_proba([bridge_features(q)])[0, 1])
            else:
                r_hat = q["conf_seq"]      # fallback NO-GO
            q["w"] = round(hard * r_hat, 4)
            q["w_source"] = src
            yield q
    n = U.write_jsonl(out_file, rows())
    log.info("gắn w cho %d quad -> %s (nguồn w = %s, verdict = %s)",
             n, out_file, src, verdict)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train_verifier", "apply_verifier", "bridge", "apply"])
    ap.add_argument("--cohort", default="T-unbiased", choices=C.COHORTS)
    ap.add_argument("--quads", default=None,
                    help="mặc định: outputs/extract/<pool_quads của --cohort>")
    ap.add_argument("--audit", default=str(C.RELIAB_DIR / C.HUMAN_AUDIT_NAME))
    ap.add_argument("--image-index",
                    default=str(C.RELIAB_DIR / "pool_image_index.json"),
                    help="json {review_uid: path ảnh} từ scripts/download_pool_photos.py")
    ap.add_argument("--out", default=None,
                    help="mặc định theo stage: apply_verifier -> reliability/"
                         "<vimg của cohort>; apply -> reliability/quads_weighted.jsonl.gz")
    args = ap.parse_args()
    C.ensure_dirs()

    # Tên file mặc định lấy từ config -> README, module C và kaggle_pipeline không
    # thể lệch nhau nữa (orchestrator tìm input THEO TÊN).
    quads = args.quads or str(C.EXTRACT_DIR / C.pool_quads_name(args.cohort))
    if args.stage == "train_verifier":
        train_verifier()
    elif args.stage == "apply_verifier":
        out = args.out or str(C.RELIAB_DIR / C.vimg_quads_name(args.cohort))
        apply_verifier(quads, args.image_index, out)
    elif args.stage == "bridge":
        # bridge CẦN file đã có v_image = output của apply_verifier
        fit_bridge(args.quads or str(C.RELIAB_DIR / C.vimg_quads_name(args.cohort)),
                   args.audit)
    else:
        apply_weights(quads, args.out or str(C.RELIAB_DIR / "quads_weighted.jsonl.gz"))


if __name__ == "__main__":
    main()
