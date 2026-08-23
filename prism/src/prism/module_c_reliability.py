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
Trượt bất kỳ ngưỡng nào -> bỏ Module C, dùng w_q = conf_seq đã temperature-scale
(pipeline vẫn chạy, bài lùi về A+B+D).

Chạy: python3 -m prism.module_c_reliability \
          --stage {train_verifier,apply_verifier,bridge,apply}
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
    for i, rel in enumerate(uniq):
        try:
            im = preprocess(Image.open(C.HAMOS_ROOT / "data" / rel).convert("RGB"))
        except Exception:
            continue
        with torch.no_grad():
            f = model.encode_image(im.unsqueeze(0).to(device))
            feats[rel] = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        if i % 500 == 0:
            log.info("  embed ảnh %d/%d", i, len(uniq))
    np.save(C.RELIAB_DIR / "image_feats.npy",
            np.stack([feats[k] for k in sorted(feats)]))
    U.write_json(C.RELIAB_DIR / "image_feats_index.json", sorted(feats))

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
    pickle.dump({"head": head, "model_name": model_name, "pretrained": pretrained,
                 "prompts": PROMPTS},
                open(C.RELIAB_DIR / "verifier.pkl", "wb"))
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
    saved = pickle.load(open(vp, "rb"))
    head, prompts = saved["head"], saved["prompts"]

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
    feats: dict[str, "np.ndarray"] = {}

    def embed(uid: str):
        if uid in feats:
            return feats[uid]
        try:
            im = preprocess(Image.open(index[uid]).convert("RGB"))
        except Exception:
            feats[uid] = None
            return None
        with torch.no_grad():
            f = model.encode_image(im.unsqueeze(0).to(device))
            feats[uid] = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
        return feats[uid]

    n_img = 0

    def rows():
        nonlocal n_img
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
                    n_img += 1
            yield q

    n = U.write_jsonl(out_file, rows())
    log.info("v_image cho %d/%d quad -> %s", n_img, n, out_file)


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

    quads = [q for q in U.read_jsonl(quad_file)]
    with_v = [q for q in quads if "v_image" in q]
    if not with_v:
        log.error("chưa có v_image — chạy stage apply-verifier trên pool có ảnh trước")
        return

    def feat(q):
        return [q["conf_seq"], q["p_posterior"], 1.0 if q["phi"] == "POS" else 0.0,
                math.log1p(q["n_words"]),
                1.0 if q["provenance_flip"] else 0.0]

    # IPW: propensity P(has_photo | length, score) trên toàn bộ quads
    Xp = np.array([[math.log1p(q["n_words"]), (q["score"] or 8.7)] for q in quads])
    yp = np.array([1 if q["has_photo"] else 0 for q in quads])
    prop = LogisticRegression(max_iter=1000).fit(Xp, yp)
    w_ipw = 1.0 / np.clip(prop.predict_proba(
        np.array([[math.log1p(q["n_words"]), (q["score"] or 8.7)] for q in with_v])
    )[:, 1], 0.02, 1.0)

    Xb = np.array([feat(q) for q in with_v])
    yb = np.array([1 if q["v_image"] >= 0.5 else 0 for q in with_v])
    bridge = LogisticRegression(max_iter=1000).fit(Xb, yb, sample_weight=w_ipw)
    bridge_noipw = LogisticRegression(max_iter=1000).fit(Xb, yb)

    report = {"n_with_image": len(with_v), "n_total": len(quads),
              "coef": bridge.coef_.tolist(),
              "feature_names": ["conf_seq", "p_posterior", "phi_pos",
                                "log_len", "prov_flip"]}
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
    import pickle
    pickle.dump({"bridge": bridge, "bridge_noipw": bridge_noipw,
                 "propensity": prop}, open(C.RELIAB_DIR / "bridge.pkl", "wb"))
    U.write_json(C.RELIAB_DIR / "bridge_report.json", report)


def apply_weights(quad_file, out_file) -> None:
    """Gắn w_q cho mọi quad. Fallback conf_seq nếu chưa có bridge (NO-GO)."""
    import pickle
    bridge = None
    bp = C.RELIAB_DIR / "bridge.pkl"
    if bp.exists():
        bridge = pickle.load(open(bp, "rb"))["bridge"]

    def rows():
        for q in U.read_jsonl(quad_file):
            q.setdefault("quad_uid", U.quad_uid(q))   # truy vết row-level
            hard = 1.0 if q["taxonomy_code"] in C.CODE2CAT else 0.0
            if bridge is not None:
                r_hat = float(bridge.predict_proba([[
                    q["conf_seq"], q["p_posterior"],
                    1.0 if q["phi"] == "POS" else 0.0,
                    math.log1p(q["n_words"]),
                    1.0 if q["provenance_flip"] else 0.0]])[0, 1])
            else:
                r_hat = q["conf_seq"]      # fallback NO-GO
            q["w"] = round(hard * r_hat, 4)
            yield q
    n = U.write_jsonl(out_file, rows())
    log.info("gắn w cho %d quad -> %s (bridge=%s)", n, out_file, bridge is not None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["train_verifier", "apply_verifier", "bridge", "apply"])
    ap.add_argument("--quads", default=str(C.EXTRACT_DIR / "pool_quads.T-unbiased.jsonl.gz"))
    ap.add_argument("--audit", default=str(C.RELIAB_DIR / "human_audit_300.jsonl"))
    ap.add_argument("--image-index",
                    default=str(C.RELIAB_DIR / "pool_image_index.json"),
                    help="json {review_uid: path ảnh} từ scripts/download_pool_photos.py")
    ap.add_argument("--out", default=str(C.RELIAB_DIR / "quads_weighted.jsonl.gz"))
    args = ap.parse_args()
    C.ensure_dirs()
    if args.stage == "train_verifier":
        train_verifier()
    elif args.stage == "apply_verifier":
        apply_verifier(args.quads, args.image_index, args.out)
    elif args.stage == "bridge":
        fit_bridge(args.quads, args.audit)
    else:
        apply_weights(args.quads, args.out)


if __name__ == "__main__":
    main()
