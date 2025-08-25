#!/usr/bin/env python3
import os, csv
from typing import List, Dict, Tuple, Any
from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
UPLOADS = os.path.join(BASE_DIR, "uploads")
DATA_CSV = os.path.join(BASE_DIR, "data", "outfits_unisex.csv")
os.makedirs(UPLOADS, exist_ok=True)

# ------------------ Utilities: dataset ------------------
def read_dataset(path: str) -> List[Dict[str,str]]:
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # normalize some fields
            r = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            # ensure lists
            r["color"] = [c.strip().lower() for c in r.get("color","").split("/") if c.strip()] or ["any"]
            r["gender"] = r.get("gender","unisex").lower() or "unisex"
            r["category"] = r.get("category","").strip().lower()
            r["occasion"] = r.get("occasion","any").strip().lower() or "any"
            r["recommended_for_height"] = r.get("recommended_for_height","all")
            rows.append(r)
    return rows

DATA = read_dataset(DATA_CSV)

# ------------------ Utilities: color detection ------------------
BASIC_COLOR_NAMES = {
    "white": np.array([245,245,245]),
    "black": np.array([20,20,20]),
    "gray": np.array([128,128,128]),
    "navy": np.array([20,40,80]),
    "blue": np.array([60,120,220]),
    "denim": np.array([30,60,120]),
    "light blue": np.array([180,210,250]),
    "red": np.array([200,40,40]),
    "maroon": np.array([128,0,0]),
    "pink": np.array([245,160,200]),
    "beige": np.array([220,210,180]),
    "brown": np.array([120,80,40]),
    "tan": np.array([210,180,140]),
    "olive": np.array([120,130,60]),
    "green": np.array([60,160,80]),
    "teal": np.array([60,140,140]),
    "purple": np.array([120,60,150]),
    "yellow": np.array([240,230,140]),
    "orange": np.array([240,140,50]),
    "khaki": np.array([195,176,145]),
    "charcoal": np.array([54,69,79]),
}

def nearest_color_name(rgb: np.ndarray) -> str:
    best_name, best_d = None, 1e9
    for name, ref in BASIC_COLOR_NAMES.items():
        d = np.linalg.norm(rgb.astype(float) - ref.astype(float))
        if d < best_d:
            best_name, best_d = name, d
    return best_name or "unknown"

def dominant_color_from_pil(img: Image.Image, k: int=3) -> np.ndarray:
    small = img.resize((160,160)).convert("RGB")
    arr = np.array(small).reshape(-1,3).astype(np.float32)
    if cv2 is not None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
        _ret, label, center = cv2.kmeans(arr, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(label.flatten())
        dom = center[counts.argmax()]
        return np.clip(dom, 0, 255).astype(np.uint8)
    # numpy kmeans fallback (small k)
    np.random.seed(1)
    centers = arr[np.random.choice(len(arr), k, replace=False)]
    for _ in range(8):
        dists = np.sqrt(((arr[:,None,:] - centers[None,:,:])**2).sum(axis=2))
        labels = dists.argmin(axis=1)
        new_centers = np.array([arr[labels==j].mean(axis=0) if (labels==j).any() else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers, atol=1.0):
            centers = new_centers; break
        centers = new_centers
    counts = np.bincount(labels, minlength=k)
    dom_idx = counts.argmax()
    return np.clip(centers[dom_idx], 0, 255).astype(np.uint8)

def rgb_to_hsv(rgb: np.ndarray) -> Tuple[float,float,float]:
    r,g,b = rgb/255.0
    mx, mn = max(r,g,b), min(r,g,b)
    diff = mx - mn
    if diff == 0: h = 0
    elif mx == r: h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g: h = (60 * ((b - r) / diff) + 120) % 360
    else: h = (60 * ((r - g) / diff) + 240) % 360
    s = 0 if mx == 0 else diff / mx
    v = mx
    return h,s,v

def normalize_color_name(rgb: np.ndarray, name: str) -> str:
    # map cream/ivory/beige-like to white, reduce minor variations
    h,s,v = rgb_to_hsv(rgb)
    if v > 0.85 and s < 0.18:
        return "white"
    if name in {"beige","tan","khaki"} and v > 0.80 and s < 0.35:
        return "white"
    return name

# ------------------ Rule-based category detection from filename ------------------
FILENAME_HINTS = {
    "saree":"saree","lehenga":"lehenga","kurta":"kurta","kurti":"kurti","shirt":"shirt","tshirt":"t-shirt",
    "t-shirt":"t-shirt","jeans":"jeans","jacket":"jacket","blazer":"blazer","suit":"suit","pants":"trousers",
    "trouser":"trousers","pant":"trousers","skirt":"skirt","dress":"dress","shoe":"shoes","sneaker":"sneakers",
    "loaf":"loafers","boot":"boots","heel":"heels","coat":"coat","hoodie":"hoodie","sweater":"sweater",
    "kurti":"kurti","lehenga":"lehenga","sherwani":"sherwani","dupatta":"dupatta"
}

def guess_category_from_filename(name: str) -> str:
    n = (name or "").lower()
    for k,v in FILENAME_HINTS.items():
        if k in n:
            return v
    return ""

def infer_gender_from_category(cat: str) -> str:
    female_only = {"saree","lehenga","kurti","skirt","dress","heels","dupatta"}
    male_only = {"sherwani"}
    if not cat: return "unisex"
    if cat in female_only: return "female"
    if cat in male_only: return "male"
    return "unisex"

# ------------------ Matching engine ------------------
def match_outfit(category: str, color: str, gender: str, occasion: str, height: str) -> Tuple[Dict[str,str], List[str]]:
    category = (category or "").lower()
    color = (color or "").lower()
    gender = (gender or "unisex").lower()
    occasion = (occasion or "any").lower()

    # ranking: exact category + color + gender + occasion preferred
    candidates = []
    for r in DATA:
        if r["category"] != category:
            continue
        # color match if any color entry matches or 'any'
        if not ("any" in r["color"] or color in r["color"]):
            continue
        # gender check: allow unisex or matching gender
        if not (r["gender"] == "unisex" or gender == "unisex" or r["gender"] == gender):
            continue
        # occasion check: if dataset row has 'any' allow, else require match or be flexible
        if r["occasion"] != "any" and occasion != "any" and r["occasion"] != occasion:
            # allow if r.occasion is generic category like 'casual' or 'smart casual'
            pass
        score = 0
        if color in r["color"]: score += 2.0
        if r["gender"] == gender: score += 0.8
        if r["occasion"] == occasion: score += 0.8
        # height fit
        if height and (height in r.get("recommended_for_height","") or "all" in r.get("recommended_for_height","")):
            score += 0.6
        candidates.append((score, r))
    if not candidates:
        return {}, ["No direct match in the dataset. Try another image or expand CSV."]
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    reasons = [f"Category: {category or 'unknown'} • Color: {color or 'unknown'} • Gender: {gender or 'unisex'}",
               f"Selected row with occasion '{best.get('occasion','any')}' and recommended heights: {best.get('recommended_for_height','all')}"]
    return best, reasons

# ------------------ Flask routes ------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("image")
    height = request.form.get("height_cm","175")
    occasion = request.form.get("occasion","any")
    preferred_gender = (request.form.get("gender","auto") or "auto").lower()

    if not file:
        return jsonify({"ok": False, "error": "No image uploaded."}), 400

    # save
    fname = file.filename
    save_path = os.path.join(UPLOADS, fname)
    file.save(save_path)

    # open image, detect dominant color
    img = Image.open(save_path).convert("RGB")
    dom_rgb = dominant_color_from_pil(img)
    raw_name = nearest_color_name(dom_rgb)
    color_name = normalize_color_name(dom_rgb, raw_name)

    # category heuristic from filename; fallback to "shirt"
    category = guess_category_from_filename(fname) or "shirt"
    # gender
    if preferred_gender == "auto":
        gender = infer_gender_from_category(category)
    else:
        gender = preferred_gender

    # match outfit
    best, reasons = match_outfit(category, color_name, gender, occasion, "")
    if not best:
        return jsonify({"ok": True, "detected": {"category": category, "color_name": color_name, "color_rgb": dom_rgb.tolist(), "gender": gender},
                        "outfit": {}, "reasons": reasons})

    # prepare image URLs (static folder)
    def static_url(path):
        # dataset stores path relative to static folder; make safe
        return url_for('static', filename=path) if path else ""

    outfit = {
        "bottoms": best.get("bottoms",""),
        "layer": best.get("layer",""),
        "shoes": best.get("shoes",""),
        "accessory": best.get("accessory",""),
        "bottoms_img": static_url(best.get("bottoms_img","")),
        "layer_img": static_url(best.get("layer_img","")),
        "shoes_img": static_url(best.get("shoes_img","")),
        "accessory_img": static_url(best.get("accessory_img","")),
        "occasion": best.get("occasion","any"),
        "recommended_for_height": best.get("recommended_for_height","all")
    }

    return jsonify({
        "ok": True,
        "detected": {"category": category, "color_name": color_name, "color_rgb": dom_rgb.tolist(), "gender": gender},
        "outfit": outfit,
        "reasons": reasons,
        "uploaded_img": url_for('static', filename=f"uploads/{fname}")  # helpful if you copy to static/uploads, else frontend uses preview
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    