import io
import json
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Plasma Cutting Assistant",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1rem; max-width: 1200px;}
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 18px; color: #666; margin-bottom: 24px;}
.card {padding: 16px; border-radius: 18px; background: #f7f7f7; border: 1px solid #e5e5e5;}
.recommendation-box {padding: 16px; border-radius: 18px; background: #fff7e6; border: 1px solid #ffd591; margin-bottom: 12px;}
.stButton > button {width: 100%; border-radius: 12px; min-height: 44px;}
@media (max-width: 768px) {
    .main-title {font-size: 30px;}
    .subtitle {font-size: 15px;}
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔥 Plasma Cutting Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Cloud-based CNC plasma cutting calibration and learning platform</div>',
    unsafe_allow_html=True,
)

PARAMETER_NAMES = [
    "Current [A]", "Nozzle [mm]", "Speed [mm/min]", "Arc voltage [V]",
    "Cut height [mm]", "Pierce height [mm]", "Pierce delay [s]",
    "Kerf [mm]", "THC mode", "Gas / assist", "IHS setting",
]

DEFAULT_PARAMETERS = {
    "Black Steel": {
        "1.5": [35, 1.1, 3800, 110, 1.5, 3.0, 0.25, 1.20, "OFF / minimal", "Air", 0.25],
        "2.0": [40, 1.1, 3300, 112, 1.5, 3.0, 0.30, 1.25, "OFF / minimal", "Air", 0.28],
        "3.0": [45, 1.1, 2400, 113, 1.5, 3.0, 0.45, 1.35, "ON lines / OFF holes", "Air", 0.30],
        "4.0": [55, 1.3, 2100, 116, 1.5, 3.5, 0.50, 1.50, "ON lines / OFF holes", "Air", 0.32],
        "5.0": [60, 1.3, 1800, 118, 1.5, 3.5, 0.55, 1.60, "ON lines / OFF holes", "Air", 0.33],
        "6.0": [65, 1.5, 1500, 114, 1.5, 3.5, 0.60, 1.75, "ON lines / OFF holes", "Air", 0.35],
        "8.0": [80, 1.5, 1200, 125, 1.8, 4.0, 0.80, 1.90, "ON lines / OFF holes", "Air", 0.38],
        "10.0": [90, 1.5, 1000, 128, 1.8, 4.0, 0.90, 2.10, "ON", "Air", 0.40],
        "12.0": [100, 1.7, 850, 132, 2.0, 4.5, 1.00, 2.30, "ON", "Air", 0.45],
    },
    "Galvanized Steel": {
        "1.5": [35, 1.1, 3600, 110, 1.5, 3.0, 0.25, 1.20, "OFF / minimal", "Air", 0.25],
        "2.0": [40, 1.1, 3100, 112, 1.5, 3.0, 0.30, 1.25, "OFF / minimal", "Air", 0.28],
        "3.0": [45, 1.1, 2300, 112, 1.5, 3.0, 0.45, 1.35, "ON lines / OFF holes", "Air", 0.30],
        "4.0": [55, 1.3, 2000, 115, 1.5, 3.5, 0.50, 1.50, "ON lines / OFF holes", "Air", 0.32],
        "5.0": [60, 1.3, 1700, 117, 1.5, 3.5, 0.55, 1.60, "ON lines / OFF holes", "Air", 0.33],
        "6.0": [65, 1.5, 1400, 114, 1.5, 3.5, 0.60, 1.75, "ON lines / OFF holes", "Air", 0.35],
        "8.0": [80, 1.5, 1150, 124, 1.8, 4.0, 0.80, 1.90, "ON lines / OFF holes", "Air", 0.38],
        "10.0": [90, 1.5, 950, 127, 1.8, 4.0, 0.90, 2.10, "ON", "Air", 0.40],
        "12.0": [100, 1.7, 800, 131, 2.0, 4.5, 1.00, 2.30, "ON", "Air", 0.45],
    },
    "Stainless Steel": {
        "1.5": [35, 1.1, 3000, 112, 1.5, 3.0, 0.30, 1.25, "OFF / minimal", "Air / Nitrogen optional", 0.25],
        "2.0": [40, 1.1, 2600, 114, 1.5, 3.0, 0.35, 1.30, "OFF / minimal", "Air / Nitrogen optional", 0.28],
        "3.0": [50, 1.1, 1900, 116, 1.5, 3.5, 0.50, 1.45, "ON lines / OFF holes", "Air / Nitrogen optional", 0.30],
        "4.0": [60, 1.3, 1600, 118, 1.5, 3.5, 0.55, 1.60, "ON lines / OFF holes", "Air / Nitrogen optional", 0.32],
        "5.0": [70, 1.5, 1300, 120, 1.8, 4.0, 0.65, 1.80, "ON lines / OFF holes", "Air / Nitrogen optional", 0.35],
        "6.0": [80, 1.5, 1100, 123, 1.8, 4.0, 0.75, 1.95, "ON lines / OFF holes", "Air / Nitrogen optional", 0.38],
        "8.0": [90, 1.5, 850, 126, 1.8, 4.0, 0.90, 2.15, "ON", "Air / Nitrogen optional", 0.40],
        "10.0": [100, 1.7, 700, 130, 2.0, 4.5, 1.10, 2.35, "ON", "Air / Nitrogen optional", 0.45],
        "12.0": [100, 1.7, 600, 134, 2.0, 4.5, 1.25, 2.55, "ON", "Air / Nitrogen optional", 0.48],
    },
    "Aluminum": {
        "1.5": [35, 1.1, 4200, 110, 1.5, 3.0, 0.25, 1.25, "OFF / minimal", "Air / Nitrogen optional", 0.25],
        "2.0": [40, 1.1, 3600, 112, 1.5, 3.0, 0.30, 1.30, "OFF / minimal", "Air / Nitrogen optional", 0.28],
        "3.0": [50, 1.1, 2700, 114, 1.5, 3.5, 0.40, 1.45, "ON lines / OFF holes", "Air / Nitrogen optional", 0.30],
        "4.0": [60, 1.3, 2200, 116, 1.5, 3.5, 0.50, 1.60, "ON lines / OFF holes", "Air / Nitrogen optional", 0.32],
        "5.0": [70, 1.5, 1800, 118, 1.8, 4.0, 0.60, 1.80, "ON lines / OFF holes", "Air / Nitrogen optional", 0.35],
        "6.0": [80, 1.5, 1500, 120, 1.8, 4.0, 0.70, 1.95, "ON lines / OFF holes", "Air / Nitrogen optional", 0.38],
        "8.0": [90, 1.5, 1200, 123, 1.8, 4.0, 0.85, 2.15, "ON", "Air / Nitrogen optional", 0.40],
        "10.0": [100, 1.7, 950, 126, 2.0, 4.5, 1.00, 2.35, "ON", "Air / Nitrogen optional", 0.45],
        "12.0": [100, 1.7, 750, 130, 2.0, 4.5, 1.20, 2.55, "ON", "Air / Nitrogen optional", 0.48],
    },
}

MACHINE_PROFILES = {
    "Rayline CNC + P80 + F1620 V3 THC": {"voltage_offset": -1, "speed_factor": 1.00, "kerf_offset": 0.10},
    "Rayline CNC + P80 + generic THC": {"voltage_offset": -1, "speed_factor": 1.00, "kerf_offset": 0.10},
    "Generic CNC plasma + P80": {"voltage_offset": 0, "speed_factor": 1.00, "kerf_offset": 0.00},
    "Hypertherm-style system": {"voltage_offset": 0, "speed_factor": 1.10, "kerf_offset": -0.10},
    "Custom machine": {"voltage_offset": 0, "speed_factor": 1.00, "kerf_offset": 0.00},
}

SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]

REST_URL = f"{SUPABASE_URL}/rest/v1"
STORAGE_URL = f"{SUPABASE_URL}/storage/v1"

HEADERS_JSON = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def request_or_error(method, url, **kwargs):
    res = requests.request(method, url, timeout=30, **kwargs)
    if not res.ok:
        st.error(f"Supabase error {res.status_code}: {res.text}")
        res.raise_for_status()
    if res.text:
        return res.json()
    return None


def sort_thickness_values(values):
    return sorted(values, key=lambda x: float(x))


def param_row_to_list(row):
    return [
        row["current_a"], row["nozzle_mm"], row["speed_mm_min"], row["arc_voltage_v"],
        row["cut_height_mm"], row["pierce_height_mm"], row["pierce_delay_s"],
        row["kerf_mm"], row["thc_mode"], row["gas_assist"], row["ihs_setting"],
    ]


def param_list_to_row(material, thickness, values):
    return {
        "material": material,
        "thickness": str(thickness),
        "current_a": values[0],
        "nozzle_mm": values[1],
        "speed_mm_min": values[2],
        "arc_voltage_v": values[3],
        "cut_height_mm": values[4],
        "pierce_height_mm": values[5],
        "pierce_delay_s": values[6],
        "kerf_mm": values[7],
        "thc_mode": values[8],
        "gas_assist": values[9],
        "ihs_setting": values[10],
    }


def seed_default_parameters_if_empty():
    rows = request_or_error("GET", f"{REST_URL}/parameters?select=id&limit=1", headers=HEADERS_JSON)
    if rows:
        return

    payload = []
    for material, thicknesses in DEFAULT_PARAMETERS.items():
        for thickness, values in thicknesses.items():
            payload.append(param_list_to_row(material, thickness, values))

    request_or_error(
        "POST",
        f"{REST_URL}/parameters",
        headers={**HEADERS_JSON, "Prefer": "return=minimal"},
        data=json.dumps(payload),
    )


@st.cache_data(ttl=30)
def load_parameters_cloud():
    rows = request_or_error(
        "GET",
        f"{REST_URL}/parameters?select=*&order=material.asc,thickness.asc",
        headers=HEADERS_JSON,
    )
    database = {}
    for row in rows:
        material = row["material"]
        thickness = row["thickness"]
        database.setdefault(material, {})
        database[material][thickness] = param_row_to_list(row)
    return database


def update_parameter_cloud(material, thickness, values):
    payload = param_list_to_row(material, thickness, values)
    request_or_error(
        "POST",
        f"{REST_URL}/parameters?on_conflict=material,thickness",
        headers={**HEADERS_JSON, "Prefer": "resolution=merge-duplicates,return=minimal"},
        data=json.dumps(payload),
    )
    st.cache_data.clear()


def save_cut_history_cloud(record):
    request_or_error(
        "POST",
        f"{REST_URL}/cut_history",
        headers={**HEADERS_JSON, "Prefer": "return=minimal"},
        data=json.dumps(record),
    )


@st.cache_data(ttl=20)
def load_cut_history_cloud():
    rows = request_or_error(
        "GET",
        f"{REST_URL}/cut_history?select=*&order=created_at.desc&limit=500",
        headers=HEADERS_JSON,
    )
    return rows or []


def upload_images_to_cloud(uploaded_images):
    urls = []
    if not uploaded_images:
        return urls

    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)

        path = f"{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4().hex}.jpg"
        upload_url = f"{STORAGE_URL}/object/cut-images/{path}"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "false",
        }

        res = requests.post(upload_url, headers=headers, data=buffer.getvalue(), timeout=60)
        if not res.ok:
            st.error(f"Image upload error {res.status_code}: {res.text}")
            res.raise_for_status()

        urls.append(f"{STORAGE_URL}/object/public/cut-images/{path}")

    return urls


def apply_machine_profile(params, machine_profile):
    corrected = params.copy()
    profile = MACHINE_PROFILES[machine_profile]
    corrected["Arc voltage [V]"] += profile["voltage_offset"]
    corrected["Speed [mm/min]"] = int(corrected["Speed [mm/min]"] * profile["speed_factor"])
    corrected["Kerf [mm]"] = round(corrected["Kerf [mm]"] + profile["kerf_offset"], 2)
    return corrected


def analyze_cut_image(image):
    gray = ImageOps.grayscale(image)
    if gray.width > 600:
        gray = gray.resize((600, int(600 * gray.height / gray.width)))

    arr = np.asarray(gray).astype(float)
    brightness = arr.mean()
    gy, gx = np.gradient(arr)
    edges = np.sqrt(gx**2 + gy**2)
    edge_density = (edges > 35).mean()
    contrast = arr.std()

    left = arr[:, :arr.shape[1] // 2]
    right = np.fliplr(arr[:, -arr.shape[1] // 2:])
    min_width = min(left.shape[1], right.shape[1])
    symmetry = 100 - np.mean(np.abs(left[:, :min_width] - right[:, :min_width])) / 255 * 100

    return {
        "brightness": round(float(brightness), 1),
        "edge_density": round(float(edge_density), 3),
        "contrast": round(float(contrast), 1),
        "symmetry": round(float(symmetry), 1),
    }


def image_suggestions(image_type, analysis, params):
    suggestions = []
    speed = params["Speed [mm/min]"]
    voltage = params["Arc voltage [V]"]
    kerf = params["Kerf [mm]"]

    if analysis["brightness"] < 55 or analysis["brightness"] > 215:
        suggestions.append({"parameter": "Image quality", "change": "Retake photo", "reason": "Lighting is not reliable enough."})

    if image_type == "Bottom dross":
        suggestions.append({
            "parameter": "Speed",
            "change": f"Increase from {speed} to {speed + 150} mm/min" if analysis["edge_density"] > 0.16 else "+100 mm/min only if dross is physically confirmed",
            "reason": "Bottom texture may indicate dross.",
        })

    elif image_type == "Hole":
        if analysis["symmetry"] < 72:
            suggestions.append({"parameter": "THC", "change": "OFF during holes", "reason": "Low symmetry suggests hole distortion."})
            suggestions.append({"parameter": "Hole speed", "change": f"Use about {int(speed * 0.65)} mm/min", "reason": "Hole cuts often need 60–70% speed."})
        if analysis["edge_density"] > 0.16:
            suggestions.append({"parameter": "Pierce / lead-in", "change": "Verify pierce delay and lead-in", "reason": "High edge density around hole."})

    elif image_type == "Cut edge":
        if analysis["edge_density"] > 0.16:
            suggestions.append({"parameter": "Speed / consumables", "change": "+100 mm/min and inspect nozzle/electrode", "reason": "Rough edge texture."})
        if analysis["contrast"] > 65:
            suggestions.append({"parameter": "Cut height", "change": "Verify actual torch height", "reason": "High contrast can indicate uneven cut face."})

    elif image_type == "Top surface":
        if analysis["edge_density"] > 0.14:
            suggestions.append({"parameter": "Pierce height / delay", "change": "Check pierce height and pierce delay", "reason": "Top spatter/noise."})
            suggestions.append({"parameter": "Arc voltage", "change": f"Try {voltage - 2}V only if torch is high", "reason": "Top spatter can relate to height."})

    elif image_type == "Dimensional test":
        suggestions.append({"parameter": "Kerf", "change": f"Current kerf: {kerf} mm. Adjust after measurement.", "reason": "Need measured dimensional error."})

    return suggestions


def manual_suggestions(params, dross, cut_angle, hole_quality, arc_stability):
    suggestions = []
    speed = params["Speed [mm/min]"]
    voltage = params["Arc voltage [V]"]
    kerf = params["Kerf [mm]"]

    if dross == "Heavy bottom dross":
        suggestions.append({"parameter": "Speed", "change": f"Increase from {speed} to {speed + 150} mm/min", "reason": "Usually too slow."})
    if dross == "Hard dross":
        suggestions.append({"parameter": "Speed / height", "change": "+100 mm/min and verify actual height", "reason": "May be slow speed or wrong height."})
    if dross == "Top spatter":
        suggestions.append({"parameter": "Arc voltage / height", "change": f"Try {voltage - 2}V if torch is high", "reason": "May indicate excessive height."})
    if cut_angle == "Positive angle":
        suggestions.append({"parameter": "Speed / height", "change": "Reduce speed slightly or lower cut height", "reason": "Positive bevel may mean too fast/high."})
    if cut_angle == "Negative angle":
        suggestions.append({"parameter": "Speed", "change": f"Increase to {speed + 100} mm/min", "reason": "Negative bevel often means too slow."})
    if hole_quality == "Oval":
        suggestions.append({"parameter": "THC", "change": "OFF during holes", "reason": "THC movement can distort holes."})
        suggestions.append({"parameter": "Hole speed", "change": f"Use about {int(speed * 0.65)} mm/min", "reason": "60–70% speed is common for holes."})
    if hole_quality == "Too large":
        suggestions.append({"parameter": "Kerf", "change": f"Reduce from {kerf} to {round(kerf - 0.05, 2)} mm", "reason": "Kerf compensation may be too high."})
    if hole_quality == "Rough":
        suggestions.append({"parameter": "Hole path", "change": "Reduce hole speed, THC OFF, improve lead-in", "reason": "Rough holes are speed/height sensitive."})
    if arc_stability == "Unstable":
        suggestions.append({"parameter": "Process stability", "change": "Check air, grounding, moisture, consumables", "reason": "Fix stability before tuning."})

    return suggestions


def history_to_dataframe(history):
    rows = []
    for record in history:
        p = record.get("parameters", {}) or {}
        rows.append({
            "created_at": record.get("created_at"),
            "material": record.get("material"),
            "thickness": record.get("thickness"),
            "machine_profile": record.get("machine_profile"),
            "score": record.get("score"),
            "speed": p.get("Speed [mm/min]"),
            "voltage": p.get("Arc voltage [V]"),
            "kerf": p.get("Kerf [mm]"),
            "current": p.get("Current [A]"),
            "nozzle": p.get("Nozzle [mm]"),
            "dross": record.get("dross"),
            "cut_angle": record.get("cut_angle"),
            "hole_quality": record.get("hole_quality"),
            "arc_stability": record.get("arc_stability"),
        })
    return pd.DataFrame(rows)


seed_default_parameters_if_empty()
DATABASE = load_parameters_cloud()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚙️ Parameters", "📷 Feedback", "🧠 Recommendation", "✏️ Edit", "📚 History", "📊 Analytics"
])

with tab1:
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        material = st.selectbox("Material", list(DATABASE.keys()))

        thickness_options = sort_thickness_values(DATABASE[material].keys())
        thickness = st.selectbox("Thickness [mm]", thickness_options)

        machine_profile = st.selectbox("Machine / controller", list(MACHINE_PROFILES.keys()))

        st.markdown("</div>", unsafe_allow_html=True)

    raw_params = dict(zip(PARAMETER_NAMES, DATABASE[material][thickness]))
    params = apply_machine_profile(raw_params, machine_profile)

    with right:
        st.subheader("Suggested machine-adjusted baseline")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current", f"{params['Current [A]']} A")
        c2.metric("Nozzle", f"{params['Nozzle [mm]']} mm")
        c3.metric("Speed", f"{params['Speed [mm/min]']} mm/min")
        c4.metric("Voltage", f"{params['Arc voltage [V]']} V")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Kerf", f"{params['Kerf [mm]']} mm")
        c6.metric("Cut height", f"{params['Cut height [mm]']} mm")
        c7.metric("Pierce delay", f"{params['Pierce delay [s]']} s")
        c8.metric("IHS", str(params["IHS setting"]))

        st.table({"Parameter": list(params.keys()), "Value": list(params.values())})
        st.warning("Starting values only. Calibrate per machine.")

with tab2:
    st.subheader("Cut images + manual feedback")

    image_type = st.selectbox("Image type", ["Cut edge", "Bottom dross", "Hole", "Top surface", "Dimensional test"])
    uploaded_images = st.file_uploader("Upload cut images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    image_based_suggestions = []

    if uploaded_images:
        cols = st.columns(2)
        for i, uploaded_image in enumerate(uploaded_images):
            image = Image.open(uploaded_image).convert("RGB")
            analysis = analyze_cut_image(image)
            sugg = image_suggestions(image_type, analysis, params)
            image_based_suggestions.extend(sugg)

            with cols[i % 2]:
                st.image(image, caption=uploaded_image.name, use_container_width=True)
                m1, m2 = st.columns(2)
                m1.metric("Edge density", analysis["edge_density"])
                m2.metric("Symmetry", f"{analysis['symmetry']}%")
                for s in sugg:
                    st.info(f"{s['parameter']}: {s['change']}\n\nReason: {s['reason']}")

    st.subheader("Manual quality feedback")

    c1, c2 = st.columns(2)
    with c1:
        dross = st.selectbox("Dross", ["None / light", "Heavy bottom dross", "Hard dross", "Top spatter"])
        hole_quality = st.selectbox("Hole quality", ["Not tested", "Good", "Oval", "Too large", "Rough"])
    with c2:
        cut_angle = st.selectbox("Cut angle", ["Good / straight", "Positive angle", "Negative angle"])
        arc_stability = st.selectbox("Arc stability", ["Stable", "Unstable"])

    score = st.slider("Result score", 1, 10, 5)
    notes = st.text_area("Notes")

with tab3:
    st.subheader("Recommended next cut")

    all_suggestions = manual_suggestions(params, dross, cut_angle, hole_quality, arc_stability) + image_based_suggestions

    st.code(f"""Material: {material}
Thickness: {thickness} mm
Machine: {machine_profile}
Speed: {params['Speed [mm/min]']} mm/min
Voltage: {params['Arc voltage [V]']} V
Kerf: {params['Kerf [mm]']} mm
THC: {params['THC mode']}
IHS: {params['IHS setting']}
Gas: {params['Gas / assist']}
""")

    if st.button("Generate recommendation"):
        if not all_suggestions:
            st.info("No strong correction detected. Change only one parameter at a time.")
        else:
            seen = set()
            for s in all_suggestions:
                key = (s["parameter"], s["change"])
                if key in seen:
                    continue
                seen.add(key)
                st.markdown(
                    f"""<div class="recommendation-box">
                    <b>{s['parameter']}</b><br>
                    Recommended change: {s['change']}<br>
                    Reason: {s['reason']}
                    </div>""",
                    unsafe_allow_html=True,
                )

    if st.button("Save current test to cloud"):
        image_urls = upload_images_to_cloud(uploaded_images)

        record = {
            "material": material,
            "thickness": thickness,
            "machine_profile": machine_profile,
            "parameters": params,
            "image_type": image_type,
            "image_urls": image_urls,
            "dross": dross,
            "cut_angle": cut_angle,
            "hole_quality": hole_quality,
            "arc_stability": arc_stability,
            "score": int(score),
            "notes": notes,
            "recommendations": all_suggestions,
        }

        save_cut_history_cloud(record)
        st.cache_data.clear()
        st.success("Saved to Supabase cloud, including images.")

with tab4:
    st.subheader("Edit cloud parameters")

    edit_material = st.selectbox("Material to edit", list(DATABASE.keys()), key="edit_material")

    edit_thickness_options = sort_thickness_values(DATABASE[edit_material].keys())
    edit_thickness = st.selectbox("Thickness to edit", edit_thickness_options, key="edit_thickness")

    current_params = dict(zip(PARAMETER_NAMES, DATABASE[edit_material][edit_thickness]))
    updated = {}

    for name in PARAMETER_NAMES:
        value = current_params[name]
        if isinstance(value, (int, float)):
            updated[name] = st.number_input(name, value=float(value), key=f"{edit_material}_{edit_thickness}_{name}")
        else:
            updated[name] = st.text_input(name, value=str(value), key=f"{edit_material}_{edit_thickness}_{name}")

    if st.button("Save parameter changes to cloud"):
        values = [updated[name] for name in PARAMETER_NAMES]
        update_parameter_cloud(edit_material, edit_thickness, values)
        st.success("Cloud parameters updated. Refresh if needed.")

with tab5:
    st.subheader("Cloud cut history")

    history = load_cut_history_cloud()

    if not history:
        st.info("No cloud history yet.")
    else:
        for i, record in enumerate(history, start=1):
            with st.expander(
                f"{i}. {record['material']} | {record['thickness']} mm | Score {record['score']} | {record['created_at'][:19]}"
            ):
                st.write("Machine:", record["machine_profile"])
                st.write("Image type:", record["image_type"])
                st.write("Dross:", record["dross"])
                st.write("Cut angle:", record["cut_angle"])
                st.write("Hole quality:", record["hole_quality"])
                st.write("Arc stability:", record["arc_stability"])
                st.write("Notes:", record["notes"])

                st.subheader("Parameters")
                st.json(record["parameters"])

                if record.get("recommendations"):
                    st.subheader("Recommendations")
                    for rec in record["recommendations"]:
                        st.info(f"{rec['parameter']}: {rec['change']}\n\nReason: {rec['reason']}")

                if record.get("image_urls"):
                    st.subheader("Images")
                    cols = st.columns(2)
                    for j, url in enumerate(record["image_urls"]):
                        with cols[j % 2]:
                            st.image(url, use_container_width=True)

with tab6:
    st.subheader("Cut analytics")

    history = load_cut_history_cloud()

    if not history:
        st.info("No cut history yet. Save a few tests first.")
    else:
        df = history_to_dataframe(history)

        st.write("Saved cuts overview")
        st.dataframe(df, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total tests", len(df))
        c2.metric("Average score", round(df["score"].mean(), 2))
        c3.metric("Best score", int(df["score"].max()))

        material_filter = st.selectbox(
            "Filter material",
            ["All"] + sorted(df["material"].dropna().unique().tolist())
        )

        if material_filter != "All":
            df = df[df["material"] == material_filter]

        thickness_filter = st.selectbox(
            "Filter thickness",
            ["All"] + sort_thickness_values(df["thickness"].dropna().unique().tolist())
        )

        if thickness_filter != "All":
            df = df[df["thickness"] == thickness_filter]

        st.divider()

        st.subheader("Best cuts")
        best_df = df.sort_values("score", ascending=False).head(10)
        st.dataframe(best_df, use_container_width=True)

        st.subheader("Average best parameters")

        high_score_df = df[df["score"] >= 8]

        if high_score_df.empty:
            st.info("No high-score cuts yet. Save cuts with score 8+ to build validated parameters.")
        else:
            avg_speed = round(high_score_df["speed"].mean(), 0)
            avg_voltage = round(high_score_df["voltage"].mean(), 1)
            avg_kerf = round(high_score_df["kerf"].mean(), 2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg speed", f"{avg_speed} mm/min")
            c2.metric("Avg voltage", f"{avg_voltage} V")
            c3.metric("Avg kerf", f"{avg_kerf} mm")

        st.subheader("Learning insight")

        if len(df) < 3:
            st.info("Save at least 3 tests for this setup to generate meaningful learning insights.")
        else:
            best = df.sort_values("score", ascending=False).iloc[0]
            st.success(
                f"Best recorded setup: {best['material']} {best['thickness']} mm, "
                f"{best['speed']} mm/min, {best['voltage']} V, score {best['score']}."
            )