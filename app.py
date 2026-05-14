import io
import json
import uuid
import ezdxf
import math
from datetime import datetime
from streamlit_cookies_manager import EncryptedCookieManager

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

cookies = EncryptedCookieManager(
    prefix="plasma_assistant",
    password=st.secrets["COOKIE_PASSWORD"],
)

if not cookies.ready():
    st.stop()

st.markdown("""
<style>
.block-container {padding-top: 1rem; max-width: 1200px;}
.main-title {font-size: 42px; font-weight: 800; margin-bottom: 0;}
.subtitle {font-size: 18px; color: #666; margin-bottom: 24px;}
.card {padding: 16px; border-radius: 18px; background: #f7f7f7; border: 1px solid #e5e5e5; margin-bottom: 12px;}
.recommendation-box {padding: 16px; border-radius: 18px; background: #fff7e6; border: 1px solid #ffd591; margin-bottom: 12px;}
.stButton > button {width: 100%; border-radius: 12px; min-height: 44px;}
@media (max-width: 768px) {
    .main-title {font-size: 30px;}
    .subtitle {font-size: 15px;}
}
</style>
""", unsafe_allow_html=True)


PARAMETER_NAMES = [
    "Current [A]", "Nozzle [mm]", "Speed [mm/min]", "Arc voltage [V]",
    "Cut height [mm]", "Pierce height [mm]", "Pierce delay [s]",
    "Kerf [mm]", "THC mode", "Gas / assist", "IHS setting",
]

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
AUTH_URL = f"{SUPABASE_URL}/auth/v1"
STORAGE_URL = f"{SUPABASE_URL}/storage/v1"


def base_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def auth_headers():
    token = st.session_state.get("access_token")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def request_or_error(method, url, headers=None, **kwargs):
    res = requests.request(method, url, headers=headers or base_headers(), timeout=30, **kwargs)
    if not res.ok:
        st.error(f"Supabase error {res.status_code}: {res.text}")
        res.raise_for_status()
    if res.text:
        return res.json()
    return None


def sort_thickness_values(values):
    return sorted(values, key=lambda x: float(x))


def login_user(email, password, remember_me=False):
    payload = {"email": email, "password": password}
    res = requests.post(
        f"{AUTH_URL}/token?grant_type=password",
        headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )

    if not res.ok:
        st.error("Login failed. Check email/password.")
        return False

    data = res.json()

    st.session_state["access_token"] = data["access_token"]
    st.session_state["refresh_token"] = data.get("refresh_token")
    st.session_state["user"] = data["user"]
    st.session_state["user_id"] = data["user"]["id"]
    st.session_state["email"] = data["user"]["email"]

    if remember_me:
        cookies["access_token"] = data["access_token"]
        cookies["refresh_token"] = data.get("refresh_token", "")
        cookies["user_id"] = data["user"]["id"]
        cookies["email"] = data["user"]["email"]
        cookies.save()

    return True


def register_user(email, password):
    payload = {"email": email, "password": password}
    res = requests.post(
        f"{AUTH_URL}/signup",
        headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=30,
    )

    if not res.ok:
        st.error(f"Registration failed: {res.text}")
        return False

    st.success("User created. Now log in.")
    return True


def logout_user():
    for key in ["access_token", "refresh_token", "user_id", "email"]:
        cookies[key] = ""
    cookies.save()
    st.session_state.clear()
    st.rerun()

def restore_session_from_cookies():
    if "access_token" in st.session_state:
        return

    access_token = cookies.get("access_token")
    refresh_token = cookies.get("refresh_token")
    user_id = cookies.get("user_id")
    email = cookies.get("email")

    if access_token and user_id and email:
        st.session_state["access_token"] = access_token
        st.session_state["refresh_token"] = refresh_token
        st.session_state["user_id"] = user_id
        st.session_state["email"] = email

def show_login_screen():
    st.markdown('<div class="main-title">🔥 Plasma Cutting Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Login to your CNC plasma calibration workspace</div>', unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
     email = st.text_input("Email", key="login_email")
     password = st.text_input("Password", type="password", key="login_password")
     remember_me = st.checkbox("Keep me logged in", value=True)

     if st.button("Login"):
        if login_user(email, password, remember_me):
            st.rerun()

    with register_tab:
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Password", type="password", key="register_password")
        if st.button("Create account"):
            register_user(new_email, new_password)


def param_row_to_list(row):
    return [
        row["current_a"], row["nozzle_mm"], row["speed_mm_min"], row["arc_voltage_v"],
        row["cut_height_mm"], row["pierce_height_mm"], row["pierce_delay_s"],
        row["kerf_mm"], row["thc_mode"], row["gas_assist"], row["ihs_setting"],
    ]


@st.cache_data(ttl=30)
def load_parameters_cloud():
    rows = request_or_error(
        "GET",
        f"{REST_URL}/parameters?select=*&order=material.asc,thickness.asc",
        headers=base_headers(),
    )
    database = {}
    for row in rows:
        material = row["material"]
        thickness = row["thickness"]
        database.setdefault(material, {})
        database[material][thickness] = param_row_to_list(row)
    return database


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


def update_parameter_cloud(material, thickness, values):
    payload = param_list_to_row(material, thickness, values)
    request_or_error(
        "POST",
        f"{REST_URL}/parameters?on_conflict=material,thickness",
        headers={**base_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
        data=json.dumps(payload),
    )
    st.cache_data.clear()


@st.cache_data(ttl=20)
def load_cut_history_cloud(user_token):
    rows = request_or_error(
        "GET",
        f"{REST_URL}/cut_history?select=*&order=created_at.desc&limit=500",
        headers=auth_headers(),
    )
    return rows or []


def save_cut_history_cloud(record):
    request_or_error(
        "POST",
        f"{REST_URL}/cut_history",
        headers={**auth_headers(), "Prefer": "return=minimal"},
        data=json.dumps(record),
    )


@st.cache_data(ttl=30)
def load_pricing_table(user_token):
    rows = request_or_error(
        "GET",
        f"{REST_URL}/pricing_table?select=*&order=thickness.asc",
        headers=auth_headers(),
    )
    return rows or []


def upsert_pricing_row(row):
    request_or_error(
        "POST",
        f"{REST_URL}/pricing_table?on_conflict=user_id,thickness",
        headers={**auth_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
        data=json.dumps(row),
    )
    st.cache_data.clear()


def upload_images_to_cloud(uploaded_images):
    urls = []
    if not uploaded_images:
        return urls

    for uploaded_image in uploaded_images:
        image = Image.open(uploaded_image).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)

        path = f"{st.session_state['user_id']}/{datetime.now().strftime('%Y%m%d')}/{uuid.uuid4().hex}.jpg"
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

restore_session_from_cookies()

if "access_token" not in st.session_state:
    show_login_screen()
    st.stop()


st.markdown('<div class="main-title">🔥 Plasma Cutting Assistant</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Logged in as {st.session_state.get("email")}</div>',
    unsafe_allow_html=True,
)

top1, top2 = st.columns([4, 1])
with top2:
    if st.button("Logout"):
        logout_user()


DATABASE = load_parameters_cloud()
current_token = st.session_state["access_token"]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "⚙️ Parameters",
    "📷 Feedback",
    "🧠 Recommendation",
    "✏️ Edit",
    "📚 History",
    "📊 Analytics",
    "💰 Pricing",
])

def distance_2d(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def arc_length(radius, start_angle, end_angle):
    angle = abs(end_angle - start_angle)

    if angle > 360:
        angle = angle % 360

    if angle > 180:
        angle = 360 - angle

    return 2 * math.pi * radius * (angle / 360)


def analyze_dxf(uploaded_file):
    doc = ezdxf.read(uploaded_file)
    msp = doc.modelspace()

    total_length_mm = 0
    entity_count = 0
    pierce_estimate = 0

    for entity in msp:
        entity_type = entity.dxftype()

        try:
            if entity_type == "LINE":
                total_length_mm += distance_2d(entity.dxf.start, entity.dxf.end)
                entity_count += 1

            elif entity_type == "CIRCLE":
                total_length_mm += 2 * math.pi * entity.dxf.radius
                entity_count += 1

            elif entity_type == "ARC":
                total_length_mm += arc_length(
                    entity.dxf.radius,
                    entity.dxf.start_angle,
                    entity.dxf.end_angle,
                )
                entity_count += 1

            elif entity_type == "LWPOLYLINE":
                points = list(entity.get_points("xy"))
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        total_length_mm += distance_2d(points[i], points[i + 1])

                    if entity.closed:
                        total_length_mm += distance_2d(points[-1], points[0])

                    entity_count += 1

            elif entity_type == "POLYLINE":
                points = [v.dxf.location for v in entity.vertices]
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        total_length_mm += distance_2d(points[i], points[i + 1])

                    if entity.is_closed:
                        total_length_mm += distance_2d(points[-1], points[0])

                    entity_count += 1

        except Exception:
            continue

    pierce_estimate = entity_count

    return {
        "total_length_mm": round(total_length_mm, 2),
        "total_length_m": round(total_length_mm / 1000, 3),
        "entity_count": entity_count,
        "pierce_estimate": pierce_estimate,
    }

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
            "user_id": st.session_state["user_id"],
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
        st.success("Saved to your cloud history, including images.")


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
    st.subheader("My cut history")

    history = load_cut_history_cloud(current_token)

    if not history:
        st.info("No cut history yet.")
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
    st.subheader("My cut analytics")

    history = load_cut_history_cloud(current_token)

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

        high_score_df = df[df["score"] >= 8]

        st.subheader("Average best parameters")

        if high_score_df.empty:
            st.info("No high-score cuts yet. Save cuts with score 8+ to build validated parameters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg speed", f"{round(high_score_df['speed'].mean(), 0)} mm/min")
            c2.metric("Avg voltage", f"{round(high_score_df['voltage'].mean(), 1)} V")
            c3.metric("Avg kerf", f"{round(high_score_df['kerf'].mean(), 2)} mm")


with tab7:
    st.subheader("Pricing calculator")

    pricing_rows = load_pricing_table(current_token)

    if not pricing_rows:
        st.error("Pricing table is empty.")
    else:
        pricing_df = pd.DataFrame(pricing_rows)
        pricing_df["thickness_float"] = pricing_df["thickness"].astype(float)
        pricing_df = pricing_df.sort_values("thickness_float")

        left, right = st.columns([1, 1])

        with left:
            selected_thickness = st.selectbox(
                "Thickness [mm]",
                pricing_df["thickness"].tolist(),
                key="pricing_thickness"
            )

            row = pricing_df[pricing_df["thickness"] == selected_thickness].iloc[0]
            dxf_file = st.file_uploader(
                "Upload DXF for automatic cut length",
                type=["dxf"],
                key="pricing_dxf"
            )

            dxf_result = None

            if dxf_file is not None:
                try:
                    dxf_result = analyze_dxf(dxf_file)

                    st.success("DXF analyzed successfully.")
                    st.metric("DXF cut length", f"{dxf_result['total_length_m']} m")
                    st.metric("Estimated pierces", dxf_result["pierce_estimate"])

                    default_cut_length = float(dxf_result["total_length_m"])
                    default_pierces = int(dxf_result["pierce_estimate"])

                except Exception as e:
                    st.error(f"DXF analysis failed: {e}")
                    default_cut_length = 1.0
                    default_pierces = 1
            else:
                default_cut_length = 1.0
                default_pierces = 1
                
            cut_length_m = st.number_input(
                "Total cutting length [m]",
                min_value=0.0,
                value=default_cut_length,
                step=0.1
            )

            pierces = st.number_input(
                "Number of pierces",
                min_value=0,
                value=default_pierces,
                step=1
            )
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)

            include_file_setup = st.checkbox("Include DXF/file setup", value=True)
            include_finish = st.checkbox("Include finishing / dross cleanup", value=False)
            urgent = st.checkbox("Urgent job factor", value=False)

        with right:
            price_per_meter = float(row["price_per_meter"])
            price_per_pierce = float(row["price_per_pierce"])
            minimum_price = float(row.get("minimum_price") or 0)
            file_setup_price = float(row.get("file_setup_price") or 0)
            finish_price = float(row.get("finish_price") or 0)
            urgent_factor = float(row.get("urgent_factor") or 1)

            base_cut_price = cut_length_m * price_per_meter
            pierce_price = pierces * price_per_pierce
            setup_price = file_setup_price if include_file_setup else 0
            finishing_price = finish_price if include_finish else 0

            subtotal_single = base_cut_price + pierce_price + setup_price + finishing_price
            subtotal_all = subtotal_single * quantity

            if urgent:
                subtotal_all *= urgent_factor

            final_price = max(subtotal_all, minimum_price)

            st.metric("Cut length price", f"₪{base_cut_price:.2f}")
            st.metric("Pierce price", f"₪{pierce_price:.2f}")
            st.metric("Setup price", f"₪{setup_price:.2f}")
            st.metric("Final price", f"₪{final_price:.2f}")

            st.info(
                f"Price basis: ₪{price_per_meter}/m, ₪{price_per_pierce}/pierce, minimum ₪{minimum_price}"
            )

        st.divider()
        st.subheader("Pricing table")

        st.dataframe(
            pricing_df[
                ["thickness", "price_per_meter", "price_per_pierce", "typical_speed_mm_min", "minimum_price", "file_setup_price"]
            ],
            use_container_width=True
        )

        st.subheader("Edit my pricing")

        edit_price_thickness = st.selectbox(
            "Thickness to edit",
            pricing_df["thickness"].tolist(),
            key="edit_price_thickness"
        )

        edit_row = pricing_df[pricing_df["thickness"] == edit_price_thickness].iloc[0]

        new_price_per_meter = st.number_input(
            "Price per meter",
            value=float(edit_row["price_per_meter"]),
            key="new_price_per_meter"
        )

        new_price_per_pierce = st.number_input(
            "Price per pierce",
            value=float(edit_row["price_per_pierce"]),
            key="new_price_per_pierce"
        )

        new_minimum_price = st.number_input(
            "Minimum price",
            value=float(edit_row.get("minimum_price") or 0),
            key="new_minimum_price"
        )

        new_file_setup_price = st.number_input(
            "File setup price",
            value=float(edit_row.get("file_setup_price") or 0),
            key="new_file_setup_price"
        )

        if st.button("Save my pricing"):
            payload = {
                "user_id": st.session_state["user_id"],
                "thickness": edit_price_thickness,
                "price_per_meter": new_price_per_meter,
                "price_per_pierce": new_price_per_pierce,
                "typical_speed_mm_min": float(edit_row.get("typical_speed_mm_min") or 0),
                "minimum_price": new_minimum_price,
                "file_setup_price": new_file_setup_price,
                "finish_price": float(edit_row.get("finish_price") or 0),
                "urgent_factor": float(edit_row.get("urgent_factor") or 1),
            }

            upsert_pricing_row(payload)
            st.success("Personal pricing updated.")