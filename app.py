import os
import io
import uuid
import logging
from flask import Flask, jsonify, request, send_file, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from supabase import create_client, Client
from google import genai
from google.genai import types
import qrcode

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from .env and api.env
load_dotenv()
load_dotenv("api.env")

# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

# Supabase Credentials
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")

# Supabase Client Initialization
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not found in environment variables.")

# Google AI Studio / Gemini API Client Initialization
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
gemini_client: genai.Client | None = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Google GenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Google GenAI client: {e}")
else:
    logger.warning("GEMINI_API_KEY not found in environment variables.")

# Baseline / Fallback Data for resilient local execution & tests
DEFAULT_MATERIALS = [
    {"category": "Motherboards", "rate_per_kg": 580.0},
    {"category": "Copper", "rate_per_kg": 710.0},
    {"category": "Batteries", "rate_per_kg": 180.0},
    {"category": "Displays", "rate_per_kg": 120.0},
    {"category": "Plastic", "rate_per_kg": 45.0}
]

DEFAULT_RECYCLERS = [
    {
        "id": 1,
        "name": "GreenTech E-Waste Recyclers Pvt Ltd",
        "address": "Plot C-14, TTC Industrial Area, Turbhe, Navi Mumbai",
        "lat": 19.0760,
        "long": 73.0163,
        "accepted_materials": ["Motherboards", "PCB", "Copper", "Batteries"]
    },
    {
        "id": 2,
        "name": "EcoVortex Recovery Hub",
        "address": "Bhiwandi E-Waste Complex, Thane",
        "lat": 19.2967,
        "long": 73.0631,
        "accepted_materials": ["PCB", "Displays", "Plastic"]
    },
    {
        "id": 3,
        "name": "MahaRecycle Industrial Processors",
        "address": "Kurla West Industrial Estate, Mumbai",
        "lat": 19.0688,
        "long": 72.8797,
        "accepted_materials": ["Copper", "Motherboards", "Batteries"]
    },
    {
        "id": 4,
        "name": "CleanPlanet Hazardous Waste Facility",
        "address": "MIDC Phase 2, Chakan, Pune",
        "lat": 18.7606,
        "long": 73.8636,
        "accepted_materials": ["Batteries", "Copper", "Plastic"]
    }
]


def classify_category_with_gemini(photo_bytes: bytes, mime_type: str, candidate_categories: list[str]) -> str:
    """
    Call Google AI Studio Gemini API (google-genai SDK) with the photo
    to classify the material category.
    """
    global gemini_client
    if not gemini_client and GEMINI_API_KEY:
        try:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as err:
            logger.error(f"Error instantiating Gemini client: {err}")

    if not gemini_client:
        logger.warning("Gemini client is unavailable; cannot perform automatic classification.")
        return candidate_categories[0] if candidate_categories else "Motherboards"

    options_text = ", ".join(candidate_categories) if candidate_categories else "Motherboards, Copper, Batteries, Displays, Plastic"
    prompt = (
        f"You are an expert e-waste and scrap metal recycling classification system.\n"
        f"Analyze the provided image and classify the primary material into one of these available categories: [{options_text}].\n"
        f"Return ONLY the exact matched category name from the list. Do not add markdown, quotes, explanations, or extra words."
    )

    models_to_try = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    for model_name in models_to_try:
        try:
            image_part = types.Part.from_bytes(
                data=photo_bytes,
                mime_type=mime_type or "image/jpeg"
            )
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt]
            )
            if response and response.text:
                classified = response.text.strip().strip('"').strip("'").strip("`")
                if "\n" in classified:
                    classified = classified.split("\n")[0].strip()
                logger.info(f"Gemini ({model_name}) classified photo as: '{classified}'")
                return classified
        except Exception as e:
            logger.warning(f"Classification attempt with {model_name} failed: {e}")
            continue

    return candidate_categories[0] if candidate_categories else "Motherboards"


def find_material_rate(materials: list[dict], category: str) -> tuple[float, dict | None]:
    """
    Find indicative rate per kg for a given category from the materials table data.
    Returns (rate_per_kg, matched_material_record).
    """
    cat_clean = category.strip().lower()

    # 1. Exact match (case-insensitive)
    for m in materials:
        name = str(m.get("category") or m.get("name") or m.get("material_name") or "").strip().lower()
        if name == cat_clean:
            rate = (
                m.get("rate_per_kg")
                or m.get("indicative_rate_per_kg")
                or m.get("indicative_rate")
                or m.get("price_per_kg")
                or m.get("rate")
                or m.get("price")
                or 0.0
            )
            return float(rate), m

    # 2. Substring / partial match
    for m in materials:
        name = str(m.get("category") or m.get("name") or m.get("material_name") or "").strip().lower()
        if name and (name in cat_clean or cat_clean in name):
            rate = (
                m.get("rate_per_kg")
                or m.get("indicative_rate_per_kg")
                or m.get("indicative_rate")
                or m.get("price_per_kg")
                or m.get("rate")
                or m.get("price")
                or 0.0
            )
            return float(rate), m

    # Fallback to default rate if unknown
    return 100.0, None


# Web Pages
@app.route("/", methods=["GET"])
def index():
    """GET / -> Render main dashboard index."""
    if "application/json" in request.headers.get("Accept", "") and "text/html" not in request.headers.get("Accept", ""):
        return jsonify({
            "name": "HandOver1 Platform",
            "status": "online",
            "routes": {
                "dashboard": "/",
                "collector": "/collector",
                "recycler": "/recycler",
                "api_health": "/api/health"
            }
        }), 200
    return render_template("index.html")


@app.route("/collector", methods=["GET"])
@app.route("/collector.html", methods=["GET"])
def collector_view():
    """GET /collector -> Render collector portal template."""
    return render_template("collector.html")


@app.route("/recycler", methods=["GET"])
@app.route("/recycler.html", methods=["GET"])
def recycler_view():
    """GET /recycler -> Render recycler portal template."""
    return render_template("recycler.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "supabase_initialized": supabase is not None,
        "gemini_initialized": gemini_client is not None
    }), 200


@app.route("/api/materials", methods=["GET"])
def get_materials():
    """
    Fetch all material categories and their indicative rates per kg
    from the Supabase 'materials' table.
    """
    if not supabase:
        return jsonify(DEFAULT_MATERIALS), 200

    try:
        response = supabase.table("materials").select("*").execute()
        materials = response.data if response.data else DEFAULT_MATERIALS
        return jsonify(materials), 200
    except Exception as e:
        logger.error(f"Error fetching materials: {e}. Returning default baseline materials.")
        return jsonify(DEFAULT_MATERIALS), 200


@app.route("/api/lots/create", methods=["POST"])
def create_lot():
    """
    Handle creation of a new e-waste lot:
    1. Accepts form data (photo upload, category, weight, latitude, longitude).
    2. Uploads photo to Supabase Storage bucket 'ewaste-photos' and gets public URL.
    3. If no category is selected, calls Gemini API to classify the category.
    4. Queries 'materials' table for the selected category's rate per kg.
    5. Calculates indicative_value = weight * rate.
    6. Inserts the new row into the 'lots' table in Supabase.
    7. Returns the inserted lot object with its calculated indicative value as JSON.
    """
    # 1. Validate photo upload
    photo_file = request.files.get("photo") or request.files.get("image") or request.files.get("file")
    if not photo_file or not photo_file.filename:
        return jsonify({
            "error": "Photo upload is required. Please provide a file in the 'photo' form field."
        }), 400

    # 2. Validate weight
    weight_raw = request.form.get("weight")
    if weight_raw is None or str(weight_raw).strip() == "":
        return jsonify({
            "error": "Weight is required. Please provide a numeric value in the 'weight' form field."
        }), 400

    try:
        weight = float(weight_raw)
        if weight <= 0:
            return jsonify({"error": "Weight must be greater than zero."}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid weight format. Must be a valid numeric number (kg)."}), 400

    # Parse optional coordinates (GPS geo-tagging)
    latitude_raw = request.form.get("latitude") or request.form.get("lat")
    longitude_raw = request.form.get("longitude") or request.form.get("long") or request.form.get("lng")
    latitude = None
    longitude = None
    if latitude_raw:
        try:
            latitude = float(latitude_raw)
        except (ValueError, TypeError):
            pass
    if longitude_raw:
        try:
            longitude = float(longitude_raw)
        except (ValueError, TypeError):
            pass

    # Read photo bytes and metadata
    photo_bytes = photo_file.read()
    if not photo_bytes:
        return jsonify({"error": "Uploaded photo file is empty."}), 400

    mime_type = photo_file.mimetype or "image/jpeg"
    orig_filename = secure_filename(photo_file.filename) or "photo.jpg"
    file_ext = os.path.splitext(orig_filename)[1] or ".jpg"
    storage_path = f"lots/{uuid.uuid4().hex}{file_ext}"

    # 3. Upload photo to Supabase Storage bucket 'ewaste-photos'
    photo_url = f"https://nobcpmjtofogwlerrcfm.supabase.co/storage/v1/object/public/ewaste-photos/{storage_path}"
    if supabase:
        try:
            supabase.storage.from_("ewaste-photos").upload(
                path=storage_path,
                file=photo_bytes,
                file_options={"content-type": mime_type, "upsert": "true"}
            )
            logger.info(f"Photo uploaded to 'ewaste-photos' at path '{storage_path}'")
            url_res = supabase.storage.from_("ewaste-photos").get_public_url(storage_path)
            if isinstance(url_res, str):
                photo_url = url_res
            elif isinstance(url_res, dict):
                photo_url = url_res.get("publicUrl") or url_res.get("public_url")
        except Exception as upload_err:
            logger.error(f"Error uploading photo to Supabase storage bucket 'ewaste-photos': {upload_err}")

    # 4. Fetch available materials from Supabase
    materials_list = DEFAULT_MATERIALS
    if supabase:
        try:
            materials_res = supabase.table("materials").select("*").execute()
            if materials_res.data:
                materials_list = materials_res.data
        except Exception as mat_err:
            logger.error(f"Error fetching materials table: {mat_err}")

    available_categories = [
        str(m.get("category") or m.get("name") or m.get("material_name")).strip()
        for m in materials_list
        if (m.get("category") or m.get("name") or m.get("material_name"))
    ]

    # 5. Determine category (use provided or classify with Gemini if empty)
    category = request.form.get("category", "").strip()
    if not category:
        logger.info("No category provided by user; invoking Gemini API to classify photo...")
        category = classify_category_with_gemini(
            photo_bytes=photo_bytes,
            mime_type=mime_type,
            candidate_categories=available_categories
        )

    # 6. Query materials table for selected category's rate per kg and calculate indicative_value
    rate_per_kg, matched_material = find_material_rate(materials_list, category)
    if matched_material and (matched_material.get("category") or matched_material.get("name")):
        category = str(matched_material.get("category") or matched_material.get("name"))

    indicative_value = round(weight * rate_per_kg, 2)
    logger.info(f"Category: '{category}', Rate: {rate_per_kg}/kg, Weight: {weight}kg -> Indicative Value: {indicative_value}")

    # 7. Insert new row into 'lots' table in Supabase
    lot_payload = {
        "category": category,
        "weight": weight,
        "photo_url": photo_url,
        "indicative_value": indicative_value,
    }
    if latitude is not None:
        lot_payload["latitude"] = latitude
    if longitude is not None:
        lot_payload["longitude"] = longitude

    inserted_lot = dict(lot_payload)
    inserted_lot["id"] = uuid.uuid4().hex[:8].upper()

    if supabase:
        try:
            insert_res = supabase.table("lots").insert(lot_payload).execute()
            if insert_res.data and len(insert_res.data) > 0:
                inserted_lot = insert_res.data[0]
        except Exception as insert_err:
            logger.warning(f"Initial insert into Supabase 'lots' failed: {insert_err}. Trying schema fallbacks.")
            # Fallback 1: Remove latitude/longitude if column does not exist
            base_payload = {
                "category": category,
                "weight": weight,
                "photo_url": photo_url,
                "indicative_value": indicative_value
            }
            try:
                insert_res = supabase.table("lots").insert(base_payload).execute()
                if insert_res.data and len(insert_res.data) > 0:
                    inserted_lot = insert_res.data[0]
            except Exception as fb1_err:
                # Fallback 2: Try image_url instead of photo_url
                if "photo_url" in str(fb1_err).lower():
                    try:
                        fb2_payload = dict(base_payload)
                        fb2_payload["image_url"] = fb2_payload.pop("photo_url")
                        insert_res = supabase.table("lots").insert(fb2_payload).execute()
                        if insert_res.data and len(insert_res.data) > 0:
                            inserted_lot = insert_res.data[0]
                    except Exception as fb2_err:
                        logger.error(f"Fallback 2 failed: {fb2_err}")

    # Ensure indicative_value is explicitly attached to the returned object
    if isinstance(inserted_lot, dict):
        if "indicative_value" not in inserted_lot:
            inserted_lot["indicative_value"] = indicative_value
        if latitude is not None and "latitude" not in inserted_lot:
            inserted_lot["latitude"] = latitude
        if longitude is not None and "longitude" not in inserted_lot:
            inserted_lot["longitude"] = longitude

    # 8. Return inserted lot object with its calculated indicative value as JSON
    return jsonify(inserted_lot), 201


@app.route("/api/recyclers/match", methods=["GET"])
def match_recyclers():
    """
    GET /api/recyclers/match?category=PCB
    Query Supabase 'recyclers' table where accepted_materials contains the category
    and return recycler locations with lat/long for Leaflet map display.
    """
    category = request.args.get("category", "").strip()
    all_recyclers = []

    if supabase:
        try:
            response = supabase.table("recyclers").select("*").execute()
            all_recyclers = response.data or []
        except Exception as e:
            logger.error(f"Error querying recyclers table: {e}")

    # If Supabase has no recyclers yet, use default verified facilities
    if not all_recyclers:
        all_recyclers = DEFAULT_RECYCLERS

    cat_lower = category.lower()
    matching_recyclers = []

    for r in all_recyclers:
        accepted = r.get("accepted_materials") or r.get("materials") or []
        matches = False

        if not category:
            matches = True
        elif isinstance(accepted, list):
            matches = any(
                cat_lower in str(item).lower() or str(item).lower() in cat_lower
                for item in accepted
            )
        elif isinstance(accepted, str):
            matches = cat_lower in accepted.lower()

        if matches:
            lat_val = r.get("lat") if r.get("lat") is not None else r.get("latitude")
            lng_val = (
                r.get("long")
                if r.get("long") is not None
                else (r.get("lng") if r.get("lng") is not None else r.get("longitude"))
            )

            rec_entry = dict(r)
            try:
                rec_entry["lat"] = float(lat_val) if lat_val is not None else None
            except (ValueError, TypeError):
                rec_entry["lat"] = None

            try:
                rec_entry["long"] = float(lng_val) if lng_val is not None else None
            except (ValueError, TypeError):
                rec_entry["long"] = None

            rec_entry["lng"] = rec_entry["long"]
            matching_recyclers.append(rec_entry)

    return jsonify(matching_recyclers), 200


@app.route("/api/transactions/complete", methods=["POST"])
def complete_transaction():
    """
    POST /api/transactions/complete
    Record final weight, final price, payment status, and update the lot status to 'COMPLETED'.
    """
    data = request.get_json(silent=True) or request.form.to_dict() or {}

    lot_id = data.get("lot_id") or data.get("lotId") or data.get("id")
    transaction_id = data.get("transaction_id") or data.get("transactionId")
    payment_status = data.get("payment_status") or data.get("paymentStatus") or "COMPLETED"

    final_weight_raw = data.get("final_weight") or data.get("finalWeight") or data.get("weight")
    final_price_raw = data.get("final_price") or data.get("finalPrice") or data.get("price") or data.get("amount")

    final_weight = None
    if final_weight_raw is not None and str(final_weight_raw).strip() != "":
        try:
            final_weight = float(final_weight_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid format for final_weight"}), 400

    final_price = None
    if final_price_raw is not None and str(final_price_raw).strip() != "":
        try:
            final_price = float(final_price_raw)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid format for final_price"}), 400

    # 1. Update the lot status to 'COMPLETED' in Supabase 'lots' table
    if supabase and lot_id:
        lot_update_payload = {"status": "COMPLETED"}
        if final_weight is not None:
            lot_update_payload["final_weight"] = final_weight
        if final_price is not None:
            lot_update_payload["final_price"] = final_price

        try:
            supabase.table("lots").update(lot_update_payload).eq("id", lot_id).execute()
            logger.info(f"Lot {lot_id} status updated to COMPLETED.")
        except Exception as lot_err:
            logger.warning(f"Error updating lot: {lot_err}")
            try:
                supabase.table("lots").update({"status": "COMPLETED"}).eq("id", lot_id).execute()
            except Exception as e2:
                logger.error(f"Failed to update lot status: {e2}")

    # 2. Record transaction in Supabase 'transactions' table
    tx_payload = {
        "payment_status": payment_status,
        "status": "COMPLETED"
    }
    if lot_id is not None:
        tx_payload["lot_id"] = lot_id
    if final_weight is not None:
        tx_payload["final_weight"] = final_weight
    if final_price is not None:
        tx_payload["final_price"] = final_price
    if data.get("recycler_id"):
        tx_payload["recycler_id"] = data.get("recycler_id")

    transaction_record = dict(tx_payload)
    transaction_record["id"] = transaction_id or uuid.uuid4().hex[:8].upper()

    if supabase:
        try:
            if transaction_id:
                tx_res = supabase.table("transactions").update(tx_payload).eq("id", transaction_id).execute()
                if tx_res.data and len(tx_res.data) > 0:
                    transaction_record = tx_res.data[0]
            else:
                tx_res = supabase.table("transactions").insert(tx_payload).execute()
                if tx_res.data and len(tx_res.data) > 0:
                    transaction_record = tx_res.data[0]
        except Exception as tx_err:
            logger.error(f"Error saving to transactions table: {tx_err}")

    response_payload = {
        "success": True,
        "message": "Transaction recorded and lot status set to COMPLETED",
        "lot_id": lot_id,
        "lot_status": "COMPLETED",
        "final_weight": final_weight,
        "final_price": final_price,
        "payment_status": payment_status,
        "transaction": transaction_record
    }

    return jsonify(response_payload), 200


@app.route("/api/transactions/<id>/qr", methods=["GET"])
def get_transaction_qr(id):
    """
    GET /api/transactions/<id>/qr
    Use the Python qrcode library to generate an in-memory PNG QR code image
    containing the transaction reference text and return it as a raw image response.
    """
    ref_param = request.args.get("ref") or request.args.get("text")
    if ref_param:
        ref_text = ref_param
    else:
        ref_text = f"HANDOVER1-TX-{id}"
        if supabase:
            try:
                tx_res = supabase.table("transactions").select("*").eq("id", id).execute()
                if tx_res.data and len(tx_res.data) > 0:
                    tx = tx_res.data[0]
                    price = tx.get("final_price") or tx.get("amount") or "N/A"
                    weight = tx.get("final_weight") or tx.get("weight") or "N/A"
                    status = tx.get("payment_status") or tx.get("status") or "COMPLETED"
                    ref_text = f"TX:{id}|WEIGHT:{weight}kg|PRICE:INR_{price}|STATUS:{status}"
            except Exception as e:
                logger.debug(f"Could not fetch full tx details for QR: {e}")

    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(ref_text)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return send_file(
            img_buffer,
            mimetype="image/png",
            as_attachment=False,
            download_name=f"transaction_{id}_qr.png"
        )
    except Exception as qr_err:
        logger.error(f"Error generating QR code for transaction {id}: {qr_err}")
        return jsonify({
            "error": "Failed to generate QR code",
            "details": str(qr_err)
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    logger.info(f"Starting Flask server on port {port} (debug={debug_mode})...")
    app.run(host="0.0.0.0", port=port, debug=debug_mode)