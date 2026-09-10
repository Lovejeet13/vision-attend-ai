import os
import cv2
import numpy as np
import pandas as pd
import base64
import zipfile

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify
)

from werkzeug.utils import secure_filename
from deepface import DeepFace
from scipy.spatial.distance import cosine


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__)


# ============================================================
# Constants
# ============================================================

UPLOAD_FOLDER = "uploads"
KNOWN_FACES_FOLDER = "known_faces"
ATTENDANCE_CSV = "attendance.csv"

RECOGNITION_MODEL = "Facenet"

# Lower = stricter matching
THRESHOLD = 0.6


# ============================================================
# Global Variables
# ============================================================

# Structure:
#
# {
#     "101": {
#         "embedding": numpy_array,
#         "name": "Lovejeet Patel"
#     }
# }

known_faces = {}

# Structure:
#
# {
#     "101": "Present"
# }

attendance = {}

recognized_person = "Waiting..."

attendance_started = False


# ============================================================
# Create Required Folders
# ============================================================

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWN_FACES_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)


# ============================================================
# OpenCV Face Detector
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# 1. HOME / UPLOAD PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("upload.html")


# ============================================================
# 2. UPLOAD FILE
# ============================================================

@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    try:

        filename = secure_filename(file.filename)

        if not filename:
            return redirect(request.url)

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Save uploaded file
        file.save(file_path)

        # ====================================================
        # ZIP FILE
        # ====================================================

        if zipfile.is_zipfile(file_path):

            with zipfile.ZipFile(
                file_path,
                "r"
            ) as zip_ref:

                zip_ref.extractall(
                    KNOWN_FACES_FOLDER
                )

            os.remove(file_path)

            print(
                f"ZIP extracted successfully: {filename}"
            )

        # ====================================================
        # NORMAL IMAGE FILE
        # ====================================================

        else:

            destination = os.path.join(
                KNOWN_FACES_FOLDER,
                filename
            )

            os.replace(
                file_path,
                destination
            )

            print(
                f"File uploaded: {filename}"
            )

        # Reload known faces
        load_known_faces()

        print(
            f"Total known faces: {len(known_faces)}"
        )

        return redirect(
            url_for("attendance_page")
        )

    except Exception as e:

        print(
            "Upload error:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 3. ATTENDANCE PAGE
# ============================================================

@app.route("/attendance")
def attendance_page():

    return render_template(
        "attendance.html"
    )


# ============================================================
# 4. START ATTENDANCE
# ============================================================

@app.route("/start_attendance")
def start_attendance():

    global attendance_started
    global attendance
    global recognized_person

    attendance_started = True

    # Reset attendance
    attendance.clear()

    recognized_person = "Waiting..."

    print(
        "Attendance started"
    )

    return jsonify({
        "message": "Attendance Started!"
    })


# ============================================================
# 5. STOP ATTENDANCE
# ============================================================

@app.route("/stop_attendance")
def stop_attendance():

    global attendance_started

    attendance_started = False

    print(
        "Attendance stopped"
    )

    return jsonify({
        "message": "Attendance Stopped!"
    })


# ============================================================
# 6. PROCESS CAMERA FRAME
# ============================================================

@app.route(
    "/process_frame",
    methods=["POST"]
)
def process_frame():

    global recognized_person
    global attendance

    try:

        # ====================================================
        # Check JSON
        # ====================================================

        if not request.is_json:

            return jsonify({
                "error":
                    "Request must contain JSON data",
                "name":
                    "Error",
                "status":
                    "error"
            }), 400


        data = request.json


        if not data or "image" not in data:

            return jsonify({
                "error":
                    "No image received",
                "name":
                    "Waiting...",
                "status":
                    "waiting"
            }), 400


        image_data = data["image"]


        # ====================================================
        # Remove Base64 metadata
        # ====================================================

        if "," not in image_data:

            return jsonify({
                "error":
                    "Invalid image data",
                "name":
                    "Error",
                "status":
                    "error"
            }), 400


        encoded_data = image_data.split(
            ",",
            1
        )[1]


        # ====================================================
        # Decode image
        # ====================================================

        image_bytes = base64.b64decode(
            encoded_data
        )


        np_arr = np.frombuffer(
            image_bytes,
            np.uint8
        )


        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({
                "error":
                    "Could not decode image",
                "name":
                    "Error",
                "status":
                    "error"
            }), 400


        # ====================================================
        # Resize image
        # ====================================================

        frame = cv2.resize(
            frame,
            (640, 480)
        )


        # ====================================================
        # Detect faces using OpenCV
        # ====================================================

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )


        # ====================================================
        # No face detected
        # ====================================================

        if len(faces) == 0:

            recognized_person = "No Face Detected"

            return jsonify({

                "message":
                    "No face detected",

                "name":
                    "No Face Detected",

                "status":
                    "waiting"
            })


        # ====================================================
        # Check known faces
        # ====================================================

        if not known_faces:

            recognized_person = "No Known Faces"

            print(
                "No known faces loaded!"
            )

            return jsonify({

                "message":
                    "No known faces",

                "name":
                    "No Known Faces",

                "status":
                    "error"
            })


        # ====================================================
        # Best match variables
        # ====================================================

        overall_best_match = None

        overall_best_name = None

        overall_best_score = float("inf")


        # ====================================================
        # Process each detected face
        # ====================================================

        for (x, y, w, h) in faces:

            face_roi = frame[
                y:y+h,
                x:x+w
            ]


            # Ignore tiny faces
            if (
                face_roi.shape[0] < 60
                or face_roi.shape[1] < 60
            ):
                continue


            try:

                # =================================================
                # IMPORTANT:
                #
                # OpenCV already detected the face.
                #
                # detector_backend="skip"
                # prevents DeepFace from detecting again.
                #
                # This makes processing faster.
                # =================================================

                embeddings = DeepFace.represent(

                    img_path=face_roi,

                    model_name=RECOGNITION_MODEL,

                    detector_backend="skip",

                    enforce_detection=False,

                    align=False
                )


            except Exception as e:

                print(
                    "DeepFace error:",
                    str(e)
                )

                continue


            if not embeddings:
                continue


            # ====================================================
            # Get embedding
            # ====================================================

            face_embedding = np.array(
                embeddings[0]["embedding"]
            )


            # ====================================================
            # Compare with known students
            # ====================================================

            for student_id, student_data in known_faces.items():

                stored_embedding = student_data[
                    "embedding"
                ]

                student_name = student_data[
                    "name"
                ]


                # Calculate cosine distance
                distance = cosine(
                    face_embedding,
                    stored_embedding
                )


                print(
                    f"Comparing with "
                    f"{student_id}: "
                    f"{distance:.4f}"
                )


                # Keep closest match
                if distance < overall_best_score:

                    overall_best_score = distance

                    overall_best_match = student_id

                    overall_best_name = student_name


        # ====================================================
        # FACE RECOGNIZED
        # ====================================================

        if (
            overall_best_match is not None
            and overall_best_score < THRESHOLD
        ):

            student_id = overall_best_match

            student_name = overall_best_name


            recognized_person = student_name


            # =================================================
            # Mark attendance
            # =================================================

            attendance[student_id] = "Present"


            print(
                f"Recognized: "
                f"{student_name} "
                f"(ID: {student_id}, "
                f"Distance: "
                f"{overall_best_score:.4f})"
            )


            # =================================================
            # Send success response to browser
            # =================================================

            return jsonify({

                "message":
                    "Attendance marked",

                "name":
                    student_name,

                "id":
                    student_id,

                "status":
                    "Present",

                "distance":
                    round(
                        float(overall_best_score),
                        4
                    )
            })


        # ====================================================
        # UNKNOWN FACE
        # ====================================================

        recognized_person = "Unknown"


        if overall_best_match is not None:

            print(
                f"Unknown face. "
                f"Best match: "
                f"{overall_best_match}, "
                f"Distance: "
                f"{overall_best_score:.4f}, "
                f"Threshold: "
                f"{THRESHOLD}"
            )


        return jsonify({

            "message":
                "Face not recognized",

            "name":
                "Unknown",

            "status":
                "unknown",

            "distance":
                round(
                    float(overall_best_score),
                    4
                )
                if overall_best_match is not None
                else None
        })


    except Exception as e:

        print(
            "Error processing frame:",
            str(e)
        )


        return jsonify({

            "error":
                str(e),

            "name":
                "Error",

            "status":
                "error"

        }), 500


# ============================================================
# 7. GET DETECTED NAME
# ============================================================

@app.route("/get_detected_name")
def get_detected_name():

    return jsonify({

        "name":
            recognized_person

    })


# ============================================================
# 8. FINAL PAGE
# ============================================================

@app.route("/final")
def final_page():

    return render_template(
        "final.html"
    )


# ============================================================
# 9. VIEW ATTENDANCE
# ============================================================

@app.route("/view_attendance")
def view_attendance():

    all_students = {}


    # ========================================================
    # Add all known students as Absent
    # ========================================================

    for student_id, student_data in known_faces.items():

        student_name = student_data[
            "name"
        ]


        all_students[student_id] = {

            "ID":
                student_id,

            "Name":
                student_name,

            "Attendance":
                "Absent"
        }


    # ========================================================
    # Mark detected students as Present
    # ========================================================

    for student_id in attendance.keys():

        if student_id in all_students:

            all_students[
                student_id
            ]["Attendance"] = "Present"


    # ========================================================
    # Convert to DataFrame
    # ========================================================

    df = pd.DataFrame(
        list(all_students.values())
    )


    # ========================================================
    # Add Serial Number
    # ========================================================

    if not df.empty:

        df.insert(
            0,
            "Sr. No.",
            range(
                1,
                len(df) + 1
            )
        )


    # ========================================================
    # Convert to dictionary
    # ========================================================

    data = df.to_dict(
        orient="records"
    )


    return render_template(
        "view_attendance.html",
        data=data
    )


# ============================================================
# 10. DOWNLOAD ATTENDANCE
# ============================================================

@app.route("/download_attendance")
def download_attendance():

    all_students = {}


    # ========================================================
    # Add all students as Absent
    # ========================================================

    for student_id, student_data in known_faces.items():

        student_name = student_data[
            "name"
        ]


        all_students[student_id] = {

            "ID":
                student_id,

            "Name":
                student_name,

            "Attendance":
                "Absent"
        }


    # ========================================================
    # Mark Present Students
    # ========================================================

    for student_id in attendance.keys():

        if student_id in all_students:

            all_students[
                student_id
            ]["Attendance"] = "Present"


    # ========================================================
    # Create DataFrame
    # ========================================================

    df = pd.DataFrame(
        list(all_students.values())
    )


    # ========================================================
    # Add Serial Number
    # ========================================================

    if not df.empty:

        df.insert(
            0,
            "Sr. No.",
            range(
                1,
                len(df) + 1
            )
        )


    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = os.path.join(
        "static",
        ATTENDANCE_CSV
    )


    df.to_csv(
        csv_path,
        index=False
    )


    return redirect(
        url_for(
            "static",
            filename=ATTENDANCE_CSV
        )
    )


# ============================================================
# 11. LOAD KNOWN FACES
# ============================================================

def load_known_faces():

    global known_faces

    known_faces.clear()


    print(
        "\nLoading known faces..."
    )


    # ========================================================
    # Check folder
    # ========================================================

    if not os.path.exists(
        KNOWN_FACES_FOLDER
    ):

        print(
            "known_faces folder does not exist."
        )

        return


    # ========================================================
    # Read every file
    # ========================================================

    for filename in os.listdir(
        KNOWN_FACES_FOLDER
    ):


        # ====================================================
        # Only process images
        # ====================================================

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            continue


        img_path = os.path.join(
            KNOWN_FACES_FOLDER,
            filename
        )


        try:

            print(
                f"Processing: {filename}"
            )


            # =================================================
            # Generate embedding
            # =================================================

            embedding = DeepFace.represent(

                img_path=img_path,

                model_name=RECOGNITION_MODEL,

                detector_backend="opencv",

                enforce_detection=True,

                align=True
            )


            if not embedding:

                print(
                    f"No face found in {filename}"
                )

                continue


            # =================================================
            # Get embedding
            # =================================================

            face_embedding = np.array(
                embedding[0]["embedding"]
            )


            # =================================================
            # Extract ID and name
            #
            # Example:
            #
            # 101_Lovejeet_Patel.jpg
            #
            # ID   = 101
            # Name = Lovejeet Patel
            # =================================================

            name_without_extension, _ = os.path.splitext(
                filename
            )


            parts = name_without_extension.split(
                "_",
                1
            )


            student_id = parts[0]


            if len(parts) > 1:

                student_name = parts[
                    1
                ].replace(
                    "_",
                    " "
                )

            else:

                student_name = "Unknown"


            # =================================================
            # Store student
            # =================================================

            known_faces[student_id] = {

                "embedding":
                    face_embedding,

                "name":
                    student_name
            }


            print(
                f"Loaded successfully: "
                f"ID={student_id}, "
                f"Name={student_name}"
            )


        except Exception as e:

            print(
                f"Error processing "
                f"{filename}: "
                f"{str(e)}"
            )


    # ========================================================
    # Final result
    # ========================================================

    print(
        f"\nTotal known faces loaded: "
        f"{len(known_faces)}"
    )

    print(
        "----------------------------------------\n"
    )


# ============================================================
# LOAD KNOWN FACES WHEN FLASK STARTS
# ============================================================

load_known_faces()


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=10000,

        debug=False
    )