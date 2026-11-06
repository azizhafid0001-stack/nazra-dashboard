import time
import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

st.set_page_config(
    page_title="Nazra Live Dashboard", page_icon="☕", layout="wide"
)

st.title("☕ Nazra Live Dashboard - نظام مراقبة الطاولات الذكي")
st.markdown(
    "إعداد: **عزيز حفيظ** | **المعهد المتخصص في مهن الطاقة المتجددة بطرفاية**"
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

if "tables_db" not in st.session_state:
    st.session_state.tables_db = {
        "طاولة المجسم": {
            "status": "🟢 فارغة (نقية)",
            "orders": [],
            "start_time": None,
            "last_seen_person": None,
            "notification_sent": False,
        }
    }

st.markdown("---")
with st.form("menu_form"):
    st.subheader("🛒 قائمة الطعام (Menu) وإشعارات النادل")
    menu_items = [
        "☕ قهوة إبريسو (Espresso)",
        "🥛 كابتشينو (Cappuccino)",
        "🫖 شاي مغربي (Moroccan Tea)",
        "🧃 عصير برتقال (Orange Juice)",
        "🥐 كرواسون (Croissant)",
    ]
    selected_items = st.multiselect("اختر الطلبات من القائمة:", menu_items)
    submit_order = st.form_submit_button("إرسال الطلبات المحددة وإشعار النادل")

    if submit_order:
        if selected_items:
            orders_str = ", ".join(selected_items)
            st.session_state.tables_db["طاولة المجسم"]["orders"].extend(
                selected_items
            )
            st.success(
                f"✅ تم إرسال الطلبات بنجاح لطاولة المجسم: **{orders_str}**"
            )
            st.info(
                "🚨 [تنبيه مرسل للسرباي]: يرجى التوجه فوراً إلى طاولة المجسم لتقديم الخدمة."
            )
        else:
            st.warning("المرجو اختيار طلب واحد على الأقل من القائمة.")

st.markdown("---")

mode = st.radio(
    "🌐 اختر مصدر البث المباشر:",
    [
        "💻 كاميرا الحاسوب المباشرة (Local)",
        "📱 كاميرا الهاتف / المتصفح عبر الإنترنت (Cloud Live)",
    ],
    horizontal=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 البث المباشر ومراقبة الطاولة (ROI)")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 حالة الطاولة والعداد الزمني")
    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    notification_placeholder = st.empty()

polygon = np.array([[60, 50], [580, 50], [580, 450], [60, 450]], np.int32)
zone = sv.PolygonZone(
    polygon=polygon, triggering_anchors=[sv.Position.CENTER]
)


def process_frame(frame, table_data):
    frame = cv2.resize(frame, (640, 480))
    results = model(
        frame,
        classes=[0, 39, 41, 42, 43, 44, 45],
        conf=0.20,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_frame = results.plot()

    in_zone = zone.trigger(detections)
    classes_in_zone = detections.class_id[in_zone]

    raw_has_person = 0 in classes_in_zone
    has_dishes = any(c in [39, 41, 42, 43, 44, 45] for c in classes_in_zone)
    current_time = time.time()

    if raw_has_person:
        table_data["last_seen_person"] = current_time
        has_person = True
    elif (
        table_data["last_seen_person"] is not None
        and (current_time - table_data["last_seen_person"]) <= 3.0
    ):
        has_person = True
    else:
        has_person = False

    alert_type = None
    alert_msg = ""

    if has_person:
        if table_data["start_time"] is None:
            table_data["start_time"] = current_time

        elapsed = int(current_time - table_data["start_time"])
        occ_val, free_val = 1, 0

        if elapsed <= 5:
            box_color = (0, 255, 255)
            label_text = f"Seating... ({elapsed}s / 5s)"
            table_data["status"] = (
                f"🟡 جالس فبلاصتو ({elapsed} ثواني) - طور التأكيد"
            )
            status_type = "warning"
        elif elapsed <= 30:
            box_color = (0, 0, 255)
            label_text = f"Occupied ({elapsed}s / 30s)"
            table_data["status"] = f"🔴 الطاولة مشغولة ({elapsed} / 30 ثانية)"
            status_type = "error"
        else:
            box_color = (0, 0, 255)
            label_text = f"Waiter Needed! ({elapsed}s)"
            table_data["status"] = (
                f"🔴 مشغولة ({elapsed} ثانية) - خاص السرباي يجي!"
            )
            status_type = "error"
            alert_type = "error"
            alert_msg = "🚨 **ميساج عاجل للسرباي:** الزبون كمل 30 ثانية فـ الطابلة، خاصك تمشي عندهم دابا تسربيهم وتشد الطلب!"

    elif has_dishes:
        table_data["start_time"] = None
        table_data["last_seen_person"] = None
        occ_val, free_val = 1, 0
        box_color = (0, 165, 255)
        label_text = "Needs Cleaning! (Dishes)"
        table_data["status"] = "🟠 غير نظيفة (بقاو المواعن فوق الطابلة)"
        status_type = "warning"
        alert_type = "warning"
        alert_msg = "🧹 **ميساج للسرباي (تنظيف):** الزبون ناض وبقاو المواعن فوق الطابلة! عافاك سير جمعهم ونقي الطابلة باش تولي واجدة."

    else:
        table_data["start_time"] = None
        table_data["last_seen_person"] = None
        box_color = (0, 255, 0)
        label_text = "Free & Clean"
        table_data["status"] = "🟢 فارغة (نقية)"
        occ_val, free_val = 0, 1
        status_type = "success"

    cv2.polylines(
        annotated_frame, [polygon], isClosed=True, color=box_color, thickness=4
    )
    cv2.putText(
        annotated_frame,
        label_text,
        (polygon[0][0], polygon[0][1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        box_color,
        2,
    )

    return annotated_frame, occ_val, free_val, status_type, alert_type, alert_msg


def update_ui(table_data, occ_val, free_val, status_type, alert_type, alert_msg):
    if status_type == "warning":
        status_placeholder.warning(table_data["status"])
    elif status_type == "error":
        status_placeholder.error(table_data["status"])
    else:
        status_placeholder.success(table_data["status"])

    if alert_type == "error":
        notification_placeholder.error(alert_msg)
    elif alert_type == "warning":
        notification_placeholder.warning(alert_msg)
    else:
        notification_placeholder.empty()

    with metric_placeholder.container():
        c1, c2, c3 = st.columns(3)
        c1.metric(label="مجموع الطاولات", value="1")
        c2.metric(label="فارغة (نقية)", value=str(free_val))
        c3.metric(label="مشغولة / تنظيف", value=str(occ_val))


if mode == "💻 كاميرا الحاسوب المباشرة (Local)":
    run_video = st.checkbox("تشغيل البث المباشر (فيديو)", value=True)

    if run_video:
        if "cap" not in st.session_state or not st.session_state.cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            st.session_state.cap = cap
        cap = st.session_state.cap

        if not cap.isOpened():
            st.error(
                "⚠️ الكاميرا غير متاحة محلياً! إذا كنت تتصفح الموقع من الإنترنت، اختر الوضع الثاني (كاميرا الهاتف / المتصفح)."
            )
        else:
            while run_video:
                ret, frame = cap.read()
                if not ret:
                    break

                table_data = st.session_state.tables_db["طاولة المجسم"]
                (
                    annotated_frame,
                    occ_val,
                    free_val,
                    status_type,
                    alert_type,
                    alert_msg,
                ) = process_frame(frame, table_data)
                update_ui(
                    table_data,
                    occ_val,
                    free_val,
                    status_type,
                    alert_type,
                    alert_msg,
                )

                annotated_frame = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )
                video_placeholder.image(
                    annotated_frame, channels="RGB", use_container_width=True
                )
                time.sleep(0.01)
    else:
        if "cap" in st.session_state and st.session_state.cap.isOpened():
            st.session_state.cap.release()

else:
    try:
        import av
        from streamlit_webrtc import webrtc_streamer

        table_data = st.session_state.tables_db["طاولة المجسم"]

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            annotated_frame, _, _, _, _, _ = process_frame(img, table_data)
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        with col1:
            webrtc_streamer(
                key="nazra-cloud",
                video_frame_callback=video_frame_callback,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"video": True, "audio": False},
            )
    except ImportError:
        st.warning(
            "لتشغيل بث المتصفح السحابي، قم بتثبيت المكتبة: pip install streamlit-webrtc av"
        )