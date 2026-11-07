import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
from PIL import Image

# إعداد الصفحة
st.set_page_config(
    page_title="Nazra Live Dashboard",
    page_icon="☕",
    layout="wide"
)

# ترويسة الموقع
st.title("☕ Nazra Live Dashboard - نظام المراقبة الذكي للطاولات")
st.markdown("إعداد: **عزيز حفيظ** | **المعهد المتخصص في مهن الطاقة المتجددة بطرفاية**")

# تحميل نموذج YOLOv8
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# تهيئة قاعدة البيانات المحلية للجلسة
if "table_data" not in st.session_state:
    st.session_state.table_data = {
        "status": "🟢 فارغة ونقية",
        "start_time": None,
        "last_seen_person": None,
        "has_dishes": False,
        "is_occupied": False,
        "needs_cleaning": False,
        "elapsed_seconds": 0,
        "orders": []
    }

# قسم قائمة الطعام والطلبات
with st.expander("🛒 قائمة الطلبات (Menu) - إرسال مباشر للنادل", expanded=False):
    menu_items = [
        "☕ قهوة إبريسو (Espresso)",
        "🥛 كابتشينو (Cappuccino)",
        "🫖 شاي مغربي (Moroccan Tea)",
        "🧃 عصير برتقال (Orange Juice)",
        "🥐 كرواسون (Croissant)"
    ]
    selected_items = st.multiselect("اختر طلبات الطاولة:", menu_items)
    if st.button("إرسال الطلبات إلى السرباي"):
        if selected_items:
            st.session_state.table_data["orders"].extend(selected_items)
            st.success(f"✅ تم إرسال الطلبات بنجاح: {', '.join(selected_items)}")
        else:
            st.warning("يرجى اختيار عنصر واحد على الأقل.")

st.markdown("---")

# إحصائيات علوية منفصلة (مع خانة مستقلة للطاولات المتسخة)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

total_tables = 1
free_tables = 1 if (not st.session_state.table_data["is_occupied"] and not st.session_state.table_data["needs_cleaning"]) else 0
occupied_tables = 1 if st.session_state.table_data["is_occupied"] else 0
dirty_tables = 1 if st.session_state.table_data["needs_cleaning"] else 0

col_m1.metric("إجمالي الطاولات", f"{total_tables}")
col_m2.metric("🟢 فارغة ونقية", f"{free_tables}")
col_m3.metric("🔴 مشغولة بالزبائن", f"{occupied_tables}")
col_m4.metric("🟠 تحتاج تنظيف (موسخة)", f"{dirty_tables}")

st.markdown("---")

# تقسيم الصفحة: الكاميرا والمراقبة يساراً، والتحكم والتنبيهات يميناً
col_cam, col_info = st.columns([1.8, 1.2])

# تحديد منطقة الطاولة (Polygon ROI)
polygon = np.array([
    [30, 30],
    [610, 30],
    [610, 450],
    [30, 450]
], np.int32)
zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER, sv.Position.BOTTOM_CENTER])
box_annotator = sv.BoxAnnotator(thickness=2)

with col_cam:
    st.subheader("📷 التقاط وفحص الطاولة مباشرة")
    img_file_buffer = st.camera_input("وجّه الكاميرا نحو الطاولة واضغط لالتقاط الحالة")

    if img_file_buffer is not None:
        # قراءة الصورة
        bytes_data = img_file_buffer.getvalue()
        image_np = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (640, 480))

        # كشف الأشخاص (0) والأواني/الكؤوس/الصحون (39, 41, 42, 43, 44, 45)
        results = model(frame, imgsz=320, classes=[0, 39, 41, 42, 43, 44, 45], conf=0.20, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        in_zone = zone.trigger(detections)
        classes_in_zone = detections.class_id[in_zone]

        raw_has_person = 0 in classes_in_zone
        has_dishes = any(c in [39, 41, 42, 43, 44, 45] for c in classes_in_zone)
        current_time = time.time()

        # معالجة منطق الشخص والمؤقت
        if raw_has_person:
            st.session_state.table_data["last_seen_person"] = current_time
            has_person = True
        elif st.session_state.table_data["last_seen_person"] is not None and (current_time - st.session_state.table_data["last_seen_person"]) <= 4.0:
            has_person = True
        else:
            has_person = False

        if has_person:
            if st.session_state.table_data["start_time"] is None:
                st.session_state.table_data["start_time"] = current_time

            elapsed = int(current_time - st.session_state.table_data["start_time"])
            st.session_state.table_data["elapsed_seconds"] = elapsed
            st.session_state.table_data["is_occupied"] = True
            st.session_state.table_data["needs_cleaning"] = False

            if elapsed <= 5:
                box_color = (0, 255, 255) # أصفر
                st.session_state.table_data["status"] = f"🟡 شخص جالس ({elapsed} ثوانٍ) - طور التحقق والتأكيد"
                label_text = f"Confirming... ({elapsed}s/5s)"
            elif elapsed <= 30:
                box_color = (0, 0, 255) # أحمر
                st.session_state.table_data["status"] = f"🔴 الطاولة مشغولة ({elapsed} ثانية)"
                label_text = f"Occupied ({elapsed}s/30s)"
            else:
                box_color = (0, 0, 255) # أحمر
                st.session_state.table_data["status"] = f"🚨 تنبيه نادل! الزبون تجاوز ({elapsed} ثانية)"
                label_text = f"WAITER NEEDED! ({elapsed}s)"

        elif has_dishes:
            st.session_state.table_data["start_time"] = None
            st.session_state.table_data["last_seen_person"] = None
            st.session_state.table_data["is_occupied"] = False
            st.session_state.table_data["needs_cleaning"] = True
            st.session_state.table_data["elapsed_seconds"] = 0
            box_color = (0, 165, 255) # برتقالي
            st.session_state.table_data["status"] = "🟠 غير نظيفة (توجد كؤوس أو أوانٍ متروكة)"
            label_text = "DIRTY - Needs Cleaning"

        else:
            st.session_state.table_data["start_time"] = None
            st.session_state.table_data["last_seen_person"] = None
            st.session_state.table_data["is_occupied"] = False
            st.session_state.table_data["needs_cleaning"] = False
            st.session_state.table_data["elapsed_seconds"] = 0
            box_color = (0, 255, 0) # أخضر
            st.session_state.table_data["status"] = "🟢 فارغة ونقية وجاهزة"
            label_text = "FREE & CLEAN"

        # رسم المربعات والبيانات التوضيحية
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        cv2.polylines(annotated_frame, [polygon], isClosed=True, color=box_color, thickness=4)
        cv2.rectangle(annotated_frame, (30, 0), (610, 40), (0, 0, 0), -1)
        cv2.putText(annotated_frame, label_text, (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        # عرض النتيجة
        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="التحليل الذكي للطاولة", use_container_width=True)

with col_info:
    st.subheader("🔔 حالة الطاولة والتنبيهات المباشرة")
    
    t_status = st.session_state.table_data["status"]
    elapsed = st.session_state.table_data["elapsed_seconds"]

    if st.session_state.table_data["is_occupied"]:
        if elapsed <= 5:
            st.warning(f"⏳ **الحالة:** {t_status}")
            st.info("الزبون جلس للتو، النظام بانتظار استقراره وتأكيد الجلوس.")
        elif elapsed <= 30:
            st.error(f"👤 **الحالة:** {t_status}")
            st.write(f"⏱️ العداد الزمني للجلوس: **{elapsed} ثانية**")
        else:
            st.error(f"🚨 **تنبيه عاجل للنادل (سرباي):**\n\nمرت أكثر من 30 ثانية على جلوس الزبون دون تقديم الخدمة. يرجى التوجه فوراً للطاولة!")
            st.write(f"⏱️ العداد الزمني: **{elapsed} ثانية**")
    
    elif st.session_state.table_data["needs_cleaning"]:
        st.warning(f"🧹 **تنبيه تنظيف عاجل:**\n\n{t_status}\n\nغادر الزبون الطاولة وبقيت الأواني/الكؤوس. يرجى من عامل النظافة أو السرباي إفراغ الطاولة وتجهيزها للزبون القادم.")
    
    else:
        st.success(f"✨ **الحالة:** {t_status}")
        st.write("الطاولة نقية تماماً وبانتظار زبائن جدد.")

    # زر لإعادة ضبط المؤقت يدوياً إذا دعت الحاجة
    if st.button("🔄 إعادة ضبط حالة الطاولة"):
        st.session_state.table_data = {
            "status": "🟢 فارغة ونقية",
            "start_time": None,
            "last_seen_person": None,
            "has_dishes": False,
            "is_occupied": False,
            "needs_cleaning": False,
            "elapsed_seconds": 0,
            "orders": []
        }
        st.rerun()
