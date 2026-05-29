import streamlit as st
import pandas as pd
import math
from datetime import datetime, time
import database as db

# Page configuration
st.set_page_config(
    page_title="Teacher Attendance System",
    page_icon="📍",
    layout="wide"
)

# Initialize database
db.init_database()

# ===========================
# HELPER FUNCTIONS
# ===========================

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def get_status_color(status):
    """Return color for status badge"""
    if status == "Present":
        return "🟢"
    else:
        return "🔴"

# ===========================
# CAMPUS LOCATION (Hyderabad)
# ===========================
CAMPUS_LAT = 17.3850
CAMPUS_LON = 78.4867
CAMPUS_RADIUS = 0.1  # 100 meters

# ===========================
# ATTENDANCE WINDOW TIMINGS
# ===========================
START_TIME = time(8, 45)
END_TIME = time(9, 15)

# ===========================
# MAIN APP
# ===========================

st.title("📍 Teacher Attendance System")
st.caption("GPS-based Attendance Tracking System")

# ===========================
# SIDEBAR - LOGOUT
# ===========================
if "logged_in" in st.session_state and st.session_state.logged_in:
    with st.sidebar:
        st.write("### User Info")
        teacher_info = db.get_teacher_by_id(st.session_state.teacher_id)
        if teacher_info is not None:
            st.info(f"**Name:** {teacher_info['name']}")
            st.info(f"**ID:** {teacher_info['teacher_id']}")
            st.info(f"**Department:** {teacher_info['department']}")
            st.info(f"**Subject:** {teacher_info['subject']}")
        
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.teacher_id = None
            st.rerun()

# ===========================
# LOGIN SECTION
# ===========================
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        teacher_id = st.text_input(
            "Enter Teacher ID",
            placeholder="e.g., 1, 2, 3..."
        )
        
        if st.button("Login", type="primary", use_container_width=True):
            if teacher_id.strip() == "":
                st.error("❌ Please enter your Teacher ID")
            elif db.teacher_exists(teacher_id):
                st.session_state.logged_in = True
                st.session_state.teacher_id = teacher_id
                st.success("✅ Login Successful!")
                st.rerun()
            else:
                st.error("❌ Invalid Teacher ID. Please check and try again.")
        
        st.divider()
        st.caption("💡 Hint: Check your ID from the HR department")

# ===========================
# AFTER LOGIN - ATTENDANCE SECTION
# ===========================
else:
    teacher_info = db.get_teacher_by_id(st.session_state.teacher_id)
    
    if teacher_info is None:
        st.error("❌ Teacher data not found. Please login again.")
        st.session_state.logged_in = False
        st.rerun()
    
    # Welcome message
    st.success(f"👋 Welcome, **{teacher_info['name']}** ({teacher_info['department']} Department)")
    
    # ===========================
    # TIME VALIDATION
    # ===========================
    now = datetime.now()
    today = str(now.date())
    current_time = now.time()
    
    # Check if already marked
    already_marked = db.check_attendance_exists(st.session_state.teacher_id, today)
    
    if already_marked:
        today_record = db.get_attendance_for_date(st.session_state.teacher_id, today)
        st.info(f"✅ **Attendance Already Marked Today**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", today_record['status'])
        col2.metric("Check-in Time", today_record['checkin_time'] if today_record['checkin_time'] else "N/A")
        col3.metric("Distance", f"{today_record['distance_km']:.3f} km" if today_record['distance_km'] else "N/A")
    
    else:
        # Display attendance window status
        st.divider()
        st.subheader("⏰ Attendance Window")
        
        if current_time < START_TIME:
            time_until_start = datetime.combine(now.date(), START_TIME) - now
            minutes_until = time_until_start.seconds // 60
            st.warning(f"⏳ **Attendance window not started yet**\n\nOpens at **8:45 AM** (in {minutes_until} minutes)")
        
        elif current_time > END_TIME:
            st.error("🔒 **Attendance window closed**\n\nAttendance window was from **8:45 AM to 9:15 AM**")
            
            # Auto mark absent
            success, message = db.add_attendance(
                teacher_id=str(teacher_info["teacher_id"]),
                name=teacher_info["name"],
                department=teacher_info["department"],
                date=today,
                checkin_time="",
                latitude=None,
                longitude=None,
                status="Absent",
                distance_km=None
            )
            
            if success:
                st.info("📝 You have been automatically marked **Absent** for today.")
                st.rerun()
        
        else:
            # Within attendance window
            time_left = datetime.combine(now.date(), END_TIME) - now
            minutes_left = time_left.seconds // 60
            st.success(f"✅ **Attendance window is OPEN**\n\n⏱ Time remaining: **{minutes_left} minutes**")
            
            st.divider()
            st.subheader("📍 Mark Your Attendance")
            
            # GPS Input
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input(
                    "📍 Latitude",
                    format="%.6f",
                    min_value=-90.0,
                    max_value=90.0,
                    value=0.0,
                    help="Enter your current GPS latitude"
                )
            with col2:
                lon = st.number_input(
                    "📍 Longitude",
                    format="%.6f",
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    help="Enter your current GPS longitude"
                )
            
            # Show distance from campus if coordinates entered
            if lat != 0 or lon != 0:
                distance = calculate_distance(lat, lon, CAMPUS_LAT, CAMPUS_LON)
                
                st.markdown("---")
                st.markdown("#### 📏 Distance Check")
                
                if distance <= CAMPUS_RADIUS:
                    st.success(f"✅ **Within campus radius**\n\nDistance: **{distance:.3f} km** (Within {CAMPUS_RADIUS} km limit)")
                    will_be_present = True
                else:
                    st.error(f"❌ **Outside campus radius**\n\nDistance: **{distance:.3f} km** (Exceeds {CAMPUS_RADIUS} km limit)")
                    will_be_present = False
                
                st.markdown("---")
            
            # Mark Attendance Button
            if st.button("✔️ Mark Attendance", type="primary", use_container_width=True):
                
                if lat == 0 and lon == 0:
                    st.error("❌ Please enter valid GPS coordinates")
                else:
                    # Calculate distance and determine status
                    distance = calculate_distance(lat, lon, CAMPUS_LAT, CAMPUS_LON)
                    status = "Present" if distance <= CAMPUS_RADIUS else "Absent"
                    
                    # Save to database
                    success, message = db.add_attendance(
                        teacher_id=str(teacher_info["teacher_id"]),
                        name=teacher_info["name"],
                        department=teacher_info["department"],
                        date=today,
                        checkin_time=str(current_time.strftime("%H:%M:%S")),
                        latitude=lat,
                        longitude=lon,
                        status=status,
                        distance_km=round(distance, 3)
                    )
                    
                    if success:
                        if status == "Present":
                            st.success(f"✅ **Attendance Marked: {status}**")
                            st.balloons()
                        else:
                            st.warning(f"⚠️ **Attendance Marked: {status}** (Outside campus radius)")
                        
                        st.info(f"📍 Distance from campus: **{distance:.3f} km**")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
            
            # Quick GPS helper
            with st.expander("💡 How to get your GPS coordinates?"):
                st.markdown("""
                **On Mobile:**
                1. Open Google Maps
                2. Long press on your current location
                3. Your coordinates will appear at the top
                4. Copy and paste them here
                
                **Campus Location:**
                - Latitude: 17.3850
                - Longitude: 78.4867
                - Allowed Radius: 100 meters (0.1 km)
                """)
    
    # ===========================
    # DASHBOARD SECTION
    # ===========================
    st.divider()
    st.header("📊 Your Attendance Dashboard")
    
    # Get statistics
    stats = db.get_attendance_stats(st.session_state.teacher_id)
    
    if stats['total_days'] > 0:
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Attendance %",
                f"{stats['attendance_percent']}%",
                delta="Safe" if stats['attendance_percent'] >= 75 else "At Risk"
            )
        
        with col2:
            st.metric("Present Days", stats['present_days'], delta_color="normal")
        
        with col3:
            st.metric("Absent Days", stats['absent_days'], delta_color="inverse")
        
        with col4:
            st.metric("Total Days", stats['total_days'])
        
        # Risk Alert
        if stats['attendance_percent'] < 75:
            st.error("⚠️ **RISK ALERT:** Your attendance is below 75%. Please maintain regularity!")
        elif stats['attendance_percent'] < 85:
            st.warning("⚡ **WARNING:** Your attendance is below 85%. Try to improve!")
        else:
            st.success("✅ **EXCELLENT:** Your attendance is above 85%. Keep it up!")
        
        # Recent Attendance History
        st.markdown("---")
        st.subheader("📅 Recent Attendance History")
        
        teacher_records = db.get_teacher_attendance(st.session_state.teacher_id, limit=10)
        
        if len(teacher_records) > 0:
            # Prepare display dataframe
            display_df = teacher_records[[
                'date', 'checkin_time', 'status', 'distance_km'
            ]].copy()
            
            display_df.columns = ['Date', 'Check-in Time', 'Status', 'Distance (km)']
            
            # Add status emoji
            display_df['Status'] = display_df['Status'].apply(
                lambda x: f"{get_status_color(x)} {x}"
            )
            
            # Format distance
            display_df['Distance (km)'] = display_df['Distance (km)'].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            )
            
            # Format check-in time
            display_df['Check-in Time'] = display_df['Check-in Time'].apply(
                lambda x: x if x else "Auto Absent"
            )
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Export Option
        st.markdown("---")
        if st.button("📥 Export My Attendance to CSV"):
            all_records = db.get_teacher_attendance(st.session_state.teacher_id)
            csv = all_records.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{teacher_info['name'].replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("📭 **No attendance records yet**\n\nStart marking your attendance daily!")

# ===========================
# FOOTER
# ===========================
st.divider()
st.caption("🏫 Teacher Attendance Management System | Powered by GPS Technology")
