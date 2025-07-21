# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load model and pipeline
# model = joblib.load("models/rental_price_model.pkl")
# pipeline = joblib.load("models/preprocessing_pipeline.pkl")

# st.set_page_config(page_title="Rental Price Predictor", layout="centered")
# st.title("🏠 Rental Price Recommendation System")
# st.markdown("Enter the property details to get a rental price prediction.")

# # Input form
# with st.form("rent_form"):
#     city = st.selectbox("City", ["Delhi", "Mumbai", "Pune"])
#     house_type = st.selectbox("Select House Type", [
#     "1 RK Studio Apartment", "2 BHK Independent Floor", "3 BHK Independent House", "2 BHK Apartment",
#     "3 BHK Apartment", "3 BHK Independent Floor", "4 BHK Independent Floor", "1 BHK Independent Floor",
#     "1 BHK Apartment", "8 BHK Independent Floor", "4 BHK Apartment", "6 BHK Independent Floor",
#     "2 BHK Independent House", "1 BHK Independent House", "5 BHK Independent Floor", "4 BHK Independent House",
#     "5 BHK Villa", "5 BHK Independent House", "7 BHK Independent Floor", "8 BHK Independent House",
#     "10 BHK Independent House", "7 BHK Independent House", "9 BHK Independent House", "8 BHK Villa",
#     "4 BHK Villa", "5 BHK Apartment", "6 BHK penthouse", "12 BHK Independent House", "3 BHK Villa",
#     "6 BHK Apartment", "1 BHK Villa", "6 BHK Villa", "2 BHK Villa", "6 BHK Independent House"])
    
#     location = st.selectbox("Select Location", sorted(["Kalkaji", "Mansarover Garden", "Uttam Nagar", "Model Town", 
#     "Sector 13 Rohini", "DLF Farms", "Laxmi Nagar", "Swasthya Vihar", "Janakpuri", "Pitampura", "Gagan Vihar", "Dabri", 
#     "Govindpuri Extension", "Paschim Vihar", "Vijay Nagar", "Vasant Kunj", "Safdarjung Enclave", "Hauz Khas", "Bali Nagar", 
#     "Rajouri Garden", "Shalimar Bagh", "Green Park", "Dr Mukherji Nagar", "Subhash Nagar", "DLF Phase 5", "Patel Nagar", "Jasola", 
#     "Dwarka Mor", "Kaushambi", "Surajmal Vihar", "Sector 4 Dwarka", "Sector 6 Dwarka", "14 Dwarka", "Sarvodaya Enclave", "Chattarpur", 
#     "Ramesh Nagar", "Mayur Vihar II", "Naraina", "Greater Kailash", "Chittaranjan Park", "Sector 19 Dwarka", "Sector 23 Dwarka",
#     "Lajpat Nagar III", "South Extension 2", "Sector-18 Dwarka", "Mansa Ram Park", "Gautam Nagar", "Sector 22 Dwarka", "Sheikh Sarai", 
#     "Govindpuri", "Sector 13 Dwarka", "Shanti Niketan", "Defence Colony", "Malviya Nagar", "Sector 23 Rohini", "Kirti Nagar", 
#     "Badarpur", "Lajpat Nagar I", "Sector-7 Rohini", "Sector 23B Dwarka", "Vikaspuri", "Sultanpur", "Sector 11 Dwarka", "Karampura", 
#     "Munirka", "Mahavir Enclave", "Greater Kailash 1", "Panchsheel Park", "Sector 12 Dwarka", "Sector 7 Dwarka", "Bindapur", 
#     "Alaknanda", "Sitapuri", "Dashrath Puri", "Manglapuri", "Sector 8 Dwarka", "Sector 5 Dwarka", "Kalyan Vihar", 
#     "Sector-B Vasant Kunj", "Green Park Extension", "Safdarjung Development Area", "Panchsheel Enclave", "Lajpat Nagar", 
#     "Shastri Nagar", "Jor Bagh", "Golf Links", "Vasant Vihar", "Anand Niketan", "Anand Lok", "East of Kailash", "Gulmohar Park", 
#     "Zone L Dwarka", "Raja Garden", "Kalu Sarai", "Tagore Garden Extension", "Saket", "Sector 2 Dwarka", "Geeta Colony", "Anand Vihar", 
#     "Ashok Nagar", "Dilshad Garden", "Gujranwala Town", "Sector 10 Dwarka", "Sector 16 Dwarka", "Palam", "Vikas Puri", 
#     "Masjid Moth Village", "Sewak Park", "Sagar Pur", "Kamla Nagar", "Ajmeri Gate", "Rajpur", "Jangpura", "Greater Kailash II", "Garhi", 
#     "Nizamuddin East", "Ansari Nagar West", "Sat Bari", "Central Ridge Reserve Forest", "New Friends Colony", "Sector 3 Dwarka", 
#     "Sector 9 Dwarka", "Moti Bagh", "Sainik Farm", "Karol Bagh", "Sarvpriya Vihar", "Uday Park", "Kailash Hills", "Geetanjali Enclave", 
#     "Soami Nagar", "Masoodpur", "Mehrauli", "Shakurpur", "Razapur Khurd", "Matiala", "Khirki Extension", "Sector 11 Rohini", 
#     "Sector 8", "Khushi Ram Park Delhi", "Dwarka Sector 17", "Preet Vihar", "Mayur Vihar Phase 1", "Rajpur Khurd Village", 
#     "Freedom Fighters Enclave", "Inderpuri", "Rajpur Khurd Extension", "Navjeevan Vihar", "Vishnu Garden", "Shahdara", "Patparganj", 
#     "IP Extension", "Punjabi Bagh", "AGCR Enclave", "Rajinder Nagar", "Krishna Nagar", "Niti Bagh", "Shakurbasti", "Sundar Nagar", "Sector 11", "Sector 16A Dwarka", "Guru Angad Nagar", "SECTOR 7 DWARKA NEW DELHI", "Tuglak Road", "Maharani Bagh", "Friends Colony", "Moti Nagar", "New Moti Nagar", "Shivalik", "Shahpur Jat Village", "Naraina Vihar", "Sector 1 Dwarka", "Tihar Village", "Nizamuddin West", "Ladosarai", "Haiderpur", "New Ashok Nagar", "Jangpura Extension", "Neb Sarai", "Sunder Nagar", "Mayur Vihar Phase II", "West End", "Ghitorni", "Prithviraj Road", "Malcha Marg", "Lodhi Road", "Tilak Marg", "B1 Block Paschim Vihar", "Sector 6 Rohini", "New Rajinder Nagar", "Aurungzeb Road", "Amrita Shergill Marg", "Babar Road", "Lodhi Gardens", "Lodhi Estate", "East Patel Nagar", "Sector 17 Dwarka", "B 5 Block", "New Rajendra Nagar", "Lajpat Nagar IV", "Prakash Mohalla Amritpuri", "Rohini Sector 9", "Old Rajender Nagar", "Mayur Vihar 2 Phase", "Dwarka 11 Sector", "Dwarka Sector 12", "Kishan Ganj", "I P Extension Patparganj", "Sector 14 Rohini", "Amritpuri", "Jamia Nagar", "Kailash Colony", "Prakash Mohalla", "Hemkunt Colony", "Chhatarpur Extension", "Lajpat Nagar II", "Connaught Place", "Uttam Nagar West", "Poorvi Pitampura", "Vaishali Dakshini Pitampura", "Uttari Pitampura", "Sector 9 Rohini", "Vasant Kunj Sector A", "SectorB Vasant Kunj", "Baljeet Nagar", "PANCHSHEEL VIHAR", "Lok Vihar", "Dakshini Pitampura", "Kohat Enclave", "Saraswati Vihar", "Prashant Vihar Sector 14", "Engineers Enclave Harsh Vihar", "Tarun Enclave", "Block MP Poorvi Pitampura", "Block PP Poorvi Pitampura", "Kapil Vihar", "Hauz Khas Enclave", "Westend DLF Chattarpur Farms", "Abul Fazal Enclave Jamia Nagar", "Fateh Nagar", "Pitampura Vaishali", "Block DP Poorvi Pitampura", "Block AP Poorvi Pitampura", "Block WP Poorvi Pitampura", "Sector 28 Rohini", "Rohini Sector 16", "Block A3", "Uttam Nagar East", "Mahipalpur", "Hari Nagar", "Tri Nagar", "Jhil Mil Colony", "Yojna Vihar", "Khanpur", "West Patel Nagar", "Ashok Vihar", "Aya Nagar", "Daheli Sujanpur", "Khirki Extension Panchsheel Vihar", "C R Park", "Chhattarpur Enclave Phase1", "Bawana", "West Punjabi Bagh", "Burari", "Shakti Nagar", "Sarita Vihar", "Sector 3 Rohini", "Mandawali", "Vinod Nagar East", "Sector 22 Rohini", "Sheikh Sarai Village", "Shakurpur Colony", "Nangloi", "Nehru Place", "Mayur Vihar", "Lajpat Nagar Vinoba Puri", "Block E Lajpat Nagar I", "Nawada", "Nangli Sakrawati", "Sector 34 Rohini", "Nirman Vihar", "Chattarpur Enclave", "Vasant Kunj Enclave", "Mayur Vihar 1 Extension", "Santnagar", "Kasturba Gandhi Marg", "Vipin Garden", "West Patel Nagar Road", "dda flat", "Ulwe", "Panvel", "Kandivali West", "Chembur", "Badlapur East", "Dombivali", "Bandra West", "Andheri East", "Bhayandar East", "Goregaon West", "Colaba", "Kalamboli", "Palghar", "Nerul", "Kandivali East", "Sion", "Andheri West", "Juhu", "Vasai", "Thakurli", "Powai", "Jogeshwari East", "Mira Road East", "Thane West", "Parel", "Prabhadevi", "Ghatkopar East", "Kalyan West", "Dadar East", "Vile Parle West", "Bhandup West", "Borivali East", "Mahalaxmi", "Kanjurmarg", "Seawoods", "Dombivali East", "Titwala", "Karanjade", "Girgaon", "Malabar Hill", "Dadar West", "Goregaon East", "Mahim", "Sanpada", "Mulund West", "Tardeo", "Malad East", "Borivali West", "Malad West", "Kharghar", "Jogeshwari West", "Kamothe", "Bandra East", "Worli", "Ville Parle East", "Mulund East", "Vikhroli", "Ghatkopar West", "Dombivli (West)", "Cumballa Hill", "Khar", "Vevoor", "Kanjurmarg East", "Lower Parel", "Marine Lines", "Hiranandani Estates", "Napeansea Road", "Nala Sopara", "Khar West", "Byculla", "Cuffe Parade", "Matunga", "Vasai East", "MATUNGA WEST", "Gamdevi", "Agripada", "Jacob Circle", "Wadala", "Sewri", "Belapur", "Airoli", "Deonar", "Dahisar West", "Dahisar", "Virar", "Ghansoli", "Boisar", "Kalyan East", "Churchgate", "Kurla", "Kurla East", "Govandi", "Tilak Nagar", "Santosh Nagar", "Matunga East", "Koper Khairane", "Palava", "Vashi", "Santacruz West", "Santacruz East", "Bhayandar West", "Bandra Kurla Complex", "Dharamveer Nagar", "Nalasopara West", "Kalwa", "Ville Parle West", "Antarli", "Shil Phata", "Amrut Nagar", "Virar East", "Sector 21 Kamothe", "Naigaon East", "Khardi", "Diva Gaon", "Diva", "Vasai West", "Koproli", "DN Nagar", "Ambernath West", "St Andrew Rd", "Balkum", "CBD Belapur East", "Borivali (West)", "Breach Candy", "Sector 21 Ghansoli", "Sector6 Kopar Khairane", "Bhandup East", "Parel Village", "Pali Hill", "Juhu Tara Rd", "Juhu Scheme", "Ghansoli Gaon", "Mahape", "Koparkhairane Station Road", "Sector 5 Ghansoli", "Mahim West", "Marol Andheri East", "Badlapur West", "Juhu Tara", "Rasayani", "Bhiwandi", "Versova", "Yari Road", "Colaba Post Office", "Aarya Chanakya Nagar", "Akurli Road", "Akurli Road Number 1", "Sector-15 Ghansoli", "Walkeshwar", "Sector-16 Koparkhairane", "Carter Road", "Marine Drive", "Kasar Vadavali", "Sakinaka Andheri East", "Ambernath East", "Lokhandwala", "Nallasopara W", "Seven Bunglow", "Babhai Naka", "Babhai", "Kastur Park", "Yogi Nagar", "Jayraj Nagar Near Yogi Nagar", "Rabale", "Virar West", "Devidas Cross Lane", "Devidas Rd", "Ghodbunder Road", "Kurla West", "Piramal Nagar Housing Society Road", "Taloja", "Suyog Nagar", "Hendre Pada", "Peddar Road", "Sector-19 Koper Khairane", "Majiwada Thane", "Karanjade Panvel", "Saphale", "Syndicate", "Sector5 Kopar Khairane", "Nalasopara East", "Gulmohar Road", "Dahisar East", "Vakola Santacuz E", "Dharavi", "Kapurbawadi", "Vakola", "Sector 12 Kharghar", "Saki Naka", "Lohegaon", "Anand Nagar", "Wagholi", "Sangamvadi", "Wadgaon Sheri", "Tathawade", "Hinjewadi", "Viman Nagar", "Undri", "Pimple Nilakh", "Kharadi", "Mohammed Wadi", "NIBM Annex Mohammadwadi", "Bavdhan", "Balewadi", "Chinchwad", "Hadapsar", "Gultekdi", "Karve Nagar", "Baner", "Sopan Baug", "Dhayari", "Koregaon Park", "Gahunje", "Yerawada", "Chakan", "Wakad", "Wanowrie", "Hingne Budrukh", "Talegaon Dabhade", "Chikhali", "Loni Kalbhor", "Pashan", "Sukhsagar Nagar", "Vishrantwadi", "Pimple Gurav", "Pimpri Chinchwad", "Mahalunge", "Vadgaon Budruk", "Ganga Dham", "Aundh", "Narhe", "Tingre Nagar", "Mundhwa", "Ghorpadi", "Kondhwa", "Dhanori", "Hinjawadi Village", "Pimple Saudagar", "Bibwewadi", "Kothrud", "Shivane", "Dhayari Phata", "Ambegaon Budruk", "Warje", "New Kalyani Nagar", "Kalas", "Sus", "Khadki", "Kalyani Nagar", "Patil Nager", "Ravet", "Wanwadi", "Lulla Nagar", "NIBM Annexe", "Thergaon", "Pimpri", "Erandwane", "Rahatani", "Katraj", "Shivaji Nagar", "Kargil Vijay Nagar", "Dhaygude Wada", "Charholi Budruk", "Sunarwadi", "Chandan Nagar", "Magarpatta", "Kothrud Depot Road", "Shirur", "Moshi", "Erandavana", "Shindenagar", "Khandve Nagar", "Vanaz Corner", "Khese Park", "NIBM", "Wadgaon Budruk", "Shivtirth Nagar", "Bhairav Nagar", "Vishnupuram Colony", "Punawale", "Alandi", "Deccan", "Vishal Nagar", "Bopodi", "Vishnu Dev Nagar", "Shikshak Nagar", "Bhusari Colony Right", "Sadashiv Peth", "Narayan Peth", "Fatima Nagar", "Swargate", "Shaniwar Peth", "Warje Malwadi", "Deccan Gymkhana", "Sanaswadi", "Porwal Road", "Senapati Bapat Road", "Manjari", "Fursungi", "Keshav Nagar", "Lonikand", "Nigdi", "Kalewadi", "Yewalewadi", "Prabhat Road", "Kesnand", "Vikas Nagar", "Law College Road", "Dighi", "Akurdi", "Dattavadi", "Model Colony", "Baramati", "Yamuna Nagar", "Pradhikaran Nigdi", "Maan", "Pisoli", "NIBM Road", "Handewadi", "Marunji", "Mamurdi", "Hingne Khurd", "Manjari Budruk", "Indryani Nagar", "Chimbali", "Old Kharadi Mundwa Road", "Bopkhel", "Baner Road", "Ubale Nagar", "Kaspate Wasti", "Sector 25 Pradhikaran", "Baner Pashan Link Road", "Someshwarwadi", "Landewadi", "Sakal Nagar", "Bhukum", "Nigdi Sector 26", "Cummins College Road", "Bhosari", "Parvati Darshan", "Bhugaon", "Kondhwa Budruk", "Karve Road Erandwane", "Bhelkenagar", "Mohan Nagar", "RMC Garden, Wagholi", "Somnath Nagar", "Veerbhadra Nagar", "BT Kawde", "Somatane Phata", "New DP Road", "Pratik Nagar Mohanwadi", "New Modikhana", "Siddartha Nagar", "Balewadi High Street", "Parihar Chowk", "Siddharth Nagar", "Sanewadi", "ITI Road", "Salunke Vihar", "Bakhori", "Sahakar Nagar", "Bibwewadi Kondhwa Road", "Mukund Nagar", "Padmavati", "Dhankawadi Road", "Dhanakwadi", "Bharati Vidyapeeth Campus", "Satara Road", "New Sangavi", "Hatti Chowk", "Teen Hatti Chowk Road", "SURESH NAGAR", "Dhankawadi", "Purnanagar", "Saswad", "Ambegaon Pathar", "Dapodi", "Borhade Wadi", "Jawalkar Nagar", "Maharashtra Housing Board", "Vishal Nagar Square New DP Road", "Hadapsar Gaon", "Kiwale", "Rambaug Colony", "Indrayani Nagar Sector 2", "Walhekarwadi Chinchwad", "Kolwadi", "Sant Tukaram Nagar", "Kalewadi Main", "Sai Nagar", "Charholi Khurd", "Sector No 28", "Chinchwade Nagar", "Gokhalenagar", "Talwade", "B T Kawde Road", "Tulaja Bhawani Nagar", "Tukaram Nagar", "Kasba Peth", "Aundh Gaon", "Old Sangvi", "Shikrapur", "Renuka Nagar", "Agalambe", "Ganj Peth", "Talegaon", "Pashan Sus Road", "Shewalewadi", "Kharadi", "Kasarwadi", "Taljai Temple Road", "Dighi Gaonthan", "Pan Card Club Road"
# ]))
    
#     house_size = st.number_input("House Size (in sqft)", min_value=100, max_value=10000, value=800)
#     num_bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
#     security_deposit = st.number_input("Security Deposit (in ₹)", value=50000)
#     latitude = st.number_input("Latitude", format="%.6f", value=19.076090)
#     longitude = st.number_input("Longitude", format="%.6f", value=72.877426)
#     submit = st.form_submit_button("Predict Rent")

# # On Submit
# if submit:
#     input_dict = {
#         'id': [0],  # Dummy ID, won't affect prediction
#         'city': [city],
#         'house_type': [house_type],
#         'house_size': [house_size],
#         'location' : [location],
#         'numBathrooms': [num_bathrooms],
#         'SecurityDeposit': [security_deposit],
#         'latitude': [latitude],
#         'longitude': [longitude]
#     }

#     input_df = pd.DataFrame(input_dict)

#     try:
#         processed = pipeline.transform(input_df)
#         prediction = model.predict(processed)[0]
#         st.success(f"🏷️ Estimated Rent: ₹{prediction:,.0f}")
#     except Exception as e:
#         st.error(f"❌ Error during prediction: {e}")


import streamlit as st
import joblib
import pandas as pd
import re

pipeline = joblib.load("models/preprocessing_pipeline.pkl")
model = joblib.load("models/rental_price_model.pkl")

st.set_page_config(page_title="Rental Price Predictor", layout="centered")

st.title("🏡 Rental Price Recommendation System")
st.markdown("Enter the property details to predict the monthly rent:")

def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d.]", "", value)
    try:
        return float(value)
    except:
        return None

house_type = st.selectbox("Select House Type", [
    "1 RK Studio Apartment", "2 BHK Independent Floor", "3 BHK Independent House", "2 BHK Apartment",
    "3 BHK Apartment", "3 BHK Independent Floor", "4 BHK Independent Floor", "1 BHK Independent Floor",
    "1 BHK Apartment", "8 BHK Independent Floor", "4 BHK Apartment", "6 BHK Independent Floor",
    "2 BHK Independent House", "1 BHK Independent House", "5 BHK Independent Floor", "4 BHK Independent House",
    "5 BHK Villa", "5 BHK Independent House", "7 BHK Independent Floor", "8 BHK Independent House",
    "10 BHK Independent House", "7 BHK Independent House", "9 BHK Independent House", "8 BHK Villa",
    "4 BHK Villa", "5 BHK Apartment", "6 BHK penthouse", "12 BHK Independent House", "3 BHK Villa",
    "6 BHK Apartment", "1 BHK Villa", "6 BHK Villa", "2 BHK Villa", "6 BHK Independent House"])
house_size = st.text_input("House Size (e.g., 1100 sqft)")
location = st.text_input("Location (Area/Colony)")
city = st.selectbox("City", ["Delhi", "Mumbai", "Pune"])
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")
numBathrooms = st.number_input("Number of Bathrooms", step=1, min_value=1)
security_deposit = st.text_input("Security Deposit (e.g., ₹25000)")

# --- Predict Button ---
if st.button("Predict Rent"):
    try:
        # Clean the data
        cleaned_data = {
            'id': [0],  # Dummy ID, won't affect prediction
            "house_type": house_type,
            "house_size": clean_numeric(house_size),
            "location": location,
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
            "numBathrooms": int(numBathrooms),
            "SecurityDeposit": clean_numeric(security_deposit)
        }

        input_df = pd.DataFrame([cleaned_data])

        # Apply pipeline and predict
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)[0]

        st.success(f"💰 Estimated Monthly Rent: ₹{round(prediction):,}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
