import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

n = 8000
hours = np.random.randint(0, 24, n)
days  = np.random.randint(0, 7, n)
temps = np.random.uniform(15, 45, n)
rains = np.random.randint(0, 2, n)
availability = np.clip(80 - 30*np.sin(np.pi*hours/12) + 5*rains - 0.3*(temps-30) + np.random.normal(0,5,n), 5, 100)
congestion = np.where(availability>65, 0, np.where(availability>35, 1, 2))
df_train = pd.DataFrame({"hour":hours,"day":days,"temperature":temps.round(1),"rain":rains,"availability_percentage":availability.round(2),"congestion_level":congestion})
df_train.to_csv("/home/claude/ev_project/ev_large_data.csv", index=False)
print(f"Training dataset: {len(df_train)} rows")

ALL_STATIONS = [
    ("Hitec City EV Hub","Telangana","Hyderabad",17.44745,78.37623,"DC Fast",8),
    ("Banjara Hills EV Station","Telangana","Hyderabad",17.41560,78.43470,"AC Slow",4),
    ("Gachibowli EV Point","Telangana","Hyderabad",17.44002,78.34890,"DC Fast",6),
    ("Jubilee Hills Charger","Telangana","Hyderabad",17.43180,78.40720,"AC Fast",4),
    ("Ameerpet Charge Zone","Telangana","Hyderabad",17.43740,78.44870,"DC Fast",6),
    ("Secunderabad EV Hub","Telangana","Hyderabad",17.43990,78.49830,"AC Slow",3),
    ("Kondapur EV Station","Telangana","Hyderabad",17.46070,78.35760,"DC Fast",8),
    ("Madhapur Charger","Telangana","Hyderabad",17.44860,78.39080,"AC Fast",4),
    ("Kukatpally EV Point","Telangana","Hyderabad",17.48490,78.41380,"DC Fast",6),
    ("LB Nagar Station","Telangana","Hyderabad",17.34900,78.55180,"AC Slow",3),
    ("Dilsukhnagar Charger","Telangana","Hyderabad",17.36860,78.52650,"DC Fast",4),
    ("Mehdipatnam EV Hub","Telangana","Hyderabad",17.39290,78.43230,"AC Fast",4),
    ("Uppal Charge Zone","Telangana","Hyderabad",17.40590,78.55920,"DC Fast",6),
    ("Miyapur EV Point","Telangana","Hyderabad",17.49480,78.35630,"DC Fast",8),
    ("Begumpet Station","Telangana","Hyderabad",17.44400,78.46700,"AC Fast",4),
    ("Shamshabad EV Hub","Telangana","Hyderabad",17.24070,78.42950,"DC Fast",10),
    ("Kompally Charger","Telangana","Hyderabad",17.54340,78.48470,"DC Fast",6),
    ("Nallagandla EV Station","Telangana","Hyderabad",17.44880,78.32640,"AC Fast",4),
    ("Manikonda Charge Point","Telangana","Hyderabad",17.39420,78.38710,"AC Slow",3),
    ("PVNR Expressway EV","Telangana","Hyderabad",17.37210,78.47830,"DC Fast",8),
    ("Hanamkonda Main Charger","Telangana","Hanamkonda",18.01390,79.56470,"DC Fast",6),
    ("Warangal Station EV Hub","Telangana","Warangal",17.97840,79.59410,"AC Slow",3),
    ("Kazipet EV Point","Telangana","Warangal",17.96800,79.62300,"DC Fast",4),
    ("Hunter Road Charger","Telangana","Hanamkonda",18.00630,79.57230,"AC Fast",4),
    ("NIT Warangal EV Station","Telangana","Warangal",17.98560,79.53080,"DC Fast",6),
    ("Subedari Charge Zone","Telangana","Hanamkonda",18.01980,79.55800,"AC Slow",3),
    ("Nakkalagutta EV Hub","Telangana","Hanamkonda",18.02210,79.55020,"DC Fast",4),
    ("Hanamkonda Bus Stand EV","Telangana","Hanamkonda",18.01540,79.56110,"AC Fast",4),
    ("Mulugu Road Station","Telangana","Warangal",17.99010,79.58670,"DC Fast",6),
    ("Wardhannapet Charger","Telangana","Warangal",17.96340,79.61020,"AC Slow",3),
    ("Karimnagar EV Hub","Telangana","Karimnagar",18.43770,79.12880,"DC Fast",4),
    ("Karimnagar City Charger","Telangana","Karimnagar",18.43410,79.13910,"AC Fast",3),
    ("Nizamabad EV Station","Telangana","Nizamabad",18.67200,78.09400,"DC Fast",4),
    ("Visakhapatnam EV Hub","Andhra Pradesh","Visakhapatnam",17.68670,83.21850,"DC Fast",8),
    ("Vizag Steel Gate Charger","Andhra Pradesh","Visakhapatnam",17.69870,83.28150,"AC Fast",4),
    ("Rushikonda EV Point","Andhra Pradesh","Visakhapatnam",17.77530,83.37420,"DC Fast",4),
    ("Vijayawada EV Hub","Andhra Pradesh","Vijayawada",16.50620,80.64800,"DC Fast",8),
    ("Benz Circle Charger","Andhra Pradesh","Vijayawada",16.51430,80.63180,"AC Fast",4),
    ("Guntur EV Station","Andhra Pradesh","Guntur",16.30670,80.43650,"DC Fast",4),
    ("Tirupati EV Hub","Andhra Pradesh","Tirupati",13.62880,79.41920,"DC Fast",6),
    ("Tirumala Charger","Andhra Pradesh","Tirupati",13.68340,79.34710,"AC Fast",4),
    ("Nellore EV Station","Andhra Pradesh","Nellore",14.44490,79.98640,"DC Fast",4),
    ("Kurnool EV Hub","Andhra Pradesh","Kurnool",15.82890,78.05500,"AC Fast",3),
    ("Koramangala EV Hub","Karnataka","Bengaluru",12.93520,77.62450,"DC Fast",8),
    ("Whitefield Charger","Karnataka","Bengaluru",12.96980,77.74990,"DC Fast",10),
    ("Indiranagar EV Station","Karnataka","Bengaluru",12.97840,77.64080,"AC Fast",6),
    ("Electronic City EV Hub","Karnataka","Bengaluru",12.83990,77.67700,"DC Fast",10),
    ("HSR Layout Charger","Karnataka","Bengaluru",12.91160,77.64740,"AC Slow",4),
    ("Marathahalli EV Point","Karnataka","Bengaluru",12.95620,77.70080,"DC Fast",6),
    ("Hebbal EV Station","Karnataka","Bengaluru",13.03560,77.59740,"DC Fast",6),
    ("Yeshwanthpur Charger","Karnataka","Bengaluru",13.02310,77.54180,"AC Fast",4),
    ("Jayanagar EV Hub","Karnataka","Bengaluru",12.92470,77.58340,"DC Fast",6),
    ("Mysuru EV Hub","Karnataka","Mysuru",12.29530,76.63920,"DC Fast",6),
    ("Mysuru Palace Charger","Karnataka","Mysuru",12.30520,76.65530,"AC Fast",4),
    ("Mangaluru EV Station","Karnataka","Mangaluru",12.87380,74.84200,"DC Fast",4),
    ("Hubli EV Hub","Karnataka","Hubli",15.35970,75.13490,"DC Fast",4),
    ("OMR EV Hub","Tamil Nadu","Chennai",12.90100,80.22790,"DC Fast",8),
    ("Anna Nagar Charger","Tamil Nadu","Chennai",13.08910,80.20990,"AC Fast",4),
    ("T Nagar EV Station","Tamil Nadu","Chennai",13.04180,80.23410,"DC Fast",6),
    ("Velachery Charger","Tamil Nadu","Chennai",12.97840,80.22080,"DC Fast",6),
    ("Adyar EV Hub","Tamil Nadu","Chennai",13.00130,80.25680,"AC Fast",4),
    ("Chennai Airport EV","Tamil Nadu","Chennai",12.99390,80.17930,"DC Fast",10),
    ("Coimbatore EV Hub","Tamil Nadu","Coimbatore",11.01680,76.95580,"DC Fast",6),
    ("Gandhipuram Charger","Tamil Nadu","Coimbatore",11.00740,76.96330,"AC Fast",4),
    ("Madurai EV Station","Tamil Nadu","Madurai",9.93200,78.11940,"DC Fast",4),
    ("Salem EV Hub","Tamil Nadu","Salem",11.66430,78.14590,"DC Fast",4),
    ("Trichy EV Station","Tamil Nadu","Trichy",10.79050,78.70470,"DC Fast",4),
    ("Andheri EV Hub","Maharashtra","Mumbai",19.11360,72.86970,"DC Fast",8),
    ("BKC Charger","Maharashtra","Mumbai",19.05960,72.86560,"AC Fast",6),
    ("Powai EV Station","Maharashtra","Mumbai",19.11760,72.90600,"DC Fast",6),
    ("Lower Parel EV Hub","Maharashtra","Mumbai",18.99780,72.83040,"DC Fast",8),
    ("Thane EV Point","Maharashtra","Mumbai",19.21830,72.97810,"AC Fast",4),
    ("Navi Mumbai Charger","Maharashtra","Mumbai",19.03300,73.02990,"DC Fast",6),
    ("Hinjewadi EV Hub","Maharashtra","Pune",18.59120,73.73890,"DC Fast",8),
    ("Koregaon Park Charger","Maharashtra","Pune",18.53620,73.89380,"AC Fast",4),
    ("Viman Nagar EV Station","Maharashtra","Pune",18.56790,73.91430,"DC Fast",6),
    ("Kothrud EV Point","Maharashtra","Pune",18.49990,73.81320,"AC Slow",3),
    ("Hadapsar EV Hub","Maharashtra","Pune",18.49980,73.92610,"DC Fast",6),
    ("Nashik EV Station","Maharashtra","Nashik",19.99730,73.79010,"DC Fast",4),
    ("Nagpur EV Hub","Maharashtra","Nagpur",21.14580,79.08820,"DC Fast",6),
    ("Aurangabad EV Station","Maharashtra","Aurangabad",19.87620,75.34330,"DC Fast",4),
    ("Connaught Place EV Hub","Delhi","New Delhi",28.63290,77.21980,"DC Fast",8),
    ("Dwarka EV Station","Delhi","New Delhi",28.59220,77.04150,"DC Fast",6),
    ("Saket EV Hub","Delhi","New Delhi",28.52130,77.21360,"DC Fast",8),
    ("Rohini Charger","Delhi","New Delhi",28.71330,77.12350,"AC Fast",4),
    ("Lajpat Nagar EV Point","Delhi","New Delhi",28.56640,77.24380,"AC Fast",4),
    ("Noida EV Hub","Uttar Pradesh","Noida",28.57080,77.32620,"DC Fast",8),
    ("Sector 62 Charger","Uttar Pradesh","Noida",28.62650,77.37440,"AC Fast",4),
    ("Gurugram EV Hub","Haryana","Gurugram",28.44390,77.02620,"DC Fast",10),
    ("Cyber City Charger","Haryana","Gurugram",28.49500,77.08960,"DC Fast",8),
    ("Faridabad EV Station","Haryana","Faridabad",28.40820,77.31350,"DC Fast",4),
    ("Ahmedabad EV Hub","Gujarat","Ahmedabad",23.02250,72.57140,"DC Fast",8),
    ("SG Highway Charger","Gujarat","Ahmedabad",23.05350,72.52430,"DC Fast",6),
    ("Satellite EV Station","Gujarat","Ahmedabad",23.03070,72.52580,"AC Fast",4),
    ("Surat EV Hub","Gujarat","Surat",21.17020,72.83110,"DC Fast",8),
    ("Vadodara EV Station","Gujarat","Vadodara",22.30720,73.18120,"DC Fast",6),
    ("Rajkot EV Hub","Gujarat","Rajkot",22.30390,70.80220,"DC Fast",4),
    ("Jaipur EV Hub","Rajasthan","Jaipur",26.91240,75.78730,"DC Fast",8),
    ("MI Road Charger","Rajasthan","Jaipur",26.92010,75.80270,"AC Fast",4),
    ("Jodhpur EV Station","Rajasthan","Jodhpur",26.28460,73.02430,"DC Fast",4),
    ("Udaipur EV Hub","Rajasthan","Udaipur",24.57130,73.69120,"DC Fast",4),
    ("Indore EV Hub","Madhya Pradesh","Indore",22.71960,75.85730,"DC Fast",6),
    ("AB Road Charger","Madhya Pradesh","Indore",22.72470,75.87990,"AC Fast",4),
    ("Bhopal EV Station","Madhya Pradesh","Bhopal",23.25950,77.41220,"DC Fast",6),
    ("Jabalpur EV Hub","Madhya Pradesh","Jabalpur",23.17960,79.93640,"DC Fast",4),
    ("Lucknow EV Hub","Uttar Pradesh","Lucknow",26.84990,80.94920,"DC Fast",8),
    ("Hazratganj Charger","Uttar Pradesh","Lucknow",26.84660,80.94630,"AC Fast",4),
    ("Agra EV Station","Uttar Pradesh","Agra",27.17670,78.00810,"DC Fast",6),
    ("Taj Mahal EV Hub","Uttar Pradesh","Agra",27.17510,78.04220,"DC Fast",8),
    ("Varanasi EV Hub","Uttar Pradesh","Varanasi",25.32110,83.00180,"DC Fast",4),
    ("Kanpur EV Station","Uttar Pradesh","Kanpur",26.44990,80.33190,"DC Fast",4),
    ("Park Street EV Hub","West Bengal","Kolkata",22.55270,88.35160,"DC Fast",8),
    ("Salt Lake Charger","West Bengal","Kolkata",22.57880,88.41650,"DC Fast",6),
    ("Howrah EV Station","West Bengal","Kolkata",22.58380,88.31540,"AC Fast",4),
    ("New Town EV Hub","West Bengal","Kolkata",22.58860,88.47120,"DC Fast",6),
    ("Chandigarh EV Hub","Chandigarh","Chandigarh",30.73310,76.77940,"DC Fast",8),
    ("Sector 17 Charger","Chandigarh","Chandigarh",30.74150,76.78430,"AC Fast",4),
    ("Ludhiana EV Station","Punjab","Ludhiana",30.90100,75.85730,"DC Fast",6),
    ("Amritsar EV Hub","Punjab","Amritsar",31.63400,74.87230,"DC Fast",6),
    ("Kochi EV Hub","Kerala","Kochi",9.93160,76.26720,"DC Fast",8),
    ("Marine Drive Charger","Kerala","Kochi",9.96640,76.28220,"AC Fast",4),
    ("Thiruvananthapuram EV Hub","Kerala","Thiruvananthapuram",8.52420,76.93880,"DC Fast",6),
    ("Kozhikode EV Station","Kerala","Kozhikode",11.24960,75.78010,"DC Fast",4),
    ("Panaji EV Hub","Goa","Panaji",15.49380,73.82580,"DC Fast",4),
    ("Margao EV Station","Goa","Margao",15.27360,73.95770,"DC Fast",4),
    ("Bhubaneswar EV Hub","Odisha","Bhubaneswar",20.29600,85.82450,"DC Fast",6),
    ("Cuttack EV Station","Odisha","Cuttack",20.46120,85.87950,"DC Fast",4),
    ("Ranchi EV Hub","Jharkhand","Ranchi",23.34490,85.30960,"DC Fast",4),
    ("Guwahati EV Hub","Assam","Guwahati",26.14450,91.74620,"DC Fast",6),
    ("GS Road Charger","Assam","Guwahati",26.13080,91.79620,"AC Fast",4),
]

busy_levels = ["Low","Medium","High"]
busy_weights = [0.4, 0.35, 0.25]
rows = []
for s in ALL_STATIONS:
    busy = random.choices(busy_levels, weights=busy_weights)[0]
    slots = s[6]
    if busy=="Low": occupied = random.randint(0, max(1,slots//3))
    elif busy=="Medium": occupied = random.randint(slots//3, 2*slots//3)
    else: occupied = random.randint(2*slots//3, slots)
    avail = slots - occupied
    wait = 0 if avail>0 else random.randint(5,30)
    rows.append({"station_name":s[0],"state":s[1],"city":s[2],"latitude":s[3],"longitude":s[4],"charger_type":s[5],"total_slots":slots,"occupied_slots":occupied,"available_slots":avail,"busy_level":busy,"wait_time_min":wait})

stations_df = pd.DataFrame(rows)
stations_df.to_csv("/home/claude/ev_project/india_ev_stations.csv", index=False)
print(f"Stations: {len(stations_df)} across {stations_df['city'].nunique()} cities, {stations_df['state'].nunique()} states")
