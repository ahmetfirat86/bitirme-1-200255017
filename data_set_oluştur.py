import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Parametreler
num_records = 100000  # Kayıt sayısı
start_date = datetime(2023, 1, 1)

# Rastgele veri üretme fonksiyonları
def generate_plate():
    city_code = str(random.randint(1, 81)).zfill(2)
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPRSTUVYZ", k=random.randint(1, 3)))
    numbers = str(random.randint(100, 9999)).zfill(3)
    return f"{city_code} {letters} {numbers}"

colors = ['Beyaz', 'Siyah', 'Gri', 'Kırmızı', 'Mavi', 'Gümüş']
vehicle_types = ['Otomobil', 'Kamyon', 'Motosiklet', 'Otobüs']
lights = [1, 2, 3, 4]

# Veri listesi oluşturma
data = []
for i in range(num_records):
    # Zamanı rastgele artırarak ilerlet (ortalama her 5 dakikada bir araç)
    current_time = start_date + timedelta(seconds=random.randint(1, 31536000)) 
    
    data.append([
        current_time,
        1, # Junction ID
        random.choice(lights),
        generate_plate(),
        random.choice(colors),
        random.choice(vehicle_types)
    ])

# DataFrame oluşturma ve Kaydetme
df = pd.DataFrame(data, columns=['DateTime', 'Junction_ID', 'Light_ID', 'Plate', 'Vehicle_Color', 'Vehicle_Type'])
df = df.sort_values(by='DateTime') # Zaman sırasına diz
df.to_csv('kavsak_trafik_verisi.csv', index=False)
df["Sayac"] = 1# sayısal sütun ekledim


print("Veri seti başarıyla 'kavsak_trafik_verisi.csv' adıyla oluşturuldu!")