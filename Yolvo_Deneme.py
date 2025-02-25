import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("yolov8n.pt")
model.to("cuda")


cap = cv2.VideoCapture(0)

GERCEK_GENISLIK_INSAN = 50  
GERCEK_GENISLIK_TELEFON = 7  
GERCEK_GENISLIK_KEDI = 25  
GERCEK_GENISLIK_AT = 30  
ODAK_UZUNLUK = 700  
GERCEK_GENISLIK_ARABA = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            genislik = x2 - x1

            if cls == 0:  # İnsan
                gercek_genislik = GERCEK_GENISLIK_INSAN
                nesne_adi = "Yaya"
                renk = (0, 255, 0)  # Yeşil
            elif cls == 67:  # Telefon
                gercek_genislik = GERCEK_GENISLIK_TELEFON
                nesne_adi = "Telefon"
                renk = (0, 0, 255)  # Kırmızı
            elif cls == 15:  # Kedi
                gercek_genislik = GERCEK_GENISLIK_KEDI
                nesne_adi = "Kedi"
                renk = (255, 0, 0)
            elif cls == 17:  # Köpek
                gercek_genislik = GERCEK_GENISLIK_AT
                nesne_adi = "At"
                renk = (0, 255, 255)
            elif cls == 2:
                gercek_genislik = GERCEK_GENISLIK_ARABA
                nesne_adi = "Araba"
                renk = (100,200,100)
            else:
                continue

            mesafe = (gercek_genislik * ODAK_UZUNLUK) / genislik
            mesafe = round(mesafe, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)

            text = f"{nesne_adi} - {mesafe/100} metre"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 2)

    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




"""
İnsan: 0
Bisiklet: 1
Araba: 2
Motosiklet: 3
Uçak: 4
Otobüs: 5
Tren: 6
Kamyon: 7
Tekne: 8
Trafik ışığı: 9
Yangın söndürme direği: 10
Dur işareti: 11
Parkmetre: 12
Bank: 13
Kuş: 14
Kedi: 15
Köpek: 16
At: 17
İnek: 18
Fil: 19
Ayı: 20
Zebra: 21
Gergedan: 22
Sırt çantası: 24
Şemsiye: 25
Çanta: 26
Frizbi: 27
Kaykay: 28
Tenis raketi: 29
Beyzbol sopası: 30
Beyzbol topu: 31
Kutu: 32
Muzip: 33
Kedi (yeni bir etiket olarak eklenmiş olabilir, ancak genelde 15 kullanılır): 15
Telefon (cep telefonu): 67
"""