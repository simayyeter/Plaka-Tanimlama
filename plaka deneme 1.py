import cv2
import pytesseract
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\Users\simay\Tesseract-OCR\tesseract.exe"


resim = cv2.imread("./plaka.png")


if resim is None:
    print("❌ Resim bulunamadı, yolu kontrol et!")
else:
    print("✅ Resim basarıyla okundu.")


plaka_resmi = resim[230:320, 165:460]


gri_resim = cv2.cvtColor(plaka_resmi, cv2.COLOR_BGR2GRAY)


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gri_resim = clahe.apply(gri_resim)


_, temiz_resim = cv2.threshold(gri_resim, 110, 255, cv2.THRESH_BINARY)


custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
sonuc = pytesseract.image_to_string(temiz_resim, lang='eng', config=custom_config)


temiz_sonuc = re.sub(r'[^A-Z0-9 ]', '', sonuc.upper())


match = re.match(r'(\d{2})([A-Z]{1,3})(\d{2,4})', temiz_sonuc.replace(" ", ""))


if match:
    plaka_duzgun = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    print("Plaka:", plaka_duzgun)
else:
    print("Plaka:", temiz_sonuc)


cv2.imwrite("temiz_sonuc.png", temiz_resim)


cv2.imshow("Plaka", plaka_resmi)
cv2.waitKey(0)
cv2.destroyAllWindows()



