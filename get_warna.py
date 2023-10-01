import cv2

def get_pixel_values(image_path):
    # Muat gambar
    image = cv2.imread(image_path)

    # Split gambar menjadi saluran warna
    b, g, r = cv2.split(image)

    # Gabungkan saluran warna untuk mendapatkan nilai piksel
    pixels = cv2.merge((r, g, b))

    return pixels

# Path gambar
image_path = "img/cabe.png"

# Dapatkan nilai piksel
pixels = get_pixel_values(image_path)

# Cetak nilai RGB setiap piksel
for pixel in pixels:
    print(f"R: {pixel[0]}, G: {pixel[1]}, B: {pixel[2]}")
