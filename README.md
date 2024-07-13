# PA-PC-202231007_FAJAR-BUDI-ALAMSYAH_A

Nama  : Fajar Budi Alamsyah
NIM  : 202231007
Kelas : A
Project : Filtering

---
# TAHAPAN MENGOLAH CITRA

##  Mempersiapkan citra yang akan di olah <br>
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("foto.jpg")
img.shape
```
Pada tahap persiapan citra, langkah pertama yang dilakukan adalah membaca
file gambar menggunakan OpenCV, seperti yang ditunjukkan oleh kode img = cv2.imread("foto.jpg").
Fungsi ini memuat gambar yang disimpan dalam berkas "foto.jpg" dan menyimpannya dalam variabel img
sebagai array NumPy. Kemudian, dengan menggunakan img.shape, kita dapat memperoleh dimensi gambar tersebut,
yang mencakup informasi mengenai tinggi, lebar, dan jumlah kanal warna (misalnya, 3 untuk gambar berwarna RGB).
Langkah ini penting untuk memastikan bahwa gambar telah dimuat dengan benar dan untuk mendapatkan informasi awal mengenai
struktur gambar yang akan digunakan dalam proses pemrosesan selanjutnya. <br>

```
[baris, kolom] = img.shape[:2]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

Langkah-langkah berikut dalam persiapan citra melibatkan pengolahan dan konversi format warna gambar. Setelah gambar 
berhasil dibaca, dimensi gambar diambil dengan img.shape[:2], yang menyimpan nilai jumlah baris (tinggi) dan kolom (lebar) 
dari gambar tersebut. Selanjutnya, gambar yang semula dalam format warna BGR (Blue, Green, Red) dikonversi ke format RGB (Red, Green, Blue) 
menggunakan fungsi cv2.cvtColor(img, cv2.COLOR_BGR2RGB). Konversi ini diperlukan karena OpenCV secara default membaca gambar dalam format BGR, 
sementara banyak pustaka pemrosesan gambar lain, seperti Matplotlib, menggunakan format RGB. Langkah ini memastikan bahwa gambar ditampilkan 
dengan warna yang benar dan siap untuk proses analisis atau pemrosesan lebih lanjut. <br>

## Median Filtering <br>
```
img_median = img.copy()
img_median_after = cv2.medianBlur(img_median, 5)
```
Pada tahap berikutnya dari persiapan citra, dilakukan operasi penyaringan untuk mengurangi noise dan memperhalus gambar.
Langkah pertama adalah membuat salinan dari gambar asli dengan `img_median = img.copy()`, yang menyimpan data gambar dalam variabel
`img_median`. Kemudian, operasi penyaringan median diterapkan pada gambar ini menggunakan fungsi `cv2.medianBlur(img_median, 5)`.
 Penyaringan median bekerja dengan mengganti setiap piksel dalam gambar dengan median dari piksel-piksel tetangganya dalam jendela ukuran
 tertentu (dalam hal ini, ukuran jendela adalah 5x5 piksel).
 Hasil dari operasi ini disimpan dalam variabel `img_median_after`. Penyaringan median sangat efektif untuk menghilangkan noise 
 salt-and-pepper tanpa mengaburkan tepi-tepi penting dalam gambar, sehingga hasil akhir citra lebih halus dan bersih. <br>

```
fig, axis = plt.subplots(1, 2, figsize=(10, 10))
ax = axis.ravel()

ax[0].imshow(img)
ax[0].set_title('Citra Asli')

ax[1].imshow(img_median_after)
ax[1].set_title('Median Filtered')
plt.show()

```
Untuk menampilkan gambar asli dan gambar hasil penyaringan median berdampingan untuk memvisualisasikan efek dari operasi penyaringan. 
Dengan menggunakan Matplotlib, kita membuat subplot dengan dua kolom dan satu baris menggunakan fig, axis = plt.subplots(1, 2, 
figsize=(10, 10)). Variabel ax digunakan untuk meratakan array sumbu, sehingga mudah diakses dengan indeks. <br>

Gambar asli ditampilkan di subplot pertama menggunakan ax[0].imshow(img) dan diberi judul "Citra Asli" dengan ax[0].set_title('Citra Asli'). 
Gambar hasil penyaringan median ditampilkan di subplot kedua menggunakan ax[1].imshow(img_median_after) dan diberi judul "Median Filtered" 
dengan ax[1].set_title('Median Filtered'). Terakhir, plt.show() digunakan untuk menampilkan plot tersebut. Langkah ini memungkinkan kita 
untuk dengan jelas melihat perbedaan antara citra asli dan citra yang telah mengalami penyaringan median, sehingga membantu dalam analisis 
dan pemahaman efek dari teknik penyaringan yang digunakan. <br>

## Mean Filtering <br>
```
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
copy_gray = gray_image.copy().astype(float)

m1, n1 = copy_gray.shape
mean_filter = np.empty([m1,n1])

print("Shape copy citra abu : ", copy_gray.shape)
print("Shape output citra abu : ", mean_filter.shape)

print("m1 : ", m1)
print("n1 : ", n1)
print()
```

Pada tahap ini, dilakukan beberapa langkah tambahan dalam persiapan dan pemrosesan citra. Pertama, kita mengonversi 
gambar berwarna (RGB) menjadi gambar skala abu-abu menggunakan fungsi cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), yang menyimpan 
hasilnya dalam variabel gray_image. Konversi ke skala abu-abu ini seringkali diperlukan dalam berbagai algoritma pemrosesan 
citra yang bekerja dengan intensitas piksel tunggal. <br>

Selanjutnya, kita membuat salinan dari gambar skala abu-abu dan mengubah tipenya menjadi float dengan copy_gray = 
gray_image.copy().astype(float), yang mungkin diperlukan untuk operasi pemrosesan yang melibatkan perhitungan numerik lebih lanjut. <br>

Dimensi gambar skala abu-abu disimpan dalam variabel m1 dan n1 dengan menggunakan copy_gray.shape. Kemudian, kita membuat 
sebuah array kosong mean_filter dengan dimensi yang sama, yang akan digunakan untuk menyimpan hasil dari operasi filter rata-rata di langkah-langkah berikutnya. <br>

```

```
Pada tahap ini, kita menerapkan filter rata-rata pada gambar skala abu-abu menggunakan operasi konvolusi. 
Filter rata-rata bekerja dengan menghitung rata-rata nilai piksel di sekitar setiap piksel dalam jendela 3x3. 
Implementasi manual dari filter ini dilakukan dengan menggunakan dua loop for yang melintasi setiap piksel dalam gambar.







