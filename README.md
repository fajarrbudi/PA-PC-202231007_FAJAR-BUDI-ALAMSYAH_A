# PA-PC-202231007_FAJAR-BUDI-ALAMSYAH_A

Nama  : Fajar Budi Alamsyah <br>
NIM  : 202231007 <br>
Kelas : A <br>
Project : Filtering <br>

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


Pada tahap ini, kita menerapkan filter rata-rata pada gambar skala abu-abu menggunakan operasi konvolusi. 
Filter rata-rata bekerja dengan menghitung rata-rata nilai piksel di sekitar setiap piksel dalam jendela 3x3. 
Implementasi manual dari filter ini dilakukan dengan menggunakan dua loop for yang melintasi setiap piksel dalam gambar. <br>

1. Looping Melintasi Setiap Piksel: <br>
Dua loop for digunakan untuk melintasi setiap piksel dalam gambar, kecuali tepi luar, karena kita memerlukan piksel tetangga di sekitar setiap piksel yang diproses. <br>
```
for baris in range (0, m1-1):
    for kolom in range (0, n1-1):
```

2. Menghitung Jumlah Nilai Piksel Tetangga: <br>
Untuk setiap piksel pada posisi (a1, b1), kita menghitung jumlah dari nilai piksel di sekitar dalam jendela 3x3. <br>
```
a1 = baris
b1 = kolom
jumlah = copy_gray[a1-1, b1-1] + copy_gray[a1-1, b1] + copy_gray[a1-1, b1+1] +\
    copy_gray[a1, b1-1] + copy_gray[a1, b1] + copy_gray[a1, b1+1] +\
    copy_gray[a1+1, b1-1] + copy_gray[a1+1, b1] + copy_gray[a1+1, b1+1]
```

3. Menghitung Rata-Rata: <br>
Setelah menghitung jumlah dari nilai-nilai piksel tetangga, nilai rata-rata dihitung dengan membagi jumlah
tersebut dengan 9 (karena ada 9 piksel dalam jendela 3x3). Nilai rata-rata ini kemudian disimpan dalam array
mean_filter pada posisi yang sama. <br>
```
mean_filter[a1, b1] = 1/9 * jumlah
```
<br>


Hasil dari penerapan filter rata-rata ini adalah citra yang lebih halus dan lebih sedikit noise, 
karena filter rata-rata cenderung mengurangi variasi intensitas piksel yang tajam dengan menghaluskan perubahan antara piksel-piksel tetangga. <br>

```
fig, axis = plt.subplots(1, 2, figsize=(10,10))
ax = axis.ravel()

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Image')

ax[1].imshow(mean_filter, cmap='gray')
ax[1].set_title('Mean Filtered')
plt.show()
```

Pada tahap terakhir dari persiapan citra ini, kita akan menampilkan gambar asli dan gambar hasil penyaringan rata-rata berdampingan untuk memvisualisasikan 
efek dari operasi penyaringan. Dengan menggunakan Matplotlib, kita membuat subplot dengan dua kolom dan satu baris menggunakan fig, axis = plt.subplots(1, 2, 
figsize=(10, 10)). Variabel ax digunakan untuk meratakan array sumbu, sehingga mudah diakses dengan indeks. <br>

Gambar asli dalam skala abu-abu ditampilkan di subplot pertama menggunakan ax[0].imshow(gray_image, cmap='gray') dan diberi judul "Original Image" 
dengan ax[0].set_title('Original Image'). Gambar hasil penyaringan rata-rata ditampilkan di subplot kedua menggunakan ax[1].imshow(mean_filter, cmap='gray') 
dan diberi judul "Mean Filtered" dengan ax[1].set_title('Mean Filtered'). Terakhir, plt.show() digunakan untuk menampilkan plot tersebut. Langkah ini memungkinkan 
kita untuk dengan jelas melihat perbedaan antara citra asli dan citra yang telah mengalami penyaringan rata-rata, sehingga membantu dalam analisis dan pemahaman 
efek dari teknik penyaringan yang digunakan. <br>

## MENAMPILKAN SEMUA GAMBAR
```
ig, axs = plt.subplots(2, 2, figsize=(15, 10))
# Menampilkan gambar di setiap subplot
axs[0, 0].imshow(img)
axs[0, 0].set_title('Citra Asli')

axs[0, 1].imshow(img_median_after)
axs[0, 1].set_title('Median FIltered')

axs[1, 0].imshow(gray_image, cmap='gray')
axs[1, 0].set_title('Original Imgae')

axs[1, 1].imshow(mean_filter, cmap='gray')
axs[1, 1].set_title('Mean Filtered')

plt.subplots_adjust(hspace=0.5)
```

Kemudian  akan menampilkan empat gambar berbeda berdampingan dalam satu plot untuk memvisualisasikan berbagai 
operasi penyaringan dan konversi yang telah dilakukan. Dengan menggunakan Matplotlib, kita membuat subplot dengan 
dua baris dan dua kolom menggunakan fig, axs = plt.subplots(2, 2, figsize=(15, 10)). Gambar-gambar ini akan ditampilkan 
di setiap subplot untuk memudahkan perbandingan. <br>

Gambar asli berwarna ditampilkan di subplot pertama dengan axs[0, 0].imshow(img) dan diberi judul "Citra Asli" menggunakan 
axs[0, 0].set_title('Citra Asli'). Gambar hasil penyaringan median ditampilkan di subplot kedua dengan axs[0, 1].imshow(img_median_after) 
dan diberi judul "Median Filtered" menggunakan axs[0, 1].set_title('Median Filtered'). <br>

Selanjutnya, gambar skala abu-abu asli ditampilkan di subplot ketiga dengan axs[1, 0].imshow(gray_image, cmap='gray') dan diberi 
judul "Original Image" menggunakan axs[1, 0].set_title('Original Image'). Terakhir, gambar hasil penyaringan rata-rata 
ditampilkan di subplot keempat dengan axs[1, 1].imshow(mean_filter, cmap='gray') dan diberi judul "Mean Filtered" 
menggunakan axs[1, 1].set_title('Mean Filtered'). <br>

Dengan mengatur ruang antara subplot menggunakan plt.subplots_adjust(hspace=0.5), kita memastikan bahwa judul-judul gambar 
tidak saling tumpang tindih, sehingga plot keseluruhan menjadi lebih mudah dibaca. <br>

---
# TEORI PENDUKUNG

1. OpenCV dan Pembacaan Gambar <br>
Open Source Computer Vision Library (OpenCV) adalah pustaka perangkat lunak yang dirancang untuk aplikasi visi
komputer dan pembelajaran mesin. OpenCV menyediakan berbagai fungsi untuk memanipulasi gambar dan video, termasuk
pembacaan, penulisan, dan pemrosesan gambar. Dalam penelitian ini, fungsi cv2.imread digunakan untuk membaca gambar
dari file dan menyimpannya sebagai array NumPy. Array ini memungkinkan manipulasi numerik yang efisien terhadap gambar.
Menurut Bradski dan Kaehler (2008), OpenCV merupakan alat yang sangat efektif dalam pemrosesan gambar karena kemampuannya
yang luas dan performanya yang tinggi. <br>

2. Konversi Ruang Warna <br>
Ruang warna adalah model matematika yang menggambarkan cara warna diwakili dalam gambar digital.
Gambar yang dibaca oleh OpenCV secara default berada dalam format BGR (Blue, Green, Red), sementara
banyak pustaka pemrosesan gambar lain seperti Matplotlib menggunakan format RGB (Red, Green, Blue).
Untuk mengkonversi gambar dari BGR ke RGB, digunakan fungsi cv2.cvtColor. Menurut Gonzalez dan Woods (2018),
konversi ruang warna adalah langkah penting dalam memastikan gambar ditampilkan dengan warna yang benar saat
menggunakan alat visualisasi yang berbeda. <br>

3. Filter Median <br>
Filter median adalah teknik penyaringan non-linear yang digunakan untuk menghilangkan noise dari gambar.
 Filter ini bekerja dengan menggantikan setiap piksel dengan median dari piksel-piksel tetangganya. Fungsi
 cv2.medianBlur digunakan untuk menerapkan filter median pada gambar dengan ukuran kernel yang ditentukan.
 Menurut Jain (1989), filter median sangat efektif dalam menghilangkan noise sementara tetap mempertahankan
tepi, menjadikannya pilihan yang baik untuk pra-pemrosesan gambar dalam berbagai aplikasi visi komputer. <br>

4. Visualisasi Gambar dengan Matplotlib <br>
Matplotlib adalah pustaka Python untuk membuat visualisasi data 2D, termasuk gambar. Fungsi plt.subplots
digunakan untuk membuat beberapa plot dalam satu figur, memungkinkan perbandingan visual yang mudah antara
gambar asli dan gambar yang telah difilter. Fungsi imshow digunakan untuk menampilkan gambar, dengan kolormap
'gray' digunakan untuk gambar skala abu-abu. Menurut Hunter (2007), Matplotlib adalah alat yang sangat
fleksibel dan kuat untuk visualisasi data, termasuk dalam pemrosesan gambar. <br>

5. Konversi ke Skala Abu-abu <br>
Konversi gambar berwarna ke skala abu-abu adalah langkah penting dalam banyak algoritma pemrosesan gambar.
 Gambar skala abu-abu hanya memiliki satu saluran warna yang mewakili intensitas cahaya, sehingga lebih
sederhana dan lebih cepat untuk diproses dibandingkan dengan gambar berwarna. Fungsi cv2.cvtColor digunakan
untuk mengonversi gambar ke skala abu-abu. <nr>

6. Filter rata-rata (median filter) <br>
Filter rata-rata adalah filter linear yang digunakan untuk menghaluskan gambar dengan menghitung rata-rata
dari piksel-piksel tetangganya. Filter ini efektif dalam mengurangi noise namun dapat mengaburkan tepi.
Dalam penelitian ini, filter rata-rata diterapkan secara manual dengan menghitung jumlah nilai piksel
dalam jendela 3x3 di sekitar setiap piksel dan membaginya dengan 9. <br>

7. Visualisasi Filtering <br>
Visualisasi efek penyaringan sangat penting untuk memahami dampak dari berbagai teknik pemrosesan gambar.
Empat gambar berbeda ditampilkan dalam satu plot menggunakan Matplotlib: gambar asli berwarna, gambar hasil penyaringan median,
gambar skala abu-abu asli, dan gambar hasil penyaringan rata-rata. Dengan mengatur ruang antara subplot menggunakan plt.subplots_adjust,
visualisasi menjadi lebih jelas dan mudah dibaca. <br>
