'''
evrişimsel sinir ağları ile beyin tümörü tespit etme uygulaması
'''

# Kütüphaneler

import tensorflow as tlf
import numpy as np
import keras.utils as image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train veri seti, veri üretici aşaması

# Over fittingi engelliyoruz
train_datagen = ImageDataGenerator(rescale=1./255, # Renkleri 1-0 arasına sokuyoruz
                                   shear_range=0.2, # Her görüntü belli bir şeklide eğilecek
                                   horizontal_flip=True) # Tüm verileri istediğimiz şekle sokuyoruz


training_dataset = train_datagen.flow_from_directory('veri_seti/training',
                                                     target_size = (64,64), # Boyutlar farkı o yüzden aynı voyuta çekiyoruz
                                                     batch_size=32,
                                                     class_mode='binary') # Verileri 1 ve 0 olarak (tümörlü, tümörsüz) şeklinde ayıracak

# test verisi
test_datagen = ImageDataGenerator(rescale = 1./255)

test_dataset = test_datagen.flow_from_directory('veri_seti/test',
                                                target_size= (64,64),
                                                batch_size = 32,
                                                class_mode='binary')

# Model oluşturma

model = tlf.keras.models.Sequential()

# Model katmanlarımızı oluşturalım

# 1. Evrişim Katmanı

model.add(tlf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))

# 1. Havuzlama Katmanı

model.add(tlf.keras.layers.MaxPool2D(pool_size=2, # 2x2
                                     strides=2, # Görüntü görüntü işleyecek ve bir görüntü bitince 2 piksel kayacak
                                     ))

# 2. Evrişim Katmanı

model.add(tlf.keras.layers.Conv2D(filters=64, # Daha karmaşık
                                  kernel_size=3, # Çekirdek boyutu
                                  activation='relu'))

# 2. Havuzlama Katmanı

model.add(tlf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 3. Evrişim Katmanı

model.add(tlf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))

# Düzleştireme Katmanı

model.add(tlf.keras.layers.Flatten())

# Çıkış Dense Katmanları

model.add(tlf.keras.layers.Dense(units = 128, activation='relu'))

model.add(tlf.keras.layers.Dense(units=1, activation='sigmoid'))

# modelimizi compile etme

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# model eğitme aşaması

model.fit(x=training_dataset, validation_data=test_dataset, epochs=100) # Normalde epochs 100
# Hem eğitiyor hem de tahmin alıyor

# model performans değerlendirmesi

loss, accuracy = model.evaluate(test_dataset)

print('Doğruluk Skoru:', accuracy)
print('Kayıp Fonksiyonu', loss)

# Deneme klasörü altındaki görsel üzerinden predict alalım

test_data = image.load_img('veri_seti/deneme/Y248.JPG',
                           target_size=(64,64))

# resmi numpy arraye dönüştürecek

test_data = image.img_to_array(test_data)

# resmin boyut eklme eklenen boyutun ilk sırada olması

test_data = np.expand_dims(test_data, axis=0)

# OUTPUT NESNE

output=model.predict(test_data)

training_dataset.class_indices

if output[0][0] == 1:
    print('Tümör tespit edildi')
    
else:
    print('Tümör Yok')










