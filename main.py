import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

# Eğitim ve test için veri yolları
train_dir = "C:/Users/Umut/Desktop/Machine Learning Project/İşlenmiş Resimler/train"
test_dir = "C:/Users/Umut/Desktop/Machine Learning Project/İşlenmiş Resimler/test"

# Veri artırma işlemleri
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Eğitim ve test verisi oluşturma
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ResNet50 tabanlı model oluşturma
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.7),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Modeli derleme
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks tanımlama
checkpoint = ModelCheckpoint(
    "best_model.keras",  # Modeli .keras formatında kaydedecek
    monitor="val_loss",
    save_best_only=True
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Doğrulama kaybı 5 epoch boyunca iyileşmezse durdurur
    restore_best_weights=True
)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stopping]
)

# Test setinde performansı değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Kaydedilen en iyi modeli yükleme
best_model = tf.keras.models.load_model("best_model.keras")
print("En iyi model başarıyla yüklendi!")
