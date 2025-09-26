import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import json
import pandas as pd

# 訓練資料生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 驗證資料生成器
val_datagen = ImageDataGenerator(
    rescale=1./255
)

# 設置實際存在的資料夾路徑
train_dir = 'train'
val_dir = 'val'

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

# 包裝生成器，統一標籤類型
def generator_wrapper(generator):
    for images, labels in generator:
        yield images.astype(np.float32), labels.astype(np.float32)

# 設置 Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(train_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(val_generator),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# 計算類別權重並轉為 float32
classes = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(classes), y=classes
)
class_weights = {k: float(v) for k, v in enumerate(class_weights)}

# 定義殘差塊
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

# 建立 ResNet-34 模型
def build_resnet34(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # 增加 Dropout 層
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model

# 建立模型
model = build_resnet34(input_shape=(224, 224, 3), num_classes=1)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.00001),  # 調整學習率
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 訓練模型
history = model.fit(
    train_dataset,
    epochs=100,  # 增加訓練次數
    steps_per_epoch=len(train_generator),
    validation_data=val_dataset,
    validation_steps=len(val_generator),
    class_weight=class_weights,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    ]
)

# 儲存模型與結果
model.save('trained_model.keras')

with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

# 預測並分析結果
val_generator.reset()
predictions = (model.predict(val_dataset, steps=len(val_generator)) > 0.5).astype(int).flatten()
true_labels = val_generator.classes.astype(int)

print("分類報告:")
print(classification_report(true_labels, predictions, target_names=list(val_generator.class_indices.keys())))

print("混淆矩陣:")
print(confusion_matrix(true_labels, predictions))

# 計算準確度
accuracy = accuracy_score(true_labels, predictions)
print(f"準確度: {accuracy:.2f}")

# 保存結果到 CSV
results = {
    'True Labels': true_labels,
    'Predictions': predictions
}

df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)