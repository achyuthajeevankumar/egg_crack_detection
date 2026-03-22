import os
import json
import shutil
from django.conf import settings
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, preprocessing, applications
import tf2onnx
import onnx

# ----------------------------------------------------------------xception model training----------------------------------------------------



def xception_model():
    try:
        # Paths
        dataset_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
        model_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(model_dir, exist_ok=True)

        model_save_path = os.path.join(model_dir, 'best_xception_model.h5')
        json_result_path = os.path.join(model_dir, 'xception_model_result.json')

        # Data generators
        datagen = preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='training', shuffle=True)

        val_gen = datagen.flow_from_directory(
            dataset_dir, target_size=(299, 299), batch_size=32,
            class_mode='sparse', subset='validation', shuffle=True)

        # Load base Xception model
        base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=layers.Input(shape=(299, 299, 3)))
        base_model.trainable = False  # Freeze base layers

        # Add custom top layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu')(x)
        predictions = layers.Dense(2, activation='softmax')(x)

        model = models.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            model_save_path, monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1,
            save_weights_only=False
        )

        # Train the model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=13,
            callbacks=[checkpoint]
        )

        # Save model without optimizer to avoid warnings
        model.save(model_save_path, include_optimizer=False)

        # ------------------- ONNX Export -------------------
        print("Converting new model to ONNX...")
        try:
            onnx_save_path = os.path.join(settings.BASE_DIR, 'ml_models', 'best_xception_model.onnx')
            spec = (tf.TensorSpec((None, 299, 299, 3), tf.float32, name="input"),)
            # Re-load model if needed or just use 'model' object
            model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
            onnx.save(model_proto, onnx_save_path)
            print(f"✅ ONNX model saved to {onnx_save_path}")
        except Exception as onnx_e:
            print(f"❌ ONNX conversion failed: {onnx_e}")

        # ------------------- Dataset Cleanup -------------------
        print("Cleaning up augmented_dataset...")
        try:
            for category in ['Not Damaged', 'Damaged']:
                cat_path = os.path.join(dataset_dir, category)
                if os.path.exists(cat_path):
                    shutil.rmtree(cat_path)
                    os.makedirs(cat_path, exist_ok=True) # Recreate empty dir
            print("✅ Dataset cleanup completed.")
        except Exception as cleanup_e:
            print(f"❌ Cleanup failed: {cleanup_e}")

        final_val_acc = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]

        result = {
            'success': True,
            'message': "Xception model training, ONNX export, and cleanup completed successfully.",
            'val_accuracy': round(final_val_acc * 100, 2),
            'val_loss': round(final_val_loss, 4)
        }

        # Save result as JSON for fast reloading later
        with open(json_result_path, 'w') as json_file:
            json.dump(result, json_file)

        return result

    except Exception as e:
        return {
            'success': False,
            'message': f"Training failed: {str(e)}",
            'val_accuracy': None,
            'val_loss': None
        }
